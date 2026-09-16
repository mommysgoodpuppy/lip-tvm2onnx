from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import tempfile
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

PARAMS_MAGIC = 0xF7E58D4F05049CB7
NDARRAY_MAGIC = 0xDD5E40F096B4A13F

EXPECTED_SHAPES = {
    "conv1_weight": (20, 2, 5, 5),
    "conv1_bias": (20,),
    "conv2_weight": (48, 20, 5, 5),
    "conv2_bias": (48,),
    "conv3_weight": (64, 48, 3, 3),
    "conv3_bias": (64,),
    "fc5_ft_weight": (500, 25_600),
    "fc5_ft_bias": (500,),
    "fc6_10_weight": (30, 500),
    "fc6_10_bias": (30,),
}


class ConversionError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelSource:
    graph: bytes
    params: bytes
    description: str


class Reader:
    def __init__(self, data: bytes):
        self.data = data
        self.position = 0

    def take(self, size: int) -> memoryview:
        if size < 0 or size > len(self.data) - self.position:
            raise ConversionError("truncated parameter file")
        start = self.position
        self.position += size
        return memoryview(self.data)[start : start + size]

    def u64(self) -> int:
        return struct.unpack("<Q", self.take(8))[0]

    def u32(self) -> int:
        return struct.unpack("<I", self.take(4))[0]

    def u16(self) -> int:
        return struct.unpack("<H", self.take(2))[0]

    def u8(self) -> int:
        return self.take(1)[0]

    def i64(self) -> int:
        return struct.unpack("<q", self.take(8))[0]


def parse_params(data: bytes) -> dict[str, np.ndarray]:
    reader = Reader(data)
    if reader.u64() != PARAMS_MAGIC:
        raise ConversionError("invalid parameter-file magic")
    reader.u64()
    count = reader.u64()
    if count > 1_024:
        raise ConversionError(f"unreasonable tensor count: {count}")

    names = []
    for _ in range(count):
        size = reader.u64()
        if size > 4_096:
            raise ConversionError(f"unreasonable tensor name length: {size}")
        try:
            name = bytes(reader.take(size)).decode()
        except UnicodeDecodeError as exc:
            raise ConversionError("invalid tensor name") from exc
        if name in names:
            raise ConversionError(f"duplicate tensor name: {name}")
        names.append(name)

    array_count = reader.u64()
    if array_count != count:
        raise ConversionError(
            f"tensor-name count {count} does not match array count {array_count}"
        )

    tensors = {}
    for name in names:
        if reader.u64() != NDARRAY_MAGIC:
            raise ConversionError(f"invalid array magic for {name}")
        reader.u64()
        reader.u32()
        reader.u32()
        dimensions = reader.u32()
        dtype = (reader.u8(), reader.u8(), reader.u16())
        if dimensions > 16:
            raise ConversionError(f"{name}: unreasonable dimension count {dimensions}")
        if dtype != (2, 32, 1):
            raise ConversionError(f"{name}: expected scalar float32, got {dtype}")

        shape = tuple(reader.i64() for _ in range(dimensions))
        if any(dimension <= 0 for dimension in shape):
            raise ConversionError(f"{name}: invalid shape {shape}")
        elements = int(np.prod(shape, dtype=np.int64))
        byte_count = reader.u64()
        if byte_count != elements * 4:
            raise ConversionError(
                f"{name}: data size {byte_count} does not match shape {shape}"
            )
        tensors[name] = (
            np.frombuffer(reader.take(byte_count), dtype="<f4").reshape(shape).copy()
        )

    if reader.position != len(data):
        raise ConversionError(
            f"parameter file has {len(data) - reader.position} trailing bytes"
        )
    return tensors


def validate_weights(weights: Mapping[str, np.ndarray]) -> None:
    missing = sorted(set(EXPECTED_SHAPES) - set(weights))
    extra = sorted(set(weights) - set(EXPECTED_SHAPES))
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing {', '.join(missing)}")
        if extra:
            details.append(f"unexpected {', '.join(extra)}")
        raise ConversionError("incompatible tensor set: " + "; ".join(details))

    for name, expected in EXPECTED_SHAPES.items():
        value = weights[name]
        if tuple(value.shape) != expected:
            raise ConversionError(f"{name}: shape {value.shape}, expected {expected}")
        if not np.isfinite(value).all():
            raise ConversionError(f"{name}: contains NaN or infinity")


def validate_graph(data: bytes) -> None:
    try:
        graph = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConversionError(f"invalid graph JSON: {exc}") from exc
    if not isinstance(graph, dict) or not isinstance(graph.get("nodes"), list):
        raise ConversionError("graph JSON has no node list")


def read_files(first: Path, second: Path) -> ModelSource:
    paths = {path.suffix.lower(): path for path in (first, second)}
    if set(paths) != {".json", ".params"}:
        raise ConversionError(
            "direct input must contain one .json and one .params file"
        )
    if paths[".json"].stem != paths[".params"].stem:
        raise ConversionError("graph and parameter filenames must share the same stem")
    try:
        return ModelSource(
            paths[".json"].read_bytes(),
            paths[".params"].read_bytes(),
            f"{paths['.json']} and {paths['.params']}",
        )
    except OSError as exc:
        raise ConversionError(f"could not read model files: {exc}") from exc


def zip_pairs(archive: zipfile.ZipFile) -> list[tuple[str, str]]:
    graphs = {}
    params = {}
    for name in archive.namelist():
        path = PurePosixPath(name)
        key = str(path.with_suffix(""))
        if path.suffix.lower() == ".json":
            graphs[key] = name
        elif path.suffix.lower() == ".params":
            params[key] = name
    return [(graphs[key], params[key]) for key in sorted(set(graphs) & set(params))]


def read_zip(path: Path) -> ModelSource:
    try:
        with zipfile.ZipFile(path) as archive:
            matches = []
            for graph_name, params_name in zip_pairs(archive):
                graph = archive.read(graph_name)
                params = archive.read(params_name)
                try:
                    validate_graph(graph)
                    validate_weights(parse_params(params))
                except ConversionError:
                    continue
                matches.append((graph_name, params_name, graph, params))
    except (OSError, zipfile.BadZipFile, KeyError) as exc:
        raise ConversionError(f"could not read ZIP {path}: {exc}") from exc

    if len(matches) != 1:
        raise ConversionError(
            f"ZIP must contain exactly one compatible graph/parameter pair; found {len(matches)}"
        )
    graph_name, params_name, graph, params = matches[0]
    return ModelSource(graph, params, f"{path}:{graph_name} + {params_name}")


def read_source(inputs: list[Path]) -> ModelSource:
    if len(inputs) == 1 and inputs[0].suffix.lower() == ".zip":
        return read_zip(inputs[0])
    if len(inputs) == 2:
        return read_files(*inputs)
    raise ConversionError("provide one ZIP or one .json and one .params file")


def build_onnx(weights: Mapping[str, np.ndarray]) -> onnx.ModelProto:
    nodes = [
        helper.make_node(
            "Conv", ["input", "conv1_weight", "conv1_bias"], ["c1"], kernel_shape=[5, 5]
        ),
        helper.make_node("Relu", ["c1"], ["r1"]),
        helper.make_node(
            "MaxPool", ["r1"], ["p1"], kernel_shape=[2, 2], strides=[2, 2]
        ),
        helper.make_node(
            "Conv", ["p1", "conv2_weight", "conv2_bias"], ["c2"], kernel_shape=[5, 5]
        ),
        helper.make_node("Relu", ["c2"], ["r2"]),
        helper.make_node(
            "MaxPool", ["r2"], ["p2"], kernel_shape=[2, 2], strides=[2, 2]
        ),
        helper.make_node(
            "Conv", ["p2", "conv3_weight", "conv3_bias"], ["c3"], kernel_shape=[3, 3]
        ),
        helper.make_node("Relu", ["c3"], ["r3"]),
        helper.make_node("Flatten", ["r3"], ["f1"], axis=1),
        helper.make_node(
            "Gemm", ["f1", "fc5_ft_weight", "fc5_ft_bias"], ["d1"], transB=1
        ),
        helper.make_node(
            "Gemm", ["d1", "fc6_10_weight", "fc6_10_bias"], ["logits"], transB=1
        ),
        helper.make_node("Relu", ["logits"], ["output"]),
    ]
    graph = helper.make_graph(
        nodes,
        "lip_model",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2, 100, 100])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 30])],
        initializer=[
            numpy_helper.from_array(weights[name], name) for name in EXPECTED_SHAPES
        ],
    )
    model = helper.make_model(
        graph,
        producer_name="lip-tvm2onnx",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def conv(source: np.ndarray, kernel: np.ndarray, bias: np.ndarray) -> np.ndarray:
    channels, height, width = source.shape
    outputs, kernel_channels, kernel_height, kernel_width = kernel.shape
    if channels != kernel_channels:
        raise ConversionError("convolution channel mismatch")
    out_height = height - kernel_height + 1
    out_width = width - kernel_width + 1
    columns = np.empty(
        (channels * kernel_height * kernel_width, out_height * out_width),
        dtype=np.float32,
    )
    for y in range(kernel_height):
        for x in range(kernel_width):
            rows = (np.arange(channels) * kernel_height + y) * kernel_width + x
            columns[rows] = source[:, y : y + out_height, x : x + out_width].reshape(
                channels, -1
            )
    return (kernel.reshape(outputs, -1) @ columns + bias[:, None]).reshape(
        outputs, out_height, out_width
    )


def reference_forward(
    source: np.ndarray, weights: Mapping[str, np.ndarray]
) -> np.ndarray:
    value = np.maximum(conv(source, weights["conv1_weight"], weights["conv1_bias"]), 0)
    value = (
        value.reshape(value.shape[0], value.shape[1] // 2, 2, value.shape[2] // 2, 2)
        .max(axis=(2, 4))
        .copy()
    )
    value = np.maximum(conv(value, weights["conv2_weight"], weights["conv2_bias"]), 0)
    value = (
        value.reshape(value.shape[0], value.shape[1] // 2, 2, value.shape[2] // 2, 2)
        .max(axis=(2, 4))
        .copy()
    )
    value = np.maximum(conv(value, weights["conv3_weight"], weights["conv3_bias"]), 0)
    value = value.reshape(-1)
    value = weights["fc5_ft_weight"] @ value + weights["fc5_ft_bias"]
    return np.maximum(weights["fc6_10_weight"] @ value + weights["fc6_10_bias"], 0)


def validate_runtime(model_path: Path, weights: Mapping[str, np.ndarray]) -> float:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(model_path), options, providers=["CPUExecutionProvider"]
    )
    source = np.random.default_rng(0).standard_normal(
        (1, 2, 100, 100), dtype=np.float32
    )
    actual = session.run(None, {"input": source})[0][0]
    expected = reference_forward(source[0], weights)
    error = float(np.max(np.abs(actual - expected)))
    if not np.isfinite(error) or error >= 1e-3:
        raise ConversionError(f"numerical validation failed: maximum error {error:.3e}")
    return error


def convert(source: ModelSource, output: Path, force: bool = False) -> float:
    validate_graph(source.graph)
    weights = parse_params(source.params)
    validate_weights(weights)
    model = build_onnx(weights)

    output = output.expanduser().resolve()
    if output.exists() and not force:
        raise ConversionError(
            f"output already exists: {output} (use --force to replace it)"
        )
    output.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output.name}.", suffix=".tmp", dir=output.parent, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
        onnx.save(model, temporary_path)
        error = validate_runtime(temporary_path, weights)
        os.replace(temporary_path, output)
        output.chmod(0o644)
        temporary_path = None
        return error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a compatible TVM lip model to ONNX."
    )
    parser.add_argument("inputs", nargs="+", type=Path, metavar="INPUT")
    parser.add_argument("-o", "--output", type=Path, default=Path("lip_model.onnx"))
    parser.add_argument("--force", action="store_true")
    return parser


def run(arguments: list[str] | None = None) -> int:
    args = make_parser().parse_args(arguments)
    try:
        source = read_source(args.inputs)
        print(f"Input: {source.description}")
        error = convert(source, args.output, args.force)
        print(f"Wrote: {args.output.expanduser().resolve()}")
        print(f"Validation: max |ONNX - NumPy| = {error:.3e}")
        return 0
    except ConversionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
