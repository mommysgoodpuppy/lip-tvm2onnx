import struct
import zipfile
from pathlib import Path

import numpy as np
import pytest

from tvm2onnx import (
    NDARRAY_MAGIC,
    PARAMS_MAGIC,
    ConversionError,
    parse_params,
    read_files,
    validate_graph,
    validate_weights,
    zip_pairs,
)


def params_blob(name: str = "weight", values: tuple[float, ...] = (1.0, 2.0)) -> bytes:
    encoded = name.encode()
    payload = np.asarray(values, dtype="<f4").tobytes()
    return b"".join(
        (
            struct.pack("<QQQ", PARAMS_MAGIC, 0, 1),
            struct.pack("<Q", len(encoded)),
            encoded,
            struct.pack("<Q", 1),
            struct.pack("<QQIII", NDARRAY_MAGIC, 0, 1, 0, 1),
            struct.pack("<BBH", 2, 32, 1),
            struct.pack("<q", len(values)),
            struct.pack("<Q", len(payload)),
            payload,
        )
    )


def test_parse_params_reads_float32_tensor():
    tensors = parse_params(params_blob())
    assert tensors["weight"].dtype == np.float32
    assert tensors["weight"].tolist() == [1.0, 2.0]


def test_parse_params_rejects_trailing_bytes():
    with pytest.raises(ConversionError, match="trailing bytes"):
        parse_params(params_blob() + b"extra")


def test_graph_validation_is_structural():
    validate_graph(b'{"nodes": []}')
    with pytest.raises(ConversionError, match="node list"):
        validate_graph(b"{}")


def test_direct_files_can_have_any_shared_stem(tmp_path: Path):
    graph = tmp_path / "custom.json"
    params = tmp_path / "custom.params"
    graph.write_bytes(b'{"nodes": []}')
    params.write_bytes(b"params")
    source = read_files(params, graph)
    assert source.graph == b'{"nodes": []}'
    assert source.params == b"params"


def test_zip_pair_discovery_uses_matching_stems(tmp_path: Path):
    archive_path = tmp_path / "models.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("a/model.json", b"{}")
        archive.writestr("a/model.params", b"")
        archive.writestr("b/unpaired.json", b"{}")
    with zipfile.ZipFile(archive_path) as archive:
        assert zip_pairs(archive) == [("a/model.json", "a/model.params")]


def test_weight_validation_rejects_wrong_layout():
    with pytest.raises(ConversionError, match="incompatible tensor set"):
        validate_weights({"weight": np.zeros(2, dtype=np.float32)})
