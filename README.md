# lip-tvm2onnx

Convert a compatible TVM lip-tracking model into a portable ONNX model. 
Currently supports the model included in `SRanipal 1.3.6.5`.

The converter accepts either a ZIP containing one compatible `.json` and
`.params` pair, or those two files directly.
No model weights or third-party assets are included.

## Usage

Install [uv](https://docs.astral.sh/uv/), clone this repository, then run one
of the following commands from the repository directory:

```console
uv run src/tvm2onnx.py /path/to/model.zip
```

```console
uv run src/tvm2onnx.py \
  /path/to/model.json \
  /path/to/model.params \
  --output lip_model.onnx
```

Existing output files are preserved unless `--force` is supplied.

## Model layout

The generated model has this interface:

```text
input   float32 [1, 2, 100, 100]

Conv 5x5, 2  -> 20  + ReLU + MaxPool 2x2
Conv 5x5, 20 -> 48  + ReLU + MaxPool 2x2
Conv 3x3, 48 -> 64  + ReLU
Flatten
Dense 25600  -> 500
Dense 500    -> 30  + ReLU

output  float32 [1, 30]
```

The expected tensor names and shapes are defined in `src/tvm2onnx.py`.

## Development

```console
uv sync --dev
uv run pytest
```
