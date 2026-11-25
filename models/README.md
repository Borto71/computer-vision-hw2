# models/

- `mobilenet.py`: builds MobileNetV3-Small (`MobileNet_V3_Small_Weights.DEFAULT`, ImageNet-1K) with the last classifier layer replaced by `Linear(1024, 37)`, for the three approaches:
  - `full`: all parameters trainable;
  - `head`: `features` frozen (BatchNorm kept in eval mode), only `classifier[0]` and `classifier[3]` trainable;
  - `features`: everything frozen; `extract_features` returns the 576-d vectors after global average pooling.

  `python models/mobilenet.py` prints the output shape and the trainable parameters of each mode.

The trained checkpoints are in `../checkpoints/`.
