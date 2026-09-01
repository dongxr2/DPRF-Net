# Third-party baselines

The deep-baseline script supports Trusted Multi-View Classification (TMC) using the authors' official implementation:

https://github.com/hanmenghan/TMC

The source snapshot used in the experiment corresponded to commit `a3272b8746861c76a3461943b5eee51df5b5a8fe`. The upstream repository did not contain a clear license file when this release package was prepared, so its source is not redistributed here.

To run TMC, place the official `model.py` at:

```text
third_party/TMC/TMC ICLR/model.py
```

Review the upstream terms before using or redistributing that code. The paper's saved TMC metrics are available under `results/deep_baselines/`.
