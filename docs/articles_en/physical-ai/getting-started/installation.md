# Installation

Install the runtime package first.

```bash
pip install physicalai
```

Install hardware extras only when you need a specific backend or robot integration.

```bash
pip install "physicalai[realsense]"
pip install "physicalai[so101]"
pip install "physicalai[robots]"
pip install "physicalai[capture]"  # all camera backends + IPC transport
```

For local development, install the repository environment and run the test suite.

```bash
uv sync
uv run pytest
```

## Package Boundary

`physicalai` is the runtime package. It should remain usable on deployment hosts that do not have training dependencies such as Torch or Lightning installed.

Training commands may be provided by a separate training distribution or by plugin entry points.

## Check the Install

```python
import physicalai
from physicalai.inference import InferenceModel
```

If hardware extras are not installed, importing the base runtime should still work.
