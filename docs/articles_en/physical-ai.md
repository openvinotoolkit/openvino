---
explore_github_url: https://github.com/openvinotoolkit/physicalai
---

# OpenVINO™ Physical AI

```{toctree}
:hidden:
:maxdepth: 2

physical-ai/getting-started/README
physical-ai/how-to/README
physical-ai/explanation/README
physical-ai/reference/README
```

OpenVINO™ Physical AI provides runtime components for working with exported robot policies. The documentation is organized around the main deployment tasks and the runtime concepts behind them.

## Quick Links

| I want to               | Go to                                                   |
| ----------------------- | ------------------------------------------------------- |
| Install the package     | [Installation](physical-ai/getting-started/installation.md)         |
| Run first inference     | [Quickstart](physical-ai/getting-started/quickstart.md)             |
| Run a policy on a robot | [Run a Policy](physical-ai/getting-started/run-a-policy.md)         |
| Write runtime YAML      | [Runtime Config](physical-ai/how-to/config/write-runtime-config.md) |
| Use the runtime CLI     | [CLI Run](physical-ai/how-to/cli/run.md)                            |
| Understand architecture | [Architecture](physical-ai/explanation/architecture.md)             |
| Look up schemas         | [Config Schema](physical-ai/reference/config-schema.md)             |

## Documentation Structure

```text
docs/
├── getting-started/  # tutorials
├── how-to/           # task guides
├── explanation/      # concepts and boundaries
└── reference/        # exact commands, schemas, APIs
```

## Workflow

Most deployment workflows follow the same path from an exported package to a running robot loop.

```text
exported policy package
    -> InferenceModel
    -> PolicyRuntime
    -> Robot
```

Python example:

```python
from physicalai.inference import InferenceModel
from physicalai.runtime import PolicyRuntime, SyncExecution
from physicalai.robot import SO101
from physicalai.capture import UVCCamera

model = InferenceModel.load("./exports/act_policy")
robot = SO101(port="/dev/ttyACM0")
cameras = {"wrist": UVCCamera(device="/dev/video0", width=640, height=480)}

runtime = PolicyRuntime(
    fps=30,
    robot=robot,
    model=model,
    cameras=cameras,
    execution=SyncExecution(mode="chunk"),
)

runtime.run(duration_s=60)
```

CLI example:

```bash
physicalai run --config runtime.yaml --duration-s 60
```

> **Note:** `PolicyRuntime` and the CLI are planned APIs. See [#121](https://github.com/openvinotoolkit/physicalai/issues/121) for status.
