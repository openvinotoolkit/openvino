# PhysicalAI Documentation

PhysicalAI provides runtime components for working with exported robot policies. The documentation is organized around the main deployment tasks and the runtime concepts behind them.

## Quick Links

| I want to               | Go to                                                   |
| ----------------------- | ------------------------------------------------------- |
| Install the package     | [Installation](getting-started/installation.md)         |
| Run first inference     | [Quickstart](getting-started/quickstart.md)             |
| Run a policy on a robot | [Run a Policy](getting-started/run-a-policy.md)         |
| Write runtime YAML      | [Runtime Config](how-to/config/write-runtime-config.md) |
| Use the runtime CLI     | [CLI Run](how-to/cli/run.md)                            |
| Understand architecture | [Architecture](explanation/architecture.md)             |
| Look up schemas         | [Config Schema](reference/config-schema.md)             |

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


::: {toctree}
:hidden:
:maxdepth: 2
:caption: Getting Started

getting-started/README
getting-started/installation
getting-started/quickstart
getting-started/run-a-policy
:::

::: {toctree}
:hidden:
:maxdepth: 2
:caption: How-To

how-to/README
how-to/runtime/run-policy-on-robot
how-to/runtime/use-execution-modes
how-to/runtime/add-runtime-callbacks
how-to/inference/load-exported-policy
how-to/inference/use-manifest
how-to/inference/configure-pre-post-processing
how-to/config/write-runtime-config
how-to/config/write-inference-config
how-to/config/instantiate-components
how-to/cli/run
how-to/cli/infer
:::

::: {toctree}
:hidden:
:maxdepth: 2
:caption: Explanation

explanation/README
explanation/architecture
explanation/inference
explanation/runtime
explanation/configuration
explanation/manifests
explanation/cli
explanation/robots
explanation/cameras
:::

::: {toctree}
:hidden:
:maxdepth: 2
:caption: Reference

reference/README
reference/cli
reference/config-schema
reference/manifest-schema
reference/inference-api
reference/runtime-api
reference/robot-api
reference/camera-api
:::