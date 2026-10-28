# Architecture

The Physical AI runtime has a small set of layers with clear boundaries between exported packages, inference, runtime orchestration, and hardware IO.

```text
exported package
    -> InferenceModel
    -> RobotRuntime
    -> Robot and cameras
```

## Components

| Component        | Responsibility                                                   |
| ---------------- | ---------------------------------------------------------------- |
| `Manifest`       | describes exported artifacts and the inference pipeline          |
| `InferenceModel` | loads artifacts and computes actions                             |
| `ActionSource`   | decides the action to send each tick (policy, teleop, or custom) |
| `RobotRuntime`   | runs the robot control loop                                      |
| `Execution`      | decides where inference runs                                     |
| `ActionQueue`    | buffers and merges action chunks                                 |
| `Robot`          | reads state and sends commands                                   |
| `Camera`         | reads image frames                                               |

## Package Boundary

`physicalai` is the runtime package. It should not require training dependencies in order to import or run deployment workflows.

Training packages can add commands through CLI entry points when they need to extend the runtime CLI.

```text
physicalai
  run

training package
  fit
  validate
  test
  predict
  benchmark
  export
```

## Design Rules

- Config objects remain passive data structures.
- Orchestrators are responsible for executing workflows.
- `InferenceModel` does not own robot timing.
- `RobotRuntime` does not own policy math or the decision of what action to send — that is the action source's job.
- Manifests describe exported packages.
- Workflow configs describe the desired execution before it starts.
