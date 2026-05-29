# Quickstart

This tutorial shows the smallest useful inference flow: load an exported policy package and compute one action.

```python
from physicalai.inference import InferenceModel

model = InferenceModel.load("./exports/act_policy")
model.reset()

action = model.select_action(observation)
```

The `observation` input is a dictionary of NumPy arrays. The keys must match the names expected by the exported policy.

For example, an observation may look like this.

```python
observation = {
    "state": joint_positions,
    "image.wrist": wrist_image,
    "image.front": front_image,
}
```

## Chunk Policies

Some policies produce action chunks internally. Even in that case, `select_action()` still returns a single action on each call.

```python
for _ in range(100):
    action = model.select_action(observation)
    observation = env.step(action)
```

If you are building a robot control loop, use chunk prediction through `PolicyRuntime` instead of managing timing and buffering yourself.

```python
chunk = model.predict_action_chunk(observation)
```

## Manifest Path

Exported packages include a manifest.

```text
exports/act_policy/
├── manifest.json
├── model.xml
└── stats.safetensors
```

The manifest records the exported artifacts, the runner configuration, the preprocessing and postprocessing pipeline, and the hardware metadata.
