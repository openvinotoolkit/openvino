# Load an Exported Policy

Load the exported package with auto-detection first.

```python
from physicalai.inference import InferenceModel

model = InferenceModel.load("./exports/act_policy")
```

Then compute one action.

```python
model.reset()
action = model.select_action(observation)
```

If necessary, select the backend explicitly.

```python
model = InferenceModel.load(
    "./exports/act_policy",
    backend="openvino",
    device="CPU",
)
```

Use chunk prediction when a runtime owns the queueing and timing.

```python
chunk = model.predict_action_chunk(observation)
```

Do not build robot-loop timing around `select_action()`. Use `PolicyRuntime` when the policy is driving hardware.
