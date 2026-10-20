# Load an Exported Policy

Load the exported package with auto-detection first.

```python
from physicalai.inference import InferenceModel

model = InferenceModel("./exports/act_policy")
```

Then compute one action.

```python
model.reset()
action = model.select_action(observation)
```

Load directly from the Hugging Face Hub with a repo id.

```python
model = InferenceModel.from_pretrained("OpenVINO/act-fp16-ov")
```

Pin a revision (branch, tag, or commit SHA) for reproducible loads.

```python
model = InferenceModel.from_pretrained(
    "OpenVINO/act-fp16-ov",
    revision="main",
)
```

If necessary, select the backend explicitly.

```python
model = InferenceModel(
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
