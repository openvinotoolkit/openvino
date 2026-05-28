# Getting Started

This section contains short tutorials for the first successful PhysicalAI workflows.

## Recommended Order

1. [Installation](installation.md)
2. [Quickstart](quickstart.md)
3. [Run a Policy](run-a-policy.md)

## Minimal Path

```bash
pip install physicalai
```

```python
from physicalai.inference import InferenceModel

model = InferenceModel.load("./exports/act_policy")
model.reset()
action = model.select_action(observation)
```

Use Python when you need direct control over objects. Use YAML and the CLI when you need a reproducible run that can be shared or repeated.
