# Inference API Reference

## `InferenceModel`

```python
InferenceModel(
    export_dir: str | Path,
    policy_name: str | None = None,
    backend: str = "auto",
    device: str = "auto",
    runner: InferenceRunner | None = None,
    preprocessors: list[Preprocessor] | None = None,
    postprocessors: list[Postprocessor] | None = None,
    callbacks: list[Callback] | None = None,
    **adapter_kwargs,
)
```

The model can be constructed directly from an export directory or loaded from config.

## Constructors

```python
model = InferenceModel.load("./exports/act_policy")
```

> **Note:** `InferenceModel.from_config()` is a planned API.

## Methods

### `select_action`

```python
action = model.select_action(observation)
```

This method returns one action.

### `predict_action_chunk`

```python
chunk = model.predict_action_chunk(observation)
```

This method returns a chunk of actions for runtime queueing.

### `reset`

```python
model.reset()
```

This method clears the runner state and the action cursor.

### `close`

```python
model.close()
```

This method releases backend resources.

## Observation

Observations are dictionaries of NumPy arrays.

```python
observation = {
    "state": joint_positions,
    "image.wrist": wrist_image,
}
```

The expected keys and shapes come from the exported package and its preprocessing pipeline.
