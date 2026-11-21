# Configuration

Physical AI describes setups in a simple, repeatable way: **pick a Python class**
and **pass the arguments its constructor needs**. The same description can live in
Python, YAML, the CLI, or Studio.

## Config recipes

A config recipe names the class and its constructor arguments:

```yaml
class_path: physicalai.capture.UVCCamera
init_args:
  device: /dev/v4l/by-id/usb-Example_Camera-video-index0
  width: 640
  height: 480
```

- `class_path` — import path to the class (for example a camera driver).
- `init_args` — keyword arguments passed to `__init__`.

Classes that should round-trip through YAML can opt in with `@export_config`.
Then `Config.from_instance(live_object)` records how the object was created:
which arguments you passed, without open devices, sockets, or other runtime state.

```text
live object -> Config.from_instance() -> YAML/JSON -> config.instantiate() -> new object
```

`Config.instantiate()` only runs the constructor. Your app still calls methods like
`connect()` or `run()` when needed.

## Workflow files (`physicalai run`)

Run configs add structure around the same recipe. For example, `physicalai run`
expects runtime settings under `runtime:` and optional run options under `run:`.
You can also pass a bare exported `RobotRuntime` recipe; the loader reshapes it
into that form.

Nested parts (robot, policy, cameras, callbacks) use the same
`class_path` + `init_args` pattern, nested inside `init_args` where needed.

## Inference manifests

An exported policy folder includes a **manifest**: file paths, preprocessors,
runners, and compatibility metadata. That format is aimed at loading a trained
package. Robot and runtime YAML typically use `Config` recipes instead.

Use workflow YAML when you describe **how to run** something. Use the manifest
when you describe **what is inside an export**.

## Training and typed settings

When you already know the Python type (a trainer, dataclass, CLI model), build
it with **jsonargparse**: define a parser for that type, load YAML or CLI values,
then call `instantiate()`.

```python
parser = ArgumentParser(exit_on_error=False)
parser.add_class_arguments(Trainer, "trainer")
parsed = parser.parse_object(document, defaults=False)
trainer = parser.instantiate(parsed).trainer
```

`physicalai.config.Config` is the shared recipe format for portable YAML:
`class_path` plus `init_args`. Pair it with `@export_config` when a live
component should save to disk and be recreated later.

## Safety

Loading a config imports Python modules from `class_path`. Only use files you
wrote or otherwise trust on that machine. Do not build objects from robot or
camera metadata received over the network.
