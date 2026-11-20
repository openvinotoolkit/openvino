# Upgrade from 0.1.x

`PolicyRuntime` and `InferenceModel.load()` have been removed in Runtime
v0.2.0. Update code that uses them and adjust YAML configurations using the
replacements below.

| 0.1.x | 0.2.0 |
| --- | --- |
| `PolicyRuntime` | `RobotRuntime` with an explicit `PolicySource` |
| Policy model and execution settings directly on the runtime | Settings on `PolicySource` (under `action_source.init_args` in YAML) |
| `InferenceModel.load(...)` | `InferenceModel(export_dir=...)` for local exports, or `InferenceModel.from_pretrained(...)` for the Hub |
| Older OpenVINO releases | OpenVINO >= 2026.4.0 and OpenVINO Tokenizers >= 2026.4.0.0 |

The older configuration helpers `FromConfig` and `instantiate_obj` still work
but emit deprecation warnings and will be removed in a future release. Use
`Config` or jsonargparse in new code.

The [runtime configuration example](https://github.com/openvinotoolkit/physicalai/blob/v0.2.0/examples/runtime/runtime.yaml)
is a good starting point for migrating a 0.1.x configuration.
