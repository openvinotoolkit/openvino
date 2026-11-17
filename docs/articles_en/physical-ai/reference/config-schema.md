# Config Schema Reference

## Config

`physicalai.config.Config` uses the jsonargparse-compatible
`class_path` + `init_args` shape.

```yaml
class_path: package.module.ClassName
init_args:
  key: value
```

| Field        | Required | Type   | Description                                |
| ------------ | -------- | ------ | ------------------------------------------ |
| `class_path` | yes      | string | Non-empty import path resolving to a class |
| `init_args`  | no       | object | String-keyed constructor arguments         |

Omitted `init_args` means an empty mapping. No other top-level keys are
accepted.

Values may be `null`, booleans, integers, finite floats, strings, lists, or
string-keyed mappings. Nested components use another `Config`.
Every mapping containing `class_path` is reserved as a nested component and
must contain only `class_path` and optional `init_args`. Nesting through
components, lists, and mappings is limited to 10 levels.

Export converts `Path` to its unchanged string form, `Enum` to its JSON-safe
value, and tuples to lists. `instantiate()` accepts the JSON representation,
not those original Python objects.

## Runtime Documents

A CLI runtime document places constructor arguments under `runtime:` and may
include method arguments under `run:`.

```yaml
runtime:
  robot:
    class_path: physicalai.robot.SO101
    init_args:
      port: /dev/ttyACM0
      calibration: ./calibration.json
  action_source:
    class_path: physicalai.runtime.TeleopSource
    init_args:
      leader:
        class_path: physicalai.robot.SharedRobot
        init_args:
          name: leader-arm
  fps: 30
run:
  duration_s: 60
```

`physicalai run --config` and `RobotRuntime.from_config()` also accept a bare
exported `RobotRuntime` Config whose top-level `class_path` is
`physicalai.runtime.RobotRuntime`.

## Manifest ComponentSpec

Inference manifests retain their separate `ComponentSpec` compatibility
model. It supports registry `type` aliases, artifact resolution, and existing
extra-field behavior. It is not the strict captured `Config` schema;
manifest unification is a separate design decision.

## Security

`class_path` is executable local configuration. `instantiate()` is for
trusted application and user-authored configs only, never metadata or control
messages received from peers. See [Security Model](../getting-started/security.md)
for the full set of deployment-time trust assumptions.
