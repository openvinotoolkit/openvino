# Security Model

The runtime treats several categories of input as trusted by design and does not sandbox them further. If you
load a config, manifest, or exported policy you did not author or fully review, you are running that content
with the same privileges as the `physicalai` process itself. The sections below describe each trust boundary.

## Configs and manifests can execute arbitrary code

YAML/JSON configs passed via `--config` flag and an exported policy's `manifest.json` both use `class_path` values to
dynamically import and construct Python objects - robots, cameras, preprocessors, postprocessors, action
sources, callbacks. Currently, nothing restricts which class a `class_path` may name.

Only load configs, exported policies, and manifests from sources you trust. See also the `class_path` note in
[Config Schema Reference](../reference/config-schema.md#security).

## Use only trusted, reviewed policies

An exported policy package (`manifest.json` plus artifacts) runs with the same privileges as the
`physicalai` process.

**Loading via `InferenceModel.from_pretrained()`:** pin `revision` to the commit SHA of a version you have
reviewed and trust, rather than a mutable branch or tag, so the content you reviewed is exactly what gets
loaded on every run.

**Loading via `export_dir`:** `physicalai run` and direct `InferenceModel(export_dir=...)` construction both
load whatever package is already in that local directory. Only place a reviewed, trusted export there.
Treat populating that directory (downloading, copying, extracting) as the point where you decide to trust
its contents.

## Remote robot sharing has no built-in security controls

The `SharedRobot` network transport (used to share one robot connection across processes) has no
authentication, access control or encryption of its own.

If you enable `allow_remote=True` (`--allow_remote` on `physicalai robot serve`/`discover`), use it only on
an isolated, firewalled robot-cell network (VLAN/firewall) or with Zenoh ACL/TLS configured yourself — the
same requirement documented in [CLI Reference](../reference/cli.md#physicalai-robot-serve). Without one of
those, anyone who can reach that network can observe robot state and, if nothing else restricts it, send
actions to the robot.

## Runtime callbacks run with full trust

Callbacks registered with `RobotRuntime` (see
[Add Runtime Callbacks](../how-to/runtime/add-runtime-callbacks.md)) can inspect and modify the
action sent to the robot on every control tick. Currently, the runtime does not validate a callback's output before
sending it to hardware.

Only register callbacks you wrote or have reviewed, especially any callback that can transform the outgoing
action.

## CLI subcommands load from the active Python environment

`physicalai <subcommand>` discovers third-party subcommands via Python entry points registered by packages
installed in the current environment. There is no allow-list of which packages may register a subcommand.

Treat installing a package into the same environment as granting it the ability to run as a `physicalai`
subcommand and apply the same security review you would to any other dependency.
