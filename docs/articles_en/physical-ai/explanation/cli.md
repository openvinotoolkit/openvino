# CLI

> **Preview:** The CLI is a planned API. The content below documents the target design.

The CLI is a thin wrapper over the same config APIs used by Python.

```bash
physicalai run --config runtime.yaml --duration-s 60
```

Equivalent Python:

```python
PolicyRuntime.from_config("runtime.yaml").run(duration_s=60)
```

## Runtime Commands

| Command            | Purpose                           |
| ------------------ | --------------------------------- |
| `physicalai infer` | Runs offline inference.           |
| `physicalai run`   | Runs a policy on robot hardware.  |
| `physicalai serve` | Serves policy inference remotely. |

## Training Commands

Training commands should come from training packages or entry-point plugins.

```toml
[project.entry-points."physicalai.cli.subcommands"]
fit = "physicalai.train.cli:register_fit"
export = "physicalai.train.cli:register_export"
```

Importing `physicalai` should not pull in training dependencies.
