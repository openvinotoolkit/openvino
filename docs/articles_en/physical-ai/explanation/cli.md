# CLI

The CLI is a thin wrapper over the same config APIs used by Python.

The canonical command is `physicalai`. A shorthand alias `pai` is also
installed and behaves identically.

```bash
physicalai run --config runtime.yaml --run.duration_s=60
```

Equivalent Python control flow:

```python
runtime = PolicyRuntime(...)

with runtime:
    runtime.run(duration_s=60)
```

Shell completion is available through `jsonargparse` + `shtab`. Print a shell
completion script with the `completion` command and source or install it in
your shell. When training packages are installed, their plugin subcommands are
included in completion automatically.

```bash
source <(physicalai completion zsh)
source <(pai completion zsh)
```

## Runtime Commands

| Command          | Purpose                          |
| ---------------- | -------------------------------- |
| `physicalai run` | Runs a policy on robot hardware. |

## Training Commands

Training commands should come from training packages or entry-point plugins.

```toml
[project.entry-points."physicalai.cli.subcommands"]
fit = "physicalai.cli.fit:register"
benchmark = "physicalai.cli.benchmark:register"
export = "physicalai.cli.export:register"
```

Importing `physicalai` should not pull in training dependencies.
