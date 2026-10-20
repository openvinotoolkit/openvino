# CLI Reference

The runtime CLI commands use the same schemas as the Python APIs.

The canonical command is `physicalai`. A shorthand alias `pai` is also
installed and behaves identically.

## `physicalai run`

```bash
physicalai run --config runtime.yaml [--run.duration_s=60]
```

Arguments:

| Argument           | Required | Description                              |
| ------------------ | -------- | ---------------------------------------- |
| `--config`         | yes      | Runtime config YAML                      |
| `--run.duration_s` | no       | Stop after the given duration in seconds |

The same duration limit is available from the current Python API after you
construct a runtime directly.

```python
runtime = PolicyRuntime(...)

with runtime:
    runtime.run(duration_s=60)
```

## Shell Completion

Shell completion scripts can be printed directly from the CLI and sourced in
your shell. Completion includes any installed plugin subcommands, such as the
studio training commands.

```bash
source <(physicalai completion zsh)
source <(pai completion bash)
```

## Plugin Commands

Training packages can add commands through entry points.

```toml
[project.entry-points."physicalai.cli.subcommands"]
fit = "physicalai.cli.fit:register"
validate = "physicalai.cli.validate:register"
test = "physicalai.cli.test:register"
predict = "physicalai.cli.predict:register"
benchmark = "physicalai.cli.benchmark:register"
export = "physicalai.cli.export:register"
```
