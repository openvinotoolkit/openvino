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
runtime = RobotRuntime(...)

with runtime:
    runtime.run(duration_s=60)
```

## `physicalai robot serve`

```bash
physicalai robot serve --config examples/so101/serve.yaml
```

The YAML fields are `name`, required `robot` (a `Config` with
`class_path` + `init_args`), optional `allow_remote` (default `false`), and
optional positive `rate_hz` (default 100 Hz). Direct CLI arguments use the same
names, including nested robot constructor values such as
`--robot.init_args.port /dev/ttyACM0`. Flat `robot_class` / `robot_kwargs` are
unsupported. Serving stays in the foreground until SIGINT or SIGTERM and returns
nonzero for startup, loop, repeated-read, or disconnect failures.

Use `--verbose` to include driver construction, lock acquisition, initial observation,
endpoint declaration, and cleanup details.

`--allow_remote` exposes an unauthenticated physical action endpoint. Use it only on
an isolated robot-cell VLAN/firewall or with Zenoh ACL/TLS.

## `physicalai robot discover`

```bash
physicalai robot discover [--allow_remote] [--timeout 2] [--json]
```

Results are sorted by robot name and host. Human output is an ASCII table containing
name, class, host, and joint count, followed by the result count and elapsed discovery
time. JSON mode writes exactly one array to stdout; empty discovery is successful and
writes `[]`.

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
