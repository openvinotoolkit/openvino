# CLI: Run

Use this command to run a policy runtime from a YAML file.

```bash
physicalai run --config runtime.yaml
```

You can limit the run duration when needed.

```bash
physicalai run --config runtime.yaml --run.duration_s=60
```

The same duration limit is available from the current Python API after you
construct a runtime directly.

```python
from physicalai.runtime import PolicyRuntime

runtime = PolicyRuntime(...)

with runtime:
    runtime.run(duration_s=60)
```

Runtime commands live in the `physicalai` package. Training commands should be provided by training packages or plugin entry points so the runtime package stays lightweight.
