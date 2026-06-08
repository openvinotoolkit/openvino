# Execution graph serialization

Execution graph serialization is disabled by default and controlled by environment variable **OV_CPU_EXEC_GRAPH_PATH**:
```sh
OV_CPU_EXEC_GRAPH_PATH=<option> binary ...
```

Possible serialization options:
* cout\
Serialize to console output.
* \<path\>_index.xml\
Serialize graph into .xml and .bin files. Can be opened using, for example, *netron* app.
* \<path\>_index.dot\
Serialize graph into .dot file. Can be inspected using, for example, *graphviz* tools.

## Output filename

Output files are named `<option>_<index>_<model_name>.<ext>`, where `index` reflects compilation order. Example with `OV_CPU_EXEC_GRAPH_PATH=exec.xml`:
```
exec_0_ModelA.xml
exec_1_ModelB.xml
```

## Two-phase serialization

Each graph is serialized twice:
1. **At compile time** — written immediately after graph construction with `not_executed` perf counters. Useful for diagnosing crashes during the first inference.
2. **After inference** — the same file is overwritten with real per-layer execution times when the compiled model is destroyed.

## Multi-stream

All streams of the same model share one output file. The last stream to be destroyed overwrites it, which is fine since all streams run the same graph.
