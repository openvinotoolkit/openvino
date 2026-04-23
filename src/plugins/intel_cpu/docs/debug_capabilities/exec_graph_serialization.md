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
