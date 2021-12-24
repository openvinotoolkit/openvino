# Graph serialization

The functionality allows to serialize execution graph using environment variable:
```sh
    OV_CPU_EXEC_GRAPH_PATH=<path> binary ...
```

Possible serialization options:
* cout

    Serialize to console output
* \<path\>.xml

    Serialize graph into .xml and .bin files. Can be opened using, for example, *netron* app
* \<path\>.dot

    TBD. Serialize graph into .dot file. Can be inspected using, for example, *graphviz* tools.
