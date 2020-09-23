# Graph Debug Capabilities {#openvino_docs_IE_DG_Graph_debug_capabilities}

Inference Engine supports two different objects for a graph representation: the nGraph function and 
CNNNetwork. Both representations provide an API to get detailed information about the graph structure.

## nGraph Function

To receive additional messages about applied graph modifications, rebuild the nGraph library with 
the `-DNGRAPH_DEBUG_ENABLE=ON` option.

To enable serialization and deserialization of the nGraph function to a JSON file, rebuild the 
nGraph library with the `-DNGRAPH_JSON_ENABLE=ON` option. To serialize or deserialize the nGraph
function, call the nGraph function as follows:

```cpp
#include <ngraph/serializer.hpp>

std::shared_ptr<ngraph::Function> nGraph;
...
ngraph::serialize("test_json.json", nGraph);        // For graph serialization
std::ifstream file("test_json.json");               // Open a JSON file
nGraph = ngraph::deserialize(file);                 // For graph deserialization
```

To visualize the nGraph function to the xDot format or to an image file, use the 
`ngraph::pass::VisualizeTree` graph transformation pass:
```cpp
#include <ngraph/pass/visualize_tree.hpp>

std::shared_ptr<ngraph::Function> nGraph;
...
std::vector<std::shared_ptr<ngraph::Function>> g2{nGraph};
ngraph::pass::VisualizeTree("after.png").run_on_module(g2);     // Visualize the nGraph function to an image
```

## CNNNetwork

To serialize the CNNNetwork to the Inference Engine Intermediate Representation (IR) format, use the 
`CNNNetwork::serialize(...)` method:
```cpp
std::shared_ptr<ngraph::Function> nGraph;
...
CNNNetwork network(nGraph);
network.serialize("test_ir.xml", "test_ir.bin");
```
> **NOTE**: CNNNetwork created from the nGraph function might differ from the original nGraph 
> function because the Inference Engine applies some graph transformation.
