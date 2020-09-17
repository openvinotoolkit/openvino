# Graph Debug Capabilities {#openvino_docs_IE_DG_Graph_debug_capabilities}

Inference Engine supports two different objects for a graph representation: the nGraph function and 
CNNNetwork. Both representations provide an API to get detailed information about the graph structure.

## nGraph Function

To receive additional messages about applied graph modifications, rebuild the nGraph library with 
the `-DNGRAPH_DEBUG_ENABLE=ON` option.

To visualize the nGraph function to the xDot format or to an image file, use the 
`ngraph::pass::VisualizeTree` graph transformation pass:
```cpp
#include <ngraph/pass/visualize_tree.hpp>

std::shared_ptr<ngraph::Function> nGraph;
...
ngraph::pass::VisualizeTree("after.png").run_on_function(nGraph);     // Visualize the nGraph function to an image
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
