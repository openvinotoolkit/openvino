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

## Deprecation Notice

<table>
  <tr>
    <td><strong>Deprecation Begins</strong></td>
    <td>June 1, 2020</td>
  </tr>
  <tr>
    <td><strong>Removal Date</strong></td>
    <td>December 1, 2020</td>
  </tr>
</table> 

*Starting with the OpenVINO™ toolkit 2020.2 release, all of the features previously available through nGraph have been merged into the OpenVINO™ toolkit. As a result, all the features previously available through ONNX RT Execution Provider for nGraph have been merged with ONNX RT Execution Provider for OpenVINO™ toolkit.*

*Therefore, ONNX RT Execution Provider for nGraph will be deprecated starting June 1, 2020 and will be completely removed on December 1, 2020. Users are recommended to migrate to the ONNX RT Execution Provider for OpenVINO™ toolkit as the unified solution for all AI inferencing on Intel® hardware.*
