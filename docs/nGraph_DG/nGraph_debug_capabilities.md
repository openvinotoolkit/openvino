# nGraph Debug Capabilities {#openvino_docs_nGraph_DG_Debug_capabilities}

nGraph representation provides an API to get detailed information about the graph structure.

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
