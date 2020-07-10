# Build a Model with nGraph Library {#openvino_docs_IE_DG_nGraphTutorial}

This section illustrates how to construct an nGraph function 
composed of operations from the `opset3` namespace. Once created, 
it can wrap into a `CNNNetwork`, creating utility for data scientists 
or app developers to define a deep-learning model in a neutral way
that does not depend on existing Deep Learning (DL) frameworks.

Operation Set `opsetX` integrates a list of nGraph pre-compiled operations that work
for this purpose. In other words, `opsetX` defines a set of operations for building a graph.

For a complete list of operation sets supported by Inference Engine, see [Available Operations Sets](../ops/opset.md).

To add custom nGraph operations to an existing `CNNNetwork`, see 
the [Add Custom nGraph Operations](Extensibility_DG/Intro.md) document.

Now that you can build graphs with anything from the `opset3` definition, some 
parameters for shape-relevant (or shape-specific) inputs can be added. The 
following code prepares a graph for shape-relevant parameters. 

> **NOTE**: `validate_nodes_and_infer_types(ops)` must be included for partial shape inference. 

```cpp
#include "ngraph/opsets/opset.hpp"
#include "ngraph/opsets/opset3.hpp"

using namespace std;
using namespace ngraph;

auto arg0 = make_shared<opset3::Parameter>(element::f32, Shape{7});
auto arg1 = make_shared<opset3::Parameter>(element::f32, Shape{7});
// Create an 'Add' operation with two inputs 'arg0' and 'arg1'
auto add0 = make_shared<opset3::Add>(arg0, arg1);
auto abs0 = make_shared<opset3::Abs>(add0);
// Create a node whose inputs/attributes will be specified later
auto acos0 = make_shared<opset3::Acos>();
// Create a node using opset factories
auto add1 = shared_ptr<Node>(get_opset3().create("Add"));
// Set inputs to nodes explicitly
acos0->set_argument(0, add0);
add1->set_argument(0, acos0);
add1->set_argument(1, abs0);

// Run shape inference on the nodes
NodeVector ops{arg0, arg1, add0, abs0, acos0, add1};
validate_nodes_and_infer_types(ops);

// Create a graph with one output (add1) and four inputs (arg0, arg1)
auto ng_function = make_shared<Function>(OutputVector{add1}, ParameterVector{arg0, arg1});

```

To wrap it into a CNNNetwork, use: 
```cpp
CNNNetwork net (ng_function);
```

## See Also

* [Available Operation Sets](../ops/opset.md)
* [Operation Set `opset1` Specification](../ops/opset1.md)
* [Operation Set `opset2` Specification](../ops/opset2.md)
* [Operation Set `opset3` Specification](../ops/opset3.md)
* [Inference Engine Extensibility Developer Guide](Extensibility_DG/Intro.md)
