# Using nGraph's Python API {#openvino_docs_nGraph_DG_PythonAPI}

nGraph is the OpenVINO&trade; graph manipulation library, used to represent neural network models in the form of a computational graph. With nGraph Python APIs, you can create, inspect, and modify computational graphs.

## nGraph namespace module

nGraph is able to represent a large set of mathematical operations, chosen to cover most operations used in popular neural network models. The `ngraph` module is the basic namespace used to expose commonly used API methods. You can inspect the `ngraph` module or call `help` on any method for usage information.

```python
import ngraph as ng
dir(ng)
help(ng.space_to_batch)
```

## Create a simple graph

You can use nGraph's Python API to create computational models. For example, to create a model that represents the formula `y=ReLU(x + b)`, you can use the following code to create parameters `x` and `b` and create the computational graph:


```python
import numpy as np
import ngraph as ng

parameter_x = ng.parameter(shape=[2, 2], dtype=np.float32, name="x")
parameter_b = ng.parameter(shape=[2, 2], dtype=np.float32, name="b")

graph = ng.relu(parameter_x + parameter_b, name="y")
```

nGraph operations are organized into sets, which are released with new versions of OpenVINO. As operation signatures can change between operation sets, it is a good practice to use operations from a specific operation set in your scripts. This ensures that your code is compatible with future versions of OpenVINO.

The example, the code above can be rewritten like this to use only operations from opset 4:

```python
import ngraph.opset4 as ops

parameter_x = ops.parameter(shape=[2, 2], dtype=np.float32, name="x")
parameter_b = ops.parameter(shape=[2, 2], dtype=np.float32, name="b")

graph = ops.relu(parameter_x + parameter_b, name="y")
```


## Create an nGraph function from a graph

You can combine a graph together with its input parameters into an nGraph function object. An nGraph function can be passed to other parts of OpenVINO to perform analysis and inference.

```python
>>> function = ng.Function(graph, [parameter_x, parameter_b], "TestFunction")
<Function: 'TestFunction' ({2,2})>
```

## Run inference

In order to run inference on an nGraph function, convert it to an Inference Engine network and call its `infer` method.

```python
>>> from openvino.inference_engine import IECore
>>> 
>>> ie_network = ng.function_to_cnn(function)
>>> 
>>> ie = IECore()
>>> executable_network = ie.load_network(ie_network, 'CPU')
>>> 
>>> output = executable_network.infer({
>>>    'x': np.array([[-1, 0], [1, 2]]),
>>>    'b': np.array([[-1, 0], [0, 1]]),
>>> })
>>> output['y']
array([[-0.,  0.],
       [ 1.,  3.]], dtype=float32)
```

## Load an nGraph function from a file

You can load a model from a file using the Inference Engine `read_network` method:

    network = ie_core.read_network(model=model_path, weights=weights_path)

An nGraph function can be extracted from an Inference Engine network using the `ngraph.function_from_cnn` method.
The following example shows how an ONNX model can be loaded to an nGraph Function:

```python
from openvino.inference_engine import IECore
ie = IECore()

model_path='/path/to/model.onnx'
network = ie.read_network(model=model_path)

import ngraph as ng
function = ng.function_from_cnn(network)
```


## Inspect an nGraph function

You can use the nGraph function's `get_ordered_ops` method to get a topologically sorted list of its graph `Node`s. 

```python
>>> function.get_ordered_ops()
[<Parameter: 'b' ({2,2}, float)>,
 <Parameter: 'x' ({2,2}, float)>,
 <Add: 'Add_2' ({2,2})>,
 <Relu: 'y' ({2,2})>,
 <Result: 'Result_4' ({2,2})>]
```

Each `Node` has a unique `name` property, assigned at creation time. User-provided names can be retrieved using the `get_friendly_name` method. `get_type_name` returns the operation type of the node and the `shape` property returns the shape of the node's output tensor.

```python
>>> for node in function.get_ordered_ops():
>>>     print('Node name: {:15} Friendly name: {:10} Op: {:10} {}'.format(
>>>         node.name, node.get_friendly_name(), node.get_type_name(), node.shape))
Node name: Parameter_1     Friendly name: b          Op: Parameter  Shape{2, 2}
Node name: Parameter_0     Friendly name: x          Op: Parameter  Shape{2, 2}
Node name: Add_2           Friendly name: Add_2      Op: Add        Shape{2, 2}
Node name: Relu_3          Friendly name: y          Op: Relu       Shape{2, 2}
Node name: Result_4        Friendly name: Result_4   Op: Result     Shape{2, 2}
```

### Node relationships

`Node`'s parents can be retrieved using the `inputs()` method, which returns `Input` objects. Each `Input` is a connection to an upstream `Output`, which is connected to a `Node`. The upstream node can be retrieved using `input.get_source_output().get_node()` methods.

```python
>>> for node_input in node.inputs():
>>>     parent = node_input.get_source_output().get_node()
>>>     print('{} node has parent {}'.format(node.get_friendly_name(), parent.get_friendly_name()))
Add_2 node has parent x
Add_2 node has parent b
```

By analogy, `Node`'s children can be retrieved using the `outputs()` method, which returns a list of `Output` objects. By contrast to inputs, each output can be connected to multiple downstream nodes.

```python
>>> for node_output in node.outputs():
>>>     children = [target.get_node() for target in node_output.get_target_inputs()]
>>>     print('{} node has children {}'.format(node.get_friendly_name(), ','.join([c.get_friendly_name() for c in children])))
>>> 
Add_2 node has children y
```

### Shapes

Each `Input` and `Output` has an associated tensor shape, which describes what data it can accept or produce. Shapes can be dynamic, which means that they are only partially known. You can get information about this shape by calling the `get_partial_shape` method. If the returned `PartialShape` object's `is_dynamic` property returns `True`, then the shape is known partially. Otherwise, you can convert the shape to a fully known static shape (list of integers).

```python
partial_shape = node_input.get_partial_shape()
if partial_shape.is_dynamic:
    print('{} node has a dynamic shape {}'.format(node.get_friendly_name(), partial_shape))
else:
    shape = list(partial_shape.to_shape())
    print('{} node has a static shape {}'.format(node.get_friendly_name(), shape))
```

### Node attributes

Nodes may also have additional attributes. For example, a `Softmax` node has an `axis` attribute. Each attribute may be retrieved using a dedicated method, for example, `node.get_axis()`.

```python
>>> node = ng.softmax(data=[[1, 2], [3, 4]], axis=1)
>>> node.get_axis()
1
```

You can also set attribute values using corresponding setter methods, for example:

```python
node.set_axis(0)
```

Currently, you can get all attributes of a node using the `_get_attributes` method. Please note that this is an internal API method and may change in future versions of OpenVINO.

The following code displays all attributes for all nodes in a function:

```python
for node in function.get_ordered_ops():
    attributes = node._get_attributes()
    if(attributes):
        print('Operation {} of type {} has attributes:'.format(node.get_friendly_name(), node.get_type_name()))
        for attr, value in attributes.items():
            print("  * {}: {}".format(attr, value))
    else:
        print('Operation {} of type {} has no attributes.'.format(node.get_friendly_name(), node.get_type_name()))
```

### Node runtime information

Attributes are properties of nodes in the computational graph, which will be stored when the model is serialized to a file. However, there can be additional properties of nodes, which are only important during the execution of a graph. An example of such a property is `affinity`, which determines which operation will be executed on which hardware in a heterogeneous environment. You can get and set runtime information by using the `get_rt_info` method of a node.

> **NOTE**: If you set affinity manually, be careful at the current moment Inference Engine plugins don't support constant (`Constant`->`Result`) and empty (`Parameter`->`Result`) networks. Please avoid such subgraphs when you set affinity manually.

```python
rt_info = node.get_rt_info()
rt_info["affinity"] = "test_affinity"
```

Please note that the `rt_info` API may change in future versions of OpenVINO.

### Data of Constant nodes

You can retrieve the data tensor associated with a `Constant` node by calling its `get_data` method:

```python
if node.get_type_name() == 'Constant':
    data = node.get_data()
``` 

## Transform graphs

nGraph supports a growing number of transformation passes, which can be applied to a graph. For example, to perform constant folding of applicable subgraphs, you can use the `ConstantFolding` pass. To do this, create an optimization pass `Manager`, register the passes you want to execute, and feed an nGraph function to the configured pass manager.

The following code example illustrates the process:

```python
from ngraph.impl.passes import Manager
pass_manager = Manager()
pass_manager.register_pass('ConstantFolding')
pass_manager.run_passes(function)
```

Please note that the list of available transformations may change in future versions of OpenVINO.
