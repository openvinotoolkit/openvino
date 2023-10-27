# GNA transformations documentation

GNA Plugin provides implementation of multiple methods required by OpenVINO plugin API. Original model usually consists of variety of operations, i.e. Convolution, Add, Gather, LSTMSequence and so on. GNA Hardware is its own limitation and not all operations can be executed on GNA Hardware.
One of the main functionalities for GNA Plugin is conversion of input network to equivalent network which could be executed on the GNA hardware. This conversion is realized by LoadNetwok method.

## LoadNetwork

GNAPlugin::LoadNetwork in the future should execute following stages:
-	Converting input graph to fully GNA-supported graph (all in ngraph)
-	Creating and connecting GNA primitives within libGNA from ngraph-based network

These stages include:
-	Obtain ngraph-based network from the CNNNetwork argument (if input is not ngraph-based, proceed to CNNNetwork passes stage)
-	Pass ngraph-based network through ngraph-based transformations.
-	Convert ngraph-based network to CNNNetwork-based
-	Pass network through CNNNetwork-based transformations.
-	Creating and connecting GNA primitives withing libGNA from CNNNetwork-bases graph
Transformations are the way of modifying input graph. Ngraph-based transformations usually are of the following types:
-	inherited from ov::pass::ModelPass. They implement run_on_model method. It allows them to be a container of other transformations. For example, ngraph::pass::CommonOptimizations executes multiple transformations in it. Each of them do some basic transformations.
-	inherited from ov::pass::MatcherPass. Such transformations usually have a constructor. That constructor defines a pattern with the several connected together layers and a function that modifies found group of layers. The pattern can also handle additional predicates that do any checks on the traversed nodes. It is preferable to use that predicates explicitly rather than check and return from the transform function.
GNA-specific ngraph-based transformations are placed in src/plugins/intel_gna/src/transformations. All transformations should have brief comments in their headers. That brief should describe what pattern transformation handles and what modifications do.
There is also a directory src/plugins/intel_gna/src/transformations/rt_info with auxiliary runtime attributes. That attributes could be added into node rt_info map. That attributes can be read/write in transformations which is useful in some cases. For example, transformation can proceed some node in a special way if the node has special attribute.
All new transformations should have unit tests, that are placed in src/plugins/intel_gna/tests/unit/transformations. All that unit tests are compiled in ov_gna_unit_tests binary.
CNNNetwork transformations are so-called passes. They are placed in src/plugins/intel_gna/src/optimizer/gna_pass_manager.cpp. Passes proceed network as a
```
std::vector<InferenceEngine::CNNLayerPtr> * pLayers
```
It is preferrable to write new transformations as nGraph passes and avoid implementing CNNNetwork passes. All CNNNetwork related code is considered as a legacy. Existed CNNNetwork passes are ported to ngraph.

## GNA ngraph-based layers

OpenVino allows to work with graph nodes as ov::Node class instances. Most of them are stored in src/core/include/openvino/op directory and could be used by all plugins. GNA plugin stores own (GNA-specific) layer types.
1.	src/plugins/intel_gna/legacy/include/legacy/ngraph_ops
Here there are legacy layer types. Their names ends with “IE”. These types cannot be in graph, that pass to GNA plugin. All of these types are created within GNA transformations and used in GNA graph compiler for creating libGNA primitives. There are plans to rewrite all legacy code. These legacy types should be removed after that.
2.	src/plugins/intel_gna/src/ops
GNA-specific operations. For example, GNAConvolution type describes convolution layers. It differs from common OpenVino Convolution type as it handles NHWC data layout instead of NCHW.
Ngraph-based transformations
1.	Transformations that are common for all OpenVino plugins (are placed outside GNA plugin directory). These transformations perform different optimizations. For example, ov::pass::ConvertDivide transforms Divide operation into the sequence of nodes with Power layer. LSTMCellDecomposition extracts LSTMCell into subgraph of mathematical operations.
2.	Transformations that are specific for the GNA plugin (are placed inside GNA plugin directory)
They also include src/plugins/intel_gna/legacy/include/legacy/transformations/convert_opset1_to_legacy directory with ngraph-based legacy transformations. These transformations produce “IE” layers. After rewriting GNA legacy code these transformations should be removed.

### "Layout transformations"
There are group of transformations that work with data layout. GNA-hardware supports MaxPool and Convolution operations in a different way in comparison to OpenVino common types. GNA supports NHWC layout, OpenVino supports NCHW layout.
There are group of transformations ReplaceGnaNHWCLayers that substitutes common types with NCHW layout to GNA-specific types with NHWC layout. It is done with wrapping GNA-types with transpose operations, that converts layout on input and output of GNA-types. Unfortunately, in most situations GNA hardware cannot execute these transpose operations. To solve this issue, there are transformations that allows to push transposes through layers from GNA-specific NHWC layers to the start and end of the graph, exchanging Transpose/Gather layer with neighbor layer. Some of them (for example, TransposeSinking group of transformations) allows to push transpose layers through multiple layer types. These transformations are common for all OpenVino and stores outside GNA plugin code. They are not able to push Transpose layer through Reshape type nodes due to mathematical reasons.
To push Transpose operation through Reshape nodes there are transformations that substitute Transpose + Reshape pattern with Reshape + Gather. Gather operation is not supported by the GNA hardware and it should also be pushed through the graph to the start and end. There are group of transformations that does it.
Transpose/Gather sinking consists of multiple transformations. Each of these transformations works with a small pattern consisting of Transpose/Gather and a node with a specific kind of layers (for example, with binary elementwise operations). Sinking transformation interchanges layers. After each sinking transformation execution Transpose/Gather layer moves through one layer in the graph. There are multiple nodes between start/end of the graph and initial Transpose/Gather layer position. Node types can repeat multiple times while sinking and are going in a arbitrary order. The same Transpose/Sinking transformation should be executed multiple times. They use register_new_node functionality. This method adds new created Transpose/Gather node at the end of the matcher pass queue to allow the same transformation be executed once again without necessity to call it implicitly once again.
TransposeSinking changes Concat and Split axis while pushing Transpose nodes through them. GNA doesn't support all possible Concat and Split axis. Some TransposeSinking transformations support callbacks. These callbacks are executed inside transformations and allow to add plugin specific checks. In these checks, GNA plugin prevents sinking transposes that would make some Split/Concats unsupported.
As Transpose and Gather layers are moved to start and end of the graph they are cut from the graph and moved to ov::intel_gna::PrePostProcessModels structure as separate models. On each network inference plugin searches in this structure model for input/output, executes this model on CPU and copy resulted data as input/output of the entire model.
	TransposeSinking group of transformations doesn’t support currently StridedSlice layer. It leads to the next problem.
GNA plugin has the following Slice layer flow:
-	SliceToStridedSlice transformation in CommonOptimizations converts Slice to StridedSlice
-	ConvertStridedSliceToCropMatcher transformation convers StridedSlice to CropIE
-	convertFunctionToICNNNetwork converts CropIE to CNNNetwork CropLayer
-	GNA graph compiler converts CropLayer into affine layer
Since TransposeSInking is called after common optimizations it cannot push Transpose through the StridedSlice. If we have Slice operation in the original model we should prevent converting Slice to StridedSlice in common optimization. It is done by next steps:
-	Disable execution of SliceToStridedSlice transformation
-	Execute entire set of ngraph-based transformations
-	Execute a set of transformations to convert Slice -> StridedSlice -> CropIE nodes
When StridedSlice layer will be supported by TransposeSInking these steps could be removed from GNA plugin pipeline.

## CNNNetwork based passes

After running ngraph-based transformations model is converted with function convertFunctionToICNNNetwork into CNNNetwork-based function. The next step is the model transformation with the CNNNetwork-based passes.
All the legacy CNNNetwork-based passes are stored in src/plugins/intel_gna/src/optimizer/gna_pass_manager.cpp. One of the main difference between legacy passes and ngraph transformations is that legacy passes doesn’t have pattern matching functionality. Each of the passes iterating through the graph nodes (previously sorting toplogical) searching for sought sequence of layers and modify them.
It should be mentioned that ngraph API stores constant data as input nodes with type Constant, but CNNNetwork API stores data as a BLOB in layer info.

## Debugging

There is an ability to dump model between transformations/passes.
To dump CNNNetwork passes use -DENABLE_INTEL_GNA_DEBUG=ON option to cmake build configuration. After plugin execution, *.dot files representing the final graph will be saved in the current working directory; *.dot files can be converted to an image with the graphviz dot executable, for example:
```
dot -Tpng <dot_filename> -o <image.png>
```
To dump CNNNetwork-based model in xml add 
```
#define ENABLE_V7_SERIALIZE
```
to src/plugins/intel_gna/src/log/debug.hpp

To dump model between ngraph-based transformations use VisualizeTree and Serialize transformations.

### VisualizeTree

VisualizeTree transformation allows to dump model as image.
```
#include "openvino/pass/visualize_tree.hpp"
manager.register_pass<ov::pass::VisualizeTree>("./dump.png");
```

### Serialize

Serialize transformation allows to dump model as xml and binary files that could be loaded in neutron web application
```
#include "openvino/pass/serialize.hpp"
manager.register_pass<ov::pass::Serialize>("./dump.xml", "./dump.bin");
```
Where, manager is the ov::pass::Manager instance.
