# Troubleshooting {#ovtf_troubleshooting}


# Tips for Troubleshooting Failing models
If the model fails to run on **OpenVINO™ Integration with TensorFlow** or you are having a performance/accuracy issue, follow the steps below to debug the issue.

## 1. Activating Log Messages

There are two types of log messages in **OpenVINO™ Integration with TensorFlow**. The first one will show you the steps and detailed information about the graph clustering phase. You can activate this log placement via an environment variable:

    OPENVINO_TF_LOG_PLACEMENT=1

After activating the log placement, you will see several lines of output about the clustering and encapsulating phase. To check the brief summary of how the clusters are formed, see the lines starting with "OVTF_SUMMARY". These lines will provide the overall statistics like the reasons for clustering, the number of nodes clustered/declustered, the total number of clusters, the number of nodes per clusters, etc.

```
OVTF_SUMMARY: Summary of reasons why a pair of edge connected encapsulates did not merge
OVTF_SUMMARY: DEADNESS: 1466, STATICINPUT: 203, PATHEXISTS: 1033
OVTF_SUMMARY: Summary of reasons why a pair of edge connected clusters did not merge
OVTF_SUMMARY: NOTANOP: 173, UNSUPPORTED: 722, DEADNESS: 1466, SAMECLUSTER: 4896, STATICINPUT: 203, PATHEXISTS: 1033

OVTF_SUMMARY: Number of nodes in the graph: 4250
OVTF_SUMMARY: Number of nodes marked for clustering: 3943 (92% of total nodes)
OVTF_SUMMARY: Number of nodes assigned a cluster: 2093 (49% of total nodes)      (53% of nodes marked for clustering) 
OVTF_SUMMARY: Number of ngraph clusters :3
OVTF_SUMMARY: Nodes per cluster: 697.667
OVTF_SUMMARY: Size of nGraph Cluster[0]:        1351
OVTF_SUMMARY: Size of nGraph Cluster[35]:       361
OVTF_SUMMARY: Size of nGraph Cluster[192]:      381
OVTF_SUMMARY: Op_deassigned:  Gather -> 549, Reshape -> 197, Cast -> 183, Const -> 152, StridedSlice -> 107, Shape -> 106, Add -> 98, ZerosLike -> 92, Minimum -> 91, Split -> 90, NonMaxSuppressionV2 -> 90, Mul -> 13, Identity -> 12, Range -> 12, ConcatV2 -> 9, Pack -> 7, Sub -> 7, ExpandDims -> 6, Fill -> 4, Unpack -> 4, Transpose -> 3, Slice -> 3, Greater -> 3, Equal -> 2, Exp -> 2, Less -> 2, Squeeze -> 2, Tile -> 1, Size -> 1, Sigmoid -> 1, TopKV2 -> 1
```

The initial lines of the log are related to assigning clusters. At this stage, all nodes in the graph are visited and cluster borders are assigned. An output log like below means that the corresponding couple of consecutive nodes are grouped into the same cluster:

    NONCONTRACTION: SAMECLUSTER: Preprocessor/mul/x<Const>[0] -> Preprocessor/mul<Mul>[1]

A line like below tells that these nodes cannot be grouped into the same cluster since the second node is not supported by OpenVINO™.

    NONCONTRACTION: UNSUPPORTED: MultipleGridAnchorGenerator/assert_equal/Assert/Assert<Assert>[-1] -> Postprocessor/ExpandDims<ExpandDims>[-1]

Apart from the log placement, you can enable the VLOG level to print details from the runtime execution. VLOG can be set to any level from 1 to 5. Setting VLOG to 1 will print the least amount of detail and setting it to 5 will print the most detailed logs. For example, you can set the VLOG level to 1 by setting the environment variable below:

    OPENVINO_TF_VLOG_LEVEL=1

This will print some details for each cluster executed on **OpenVINO™ integration with TensorFlow** like below:

    OPENVINO_TF_MEM_PROFILE:  OP_ID: 0 Step_ID: 8 Cluster: ovtf_cluster_0 Input Tensors created: 0 MB Total process memory: 1 GB
    OPENVINO_TF_TIMING_PROFILE: OP_ID: 0 Step_ID: 8 Cluster: ovtf_cluster_0 Time-Compute: 10 Function-Create-or-Lookup: 0 Create-and-copy-tensors: 0 Execute: 10

## 2. Dumping Graphs/Clusters

To dump the full graph in each step of the clustering, set the environment variable below:

    OPENVINO_TF_DUMP_GRAPHS=1

This will serialize the TensorFlow graph in each step of the clustering in "pbtxt" format.

- unmarked_<graph_id>.pbtxt: This is the initial graph before the optimization pass of **OpenVINO™ Integration with TensorFlow**.
- marked_<graph_id>.pbtxt: This is the graph after the supported nodes are marked.
- clustered_<graph_id>.pbtxt: This is the graph after the clustering is completed. All the supported nodes should be grouped into clusters after this step.
- declustered_<graph_id>.pbtxt: This is the graph after some of the clusters are deassigned. For example, the clusters with very low number of ops will be deassigned after this step.
- encapsulated_<graph_id>.pbtxt: This is the graph after the encapsulation is completed. Each of the existing clusters should be encapsulated into a "_nGraphEncapsulate" op.

The OpenVINO™  Intermediate Representation (IR) files ("ovtf_cluster_<cluster_id>.xml" and "ovtf_cluster_<cluster_id>.bin") will be serialized for each of the created cluster.

Also, the TensorFlow graph for each cluster can be serialized for further debugging by setting the environment variable below:

    OPENVINO_TF_DUMP_CLUSTERS=1

This will generate one file ("ovtf_cluster_<cluster_id>.pbtxt") for each of the clusters generated.


## 3. Disabling Nodes
Disable the nodes which cause the issue. If you are able to identify the operator types causing the issue, you can try disabling that particular type of operator. You can set the environment variable "OPENVINO_TF_DISABLED_OPS" to disable the operators which are causing the problem (see the example below).

    OPENVINO_TF_DISABLED_OPS="Squeeze,Greater,Gather,Unpack"

## 4. Setting Cluster Size Limit
If there are multiple clusters executing and smaller clusters are causing the issue, you can set the cluster size limit which will only execute the larger clusters on OpenVINO™. This way, the smaller clusters will be executed on native TensorFlow and you can still have the performance benefit of executing larger clusters on OpenVINO™. You should set the cluster size limit by setting the environment variable below. Adjust the value that works best for your model (see the example below).

    OPENVINO_TF_MIN_NONTRIVIAL_NODES=25

## 5. Optimizing Keras Models for OpenVINO™ integration with TensorFlow

Some Keras models may include training ops which leads TensorFlow to produce control flow ops. Since the control flow ops are not supported by OpenVINO™, the graph might be partitioned to smaller clusters. Freezing the model can remove these ops and improve the overall performance using **OpenVINO™ integration with TensorFlow**.

Below is a sample inference application with DenseNet121 using Keras API.

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.densenet import DenseNet121
# Add two lines of code to enable OpenVINO integration with TensorFlow
import openvino_tensorflow
openvino_tensorflow.set_backend("CPU")
.
.
.
model = DenseNet121(weights='imagenet')
.
.
.
# Run the inference using Keras API    
model.predict(input_data)
```

Below is a sample code to freeze and run a Keras model to optimize it further to achieve the best performance using **OpenVINO™ integration with TensorFlow**.

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
# Add two lines of code to enable OpenVINO integration with TensorFlow
import openvino_tensorflow
openvino_tensorflow.set_backend("CPU")
.
.
.
model = DenseNet121(weights='imagenet')
.
.
.
# Freeze the model first to achieve the best performance using OpenVINO integration with TensorFlow    
full_model = tf.function(lambda x: self.model(x))
full_model = full_model.get_concrete_function(tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype, name=model.inputs[0].name))
frozen_func = convert_variables_to_constants_v2(full_model)
frozen_func.graph.as_graph_def()
session = tf.compat.v1.Session(graph=frozen_func.graph)
prob_tensor = frozen_func.graph.get_tensor_by_name(full_model.outputs[0].name)
.
.
.
# Run the inference on the frozen model
session.run(prob_tensor, feed_dict={full_model.inputs[0].name : input_data})
```