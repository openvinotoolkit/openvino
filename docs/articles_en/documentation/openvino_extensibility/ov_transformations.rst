.. {#openvino_docs_transformations}

Overview of Transformations API
===============================


.. meta::
   :description: Learn how to apply additional model optimizations or transform 
                 unsupported subgraphs and operations, using OpenVINO™ Transformations API.


.. toctree::
   :maxdepth: 1
   :hidden:

   openvino_docs_Extensibility_UG_model_pass
   openvino_docs_Extensibility_UG_matcher_pass
   openvino_docs_Extensibility_UG_graph_rewrite_pass

OpenVINO Transformation mechanism allows to develop transformation passes to modify ``:ref:`ov::Model <doxid-classov_1_1_model>```. You can use this mechanism to apply additional optimizations to the original Model or transform unsupported subgraphs and operations to new operations which are supported by the plugin.
This guide contains all necessary information that you need to start implementing OpenVINO™ transformations.

Working with Model
##################

Before the moving to transformation part it is needed to say several words about functions which allow to modify ``:ref:`ov::Model <doxid-classov_1_1_model>```.
This chapter extends the :doc:`model representation guide <openvino_docs_OV_UG_Model_Representation>` and shows an API that allows us to manipulate with ``:ref:`ov::Model <doxid-classov_1_1_model>```.

Working with node input and output ports
++++++++++++++++++++++++++++++++++++++++

First of all let's talk about ``:ref:`ov::Node <doxid-classov_1_1_node>``` input/output ports. Each OpenVINO™ operation has input and output ports except cases when operation has ``Parameter`` or ``Constant`` type.

Every port belongs to its node, so using a port we can access parent node, get shape and type for particular input/output, get all consumers in case of output port, and get producer node in case of input port.
With output port we can set inputs for newly created operations.

Lets look at the code example.

.. doxygensnippet:: docs/snippets/ov_model_snippets.cpp
   :language: cpp
   :fragment: [ov:ports_example]

Node replacement
++++++++++++++++

OpenVINO™ provides two ways for node replacement: via OpenVINO™ helper function and directly via port methods. We are going to review both of them.

Let's start with OpenVINO™ helper functions. The most popular function is ``ov::replace_node(old_node, new_node)``.

We will review real replacement case where Negative operation is replaced with Multiply.

.. image:: ./_static/images/ngraph_replace_node.png 

.. doxygensnippet:: docs/snippets/ov_model_snippets.cpp
   :language: cpp
   :fragment: [ov:replace_node]

``:ref:`ov::replace_node <doxid-namespaceov_1a75d84ee654edb73fe4fb18936a5dca6d>``` has a constraint that number of output ports for both of ops must be the same; otherwise, it raises an exception.

The alternative way to do the same replacement is the following:

.. doxygensnippet:: docs/snippets/ov_model_snippets.cpp
   :language: cpp
   :fragment: [ov:manual_replace]

Another transformation example is insertion.

.. image:: ./_static/images/ngraph_insert_node.png

.. doxygensnippet:: docs/snippets/ov_model_snippets.cpp
   :language: cpp
   :fragment: [ov:insert_node]

The alternative way to the insert operation is to make a node copy and use ``:ref:`ov::replace_node() <doxid-namespaceov_1a75d84ee654edb73fe4fb18936a5dca6d>```:

.. doxygensnippet:: docs/snippets/ov_model_snippets.cpp
   :language: cpp
   :fragment: [ov:insert_node_with_copy]

Node elimination
++++++++++++++++

Another type of node replacement is its elimination.

To eliminate operation, OpenVINO™ has special method that considers all limitations related to OpenVINO™ Runtime.

.. doxygensnippet:: docs/snippets/ov_model_snippets.cpp
   :language: cpp
   :fragment: [ov:eliminate_node]

``:ref:`ov::replace_output_update_name() <doxid-namespaceov_1a75ba2120e573883bd96bb19c887c6a1d>``` in case of successful replacement it automatically preserves friendly name and runtime info.

.. _transformations_types:

Transformations types 
#####################

OpenVINO™ Runtime has three main transformation types:

* :doc:`Model pass <openvino_docs_Extensibility_UG_model_pass>` - straightforward way to work with ``:ref:`ov::Model <doxid-classov_1_1_model>``` directly
* :doc:`Matcher pass <openvino_docs_Extensibility_UG_matcher_pass>` - pattern-based transformation approach
* :doc:`Graph rewrite pass <openvino_docs_Extensibility_UG_graph_rewrite_pass>` - container for matcher passes needed for efficient execution

.. image:: ./_static/images/transformations_structure.png

Transformation conditional compilation
######################################

Transformation library has two internal macros to support conditional compilation feature.

* ``:ref:`MATCHER_SCOPE(region) <doxid-conditional__compilation_2include_2openvino_2cc_2pass_2itt_8hpp_1a3d1377542bcf3e305c33a1b683cc77df>``` - allows to disable the MatcherPass if matcher isn't used. The region name should be unique. This macro creates a local variable ``matcher_name`` which you should use as a matcher name.
* ``:ref:`RUN_ON_MODEL_SCOPE(region) <doxid-conditional__compilation_2include_2openvino_2cc_2pass_2itt_8hpp_1ab308561b849d47b9c820506ec73c4a30>``` - allows to disable run_on_model pass if it isn't used. The region name should be unique.

.. _transformation_writing_essentials:

Transformation writing essentials 
#################################

When developing a transformation, you need to follow these transformation rules:

1. Friendly Names
+++++++++++++++++

Each ``:ref:`ov::Node <doxid-classov_1_1_node>``` has an unique name and a friendly name. In transformations we care only about friendly name because it represents the name from the model.
To avoid losing friendly name when replacing node with other node or subgraph, set the original friendly name to the latest node in replacing subgraph. See the example below.

.. doxygensnippet:: docs/snippets/ov_model_snippets.cpp
   :language: cpp
   :fragment: [ov:replace_friendly_name]

In more advanced cases, when replaced operation has several outputs and we add additional consumers to its outputs, we make a decision how to set friendly name by arrangement.

2. Runtime Info
+++++++++++++++

Runtime info is a map ``std::map<std::string, :ref:`ov::Any <doxid-classov_1_1_any>`>`` located inside ``:ref:`ov::Node <doxid-classov_1_1_node>``` class. It represents additional attributes in ``:ref:`ov::Node <doxid-classov_1_1_node>```.
These attributes can be set by users or by plugins and when executing transformation that changes ``:ref:`ov::Model <doxid-classov_1_1_model>``` we need to preserve these attributes as they will not be automatically propagated.
In most cases, transformations have the following types: 1:1 (replace node with another node), 1:N (replace node with a sub-graph), N:1 (fuse sub-graph into a single node), N:M (any other transformation).
Currently, there is no mechanism that automatically detects transformation types, so we need to propagate this runtime information manually. See the examples below.


.. doxygensnippet:: docs/snippets/ov_model_snippets.cpp
   :language: cpp
   :fragment: [ov:copy_runtime_info]

When transformation has multiple fusions or decompositions, ``:ref:`ov::copy_runtime_info <doxid-namespaceov_1a3bb5969a95703b4b4fd77f6f58837207>``` must be called multiple times for each case.

.. note:: ``copy_runtime_info`` removes ``rt_info`` from destination nodes. If you want to keep it, you need to specify them in source nodes like this: ``copy_runtime_info({a, b, c}, {a, b})``

3. Constant Folding
+++++++++++++++++++

If your transformation inserts constant sub-graphs that need to be folded, do not forget to use ``:ref:`ov::pass::ConstantFolding() <doxid-classov_1_1pass_1_1_constant_folding>``` after your transformation or call constant folding directly for operation.
The example below shows how constant subgraph can be constructed.

.. doxygensnippet:: docs/snippets/ov_model_snippets.cpp
   :language: cpp
   :fragment: [ov:constant_subgraph]

Manual constant folding is more preferable than ``:ref:`ov::pass::ConstantFolding() <doxid-classov_1_1pass_1_1_constant_folding>``` because it is much faster.

Below you can find an example of manual constant folding:

.. doxygensnippet:: docs/snippets/template_pattern_transformation.cpp
   :language: cpp
   :fragment: [manual_constant_folding]

.. _common_mistakes:

Common mistakes in transformations 
##################################

In transformation development process:

* Do not use deprecated OpenVINO™ API. Deprecated methods has the ``OPENVINO_DEPRECATED`` macros in its definition.
* Do not pass ``shared_ptr<Node>`` as an input for other node if type of node is unknown or it has multiple outputs. Use explicit output port.
* If you replace node with another node that produces different shape, remember that new shape will not be propagated until the first ``validate_nodes_and_infer_types`` call for ``:ref:`ov::Model <doxid-classov_1_1_model>```. If you are using ``:ref:`ov::pass::Manager <doxid-classov_1_1pass_1_1_manager>```, it will automatically call this method after each transformation execution.
* Do not forget to call the ``:ref:`ov::pass::ConstantFolding <doxid-classov_1_1pass_1_1_constant_folding>``` pass if your transformation creates constant subgraphs.
* Use latest OpSet if you are not developing downgrade transformation pass.
* When developing a callback for ``:ref:`ov::pass::MatcherPass <doxid-classov_1_1pass_1_1_matcher_pass>```,  do not change nodes that come after the root node in topological order.

.. _using_pass_manager:

Using pass manager
##################

``:ref:`ov::pass::Manager <doxid-classov_1_1pass_1_1_manager>``` is a container class that can store the list of transformations and execute them. The main idea of this class is to have high-level representation for grouped list of transformations.
It can register and apply any `transformation pass <#transformations_types>`__ on model.
In addition, ``:ref:`ov::pass::Manager <doxid-classov_1_1pass_1_1_manager>``` has extended debug capabilities (find more information in the `how to debug transformations <#how_to_debug_transformations>`__ section).

The example below shows basic usage of ``:ref:`ov::pass::Manager <doxid-classov_1_1pass_1_1_manager>```

.. doxygensnippet:: docs/snippets/template_pattern_transformation.cpp
   :language: cpp
   :fragment: [matcher_pass:manager3]

Another example shows how multiple matcher passes can be united into single GraphRewrite.

.. doxygensnippet:: docs/snippets/template_pattern_transformation.cpp
   :language: cpp
   :fragment: [matcher_pass:manager2]
   
.. _how_to_debug_transformations:   

How to debug transformations 
############################

If you are using ``ngraph::pass::Manager`` to run sequence of transformations, you can get additional debug capabilities by using the following environment variables:

.. code-block:: cpp
   
   OV_PROFILE_PASS_ENABLE=1 - enables performance measurement for each transformation and prints execution status
   OV_ENABLE_VISUALIZE_TRACING=1 -  enables visualization after each transformation. By default, it saves dot and svg files.


.. note:: Make sure that you have dot installed on your machine; otherwise, it will silently save only dot file without svg file.

See Also
########

* :doc:`OpenVINO™ Model Representation <openvino_docs_OV_UG_Model_Representation>` 
* :doc:`OpenVINO™ Extensions <openvino_docs_Extensibility_UG_Intro>`

