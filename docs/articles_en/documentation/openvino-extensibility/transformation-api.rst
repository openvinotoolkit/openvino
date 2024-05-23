.. {#openvino_docs_transformations}

Overview of Transformations API
===============================


.. meta::
   :description: Learn how to apply additional model optimizations or transform
                 unsupported subgraphs and operations, using OpenVINO™ Transformations API.


.. toctree::
   :maxdepth: 1
   :hidden:

   transformation-api/model-pass
   transformation-api/matcher-pass
   transformation-api/graph-rewrite-pass

OpenVINO Transformation mechanism allows to develop transformation passes to modify ``ov::Model``. You can use this mechanism to apply additional optimizations to the original Model or transform unsupported subgraphs and operations to new operations which are supported by the plugin.
This guide contains all necessary information that you need to start implementing OpenVINO™ transformations.

Working with Model
##################

Before the moving to transformation part it is needed to say several words about functions which allow to modify ``ov::Model``.
This chapter extends the :doc:`model representation guide <../../openvino-workflow/running-inference/integrate-openvino-with-your-application/model-representation>` and shows an API that allows us to manipulate with ``ov::Model``.

Working with node input and output ports
++++++++++++++++++++++++++++++++++++++++

First of all let's talk about ``ov::Node`` input/output ports. Each OpenVINO™ operation has input and output ports except cases when operation has ``Parameter`` or ``Constant`` type.

Every port belongs to its node, so using a port we can access parent node, get shape and type for particular input/output, get all consumers in case of output port, and get producer node in case of input port.
With output port we can set inputs for newly created operations.

Lets look at the code example.

.. doxygensnippet:: docs/articles_en/assets/snippets/ov_model_snippets.cpp
   :language: cpp
   :fragment: [ov:ports_example]

Node replacement
++++++++++++++++

OpenVINO™ provides two ways for node replacement: via OpenVINO™ helper function and directly via port methods. We are going to review both of them.

Let's start with OpenVINO™ helper functions. The most popular function is ``ov::replace_node(old_node, new_node)``.

We will review real replacement case where Negative operation is replaced with Multiply.

.. image:: ../../assets/images/ov_replace_node.png

.. doxygensnippet:: docs/articles_en/assets/snippets/ov_model_snippets.cpp
   :language: cpp
   :fragment: [ov:replace_node]

``ov::replace_node`` has a constraint that number of output ports for both of ops must be the same; otherwise, it raises an exception.

The alternative way to do the same replacement is the following:

.. doxygensnippet:: docs/articles_en/assets/snippets/ov_model_snippets.cpp
   :language: cpp
   :fragment: [ov:manual_replace]

Another transformation example is insertion.

.. image:: ../../assets/images/ov_insert_node.png

.. doxygensnippet:: docs/articles_en/assets/snippets/ov_model_snippets.cpp
   :language: cpp
   :fragment: [ov:insert_node]

The alternative way to the insert operation is to make a node copy and use ``ov::replace_node()``:

.. doxygensnippet:: docs/articles_en/assets/snippets/ov_model_snippets.cpp
   :language: cpp
   :fragment: [ov:insert_node_with_copy]

Node elimination
++++++++++++++++

Another type of node replacement is its elimination.

To eliminate operation, OpenVINO™ has special method that considers all limitations related to OpenVINO™ Runtime.

.. doxygensnippet:: docs/articles_en/assets/snippets/ov_model_snippets.cpp
   :language: cpp
   :fragment: [ov:eliminate_node]

``ov::replace_output_update_name()`` in case of successful replacement it automatically preserves friendly name and runtime info.

.. _transformations_types:

Transformations types
#####################

OpenVINO™ Runtime has three main transformation types:

* :doc:`Model pass <transformation-api/model-pass>` - straightforward way to work with ``ov::Model`` directly
* :doc:`Matcher pass <transformation-api/matcher-pass>` - pattern-based transformation approach
* :doc:`Graph rewrite pass <transformation-api/graph-rewrite-pass>` - container for matcher passes needed for efficient execution

.. image:: ../../assets/images/transformations_structure.png

Transformation conditional compilation
######################################

Transformation library has two internal macros to support conditional compilation feature.

* ``MATCHER_SCOPE(region)`` - allows to disable the MatcherPass if matcher isn't used. The region name should be unique. This macro creates a local variable ``matcher_name`` which you should use as a matcher name.
* ``RUN_ON_MODEL_SCOPE(region)`` - allows to disable run_on_model pass if it isn't used. The region name should be unique.

.. _transformation_writing_essentials:

Transformation writing essentials
#################################

When developing a transformation, you need to follow these transformation rules:

1. Friendly Names
+++++++++++++++++

Each ``ov::Node`` has an unique name and a friendly name. In transformations we care only about friendly name because it represents the name from the model.
To avoid losing friendly name when replacing node with other node or subgraph, set the original friendly name to the latest node in replacing subgraph. See the example below.

.. doxygensnippet:: docs/articles_en/assets/snippets/ov_model_snippets.cpp
   :language: cpp
   :fragment: [ov:replace_friendly_name]

In more advanced cases, when replaced operation has several outputs and we add additional consumers to its outputs, we make a decision how to set friendly name by arrangement.

2. Runtime Info
+++++++++++++++

Runtime info is a map ``std::map<std::string, ov::Any>`` located inside ``ov::Node`` class. It represents additional attributes in ``ov::Node``.
These attributes can be set by users or by plugins and when executing transformation that changes ``ov::Model`` we need to preserve these attributes as they will not be automatically propagated.
In most cases, transformations have the following types: 1:1 (replace node with another node), 1:N (replace node with a sub-graph), N:1 (fuse sub-graph into a single node), N:M (any other transformation).
Currently, there is no mechanism that automatically detects transformation types, so we need to propagate this runtime information manually. See the examples below.


.. doxygensnippet:: docs/articles_en/assets/snippets/ov_model_snippets.cpp
   :language: cpp
   :fragment: [ov:copy_runtime_info]

When transformation has multiple fusions or decompositions, ``ov::copy_runtime_info`` must be called multiple times for each case.

.. note:: ``copy_runtime_info`` removes ``rt_info`` from destination nodes. If you want to keep it, you need to specify them in source nodes like this: ``copy_runtime_info({a, b, c}, {a, b})``

3. Constant Folding
+++++++++++++++++++

If your transformation inserts constant sub-graphs that need to be folded, do not forget to use ``ov::pass::ConstantFolding()`` after your transformation or call constant folding directly for operation.
The example below shows how constant subgraph can be constructed.

.. doxygensnippet:: docs/articles_en/assets/snippets/ov_model_snippets.cpp
   :language: cpp
   :fragment: [ov:constant_subgraph]

Manual constant folding is more preferable than ``ov::pass::ConstantFolding()`` because it is much faster.

Below you can find an example of manual constant folding:

.. doxygensnippet:: docs/articles_en/assets/snippets/template_pattern_transformation.cpp
   :language: cpp
   :fragment: [manual_constant_folding]

.. _common_mistakes:

Common mistakes in transformations
##################################

In transformation development process:

* Do not use deprecated OpenVINO™ API. Deprecated methods has the ``OPENVINO_DEPRECATED`` macros in its definition.
* Do not pass ``shared_ptr<Node>`` as an input for other node if type of node is unknown or it has multiple outputs. Use explicit output port.
* If you replace node with another node that produces different shape, remember that new shape will not be propagated until the first ``validate_nodes_and_infer_types`` call for ``ov::Model``. If you are using ``ov::pass::Manager``, it will automatically call this method after each transformation execution.
* Do not forget to call the ``ov::pass::ConstantFolding`` pass if your transformation creates constant subgraphs.
* Use latest OpSet if you are not developing downgrade transformation pass.
* When developing a callback for ``ov::pass::MatcherPass``,  do not change nodes that come after the root node in topological order.

.. _using_pass_manager:

Using pass manager
##################

``ov::pass::Manager`` is a container class that can store the list of transformations and execute them. The main idea of this class is to have high-level representation for grouped list of transformations.
It can register and apply any `transformation pass <#transformations_types>`__ on model.
In addition, ``ov::pass::Manager`` has extended debug capabilities (find more information in the `how to debug transformations <#how_to_debug_transformations>`__ section).

The example below shows basic usage of ``ov::pass::Manager``

.. doxygensnippet:: docs/articles_en/assets/snippets/template_pattern_transformation.cpp
   :language: cpp
   :fragment: [matcher_pass:manager3]

Another example shows how multiple matcher passes can be united into single GraphRewrite.

.. doxygensnippet:: docs/articles_en/assets/snippets/template_pattern_transformation.cpp
   :language: cpp
   :fragment: [matcher_pass:manager2]

.. _how_to_debug_transformations:

How to debug transformations
############################

If you are using ``ov::pass::Manager`` to run sequence of transformations, you can get additional debug capabilities by using the following environment variables:

.. code-block:: cpp

   OV_PROFILE_PASS_ENABLE=1 - enables performance measurement for each transformation and prints execution status
   OV_ENABLE_VISUALIZE_TRACING=1 -  enables visualization after each transformation. By default, it saves dot and svg files.


.. note:: Make sure that you have dot installed on your machine; otherwise, it will silently save only dot file without svg file.

See Also
########

* :doc:`OpenVINO™ Model Representation <../../openvino-workflow/running-inference/integrate-openvino-with-your-application/model-representation>`
* :doc:`OpenVINO™ Extensions <../openvino-extensibility>`

