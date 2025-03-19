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
   transformation-api/patterns-python-api

OpenVINO Transformation mechanism allows to develop transformation passes to modify ``ov::Model``.
You can use this mechanism to apply additional optimizations to the original Model or transform
unsupported subgraphs and operations to new operations supported by the plugin.
This guide contains all the necessary information to start implementing OpenVINO™ transformations.

Working with Model
##################

Before moving to the transformation part, it is important to say a few words about the functions which allow modifying ``ov::Model``.
This section extends the :doc:`model representation guide <../../openvino-workflow/running-inference/model-representation>`
and introduces an API for ``ov::Model`` manipulation.

Working with node input and output ports
++++++++++++++++++++++++++++++++++++++++

Each OpenVINO operation has ``ov::Node`` input and output ports, except for ``Parameter`` and ``Constant`` types.
The terms ``node`` and ``operation`` are used interchangeably in OpenVINO, but this article will maintain consistency in their use.

Every port is associated with a node, allowing access to the node it belongs to, including
its shape, type, all consumers for output ports and the producer node for input ports.

Take a look at the code example:

.. doxygensnippet:: docs/articles_en/assets/snippets/ov_model_snippets.cpp
   :language: cpp
   :fragment: [ov:ports_example]

Node replacement
++++++++++++++++

OpenVINO™ provides two ways for node replacement: via OpenVINO™ helper function and directly
via port methods. We are going to review both of them.

Let's start with OpenVINO™ helper functions. The most popular function is ``ov::replace_node(old_node, new_node)``.

Let's review a replacement case where a Negative operation is replaced with Multiply.

.. image:: ../../assets/images/ov_replace_node.png

.. doxygensnippet:: docs/articles_en/assets/snippets/ov_model_snippets.cpp
   :language: cpp
   :fragment: [ov:replace_node]

``ov::replace_node`` has a constraint that number of output ports for both nodes must be the same.
Otherwise, the attempt to replace the nodes will result in an exception.

The alternative way to do the same replacement is the following:

.. doxygensnippet:: docs/articles_en/assets/snippets/ov_model_snippets.cpp
   :language: cpp
   :fragment: [ov:manual_replace]

Another transformation example is insertion. Let's insert an additional Relu node.

.. image:: ../../assets/images/ov_insert_node.png

.. doxygensnippet:: docs/articles_en/assets/snippets/ov_model_snippets.cpp
   :language: cpp
   :fragment: [ov:insert_node]

The alternative way of inserting a node is to make a copy of the node and use ``ov::replace_node()``:

.. doxygensnippet:: docs/articles_en/assets/snippets/ov_model_snippets.cpp
   :language: cpp
   :fragment: [ov:insert_node_with_copy]

Node elimination
++++++++++++++++

Another type of node replacement is elimination of a node.

To eliminate a node, OpenVINO provides a method that considers all limitations of the OpenVINO Runtime.

.. doxygensnippet:: docs/articles_en/assets/snippets/ov_model_snippets.cpp
   :language: cpp
   :fragment: [ov:eliminate_node]

If the replacement is successful, ``ov::replace_output_update_name()`` automatically preserves the friendly name and runtime info.

.. _transformations_types:

Transformations types
#####################

OpenVINO™ Runtime has three main transformation types:

* :doc:`Model pass <transformation-api/model-pass>` - straightforward way to work with ``ov::Model`` directly
* :doc:`Matcher pass <transformation-api/matcher-pass>` - pattern-based transformation approach
* :doc:`Graph rewrite pass <transformation-api/graph-rewrite-pass>` - container for matcher passes used for efficient execution

.. image:: ../../assets/images/transformations_structure.png

Transformation conditional compilation
######################################

Transformation library has two internal macros to support conditional compilation feature.

* ``MATCHER_SCOPE(region)`` - allows to disable the MatcherPass if matcher isn't used. The region
  name should be unique. This macro creates a local variable ``matcher_name`` which you should use as a matcher name.
* ``RUN_ON_MODEL_SCOPE(region)`` - allows to disable run_on_model pass if it isn't used. The region name should be unique.

.. _transformation_writing_essentials:

Transformation writing essentials
#################################

To develop a transformation, follow these transformation rules:

1. Friendly Names
+++++++++++++++++

Each ``ov::Node`` has a unique name and a friendly name. In transformations, only the friendly
name matters because it represents the name from the model's perspective.
To prevent losing the friendly name when replacing a node with another node or a subgraph,
the original friendly name is set to the last node in the replacing subgraph. See the example below.

.. doxygensnippet:: docs/articles_en/assets/snippets/ov_model_snippets.cpp
   :language: cpp
   :fragment: [ov:replace_friendly_name]

In more complicated cases, when a replaced operation has several outputs and additional
consumers are added to its outputs, the decision on how to set the friendly name is determined by an agreement.

2. Runtime Info
+++++++++++++++

Runtime info is a map ``std::map<std::string, ov::Any>`` located inside the ``ov::Node`` class.
It represents additional attributes of the ``ov::Node``.
These attributes, which can be set by users or plugins, need to be preserved when executing
a transformation that changes ``ov::Model``, as they are not automatically propagated.
In most cases, transformations have the following types: 1:1 (replace node with another node),
1:N (replace node with a sub-graph), N:1 (fuse sub-graph into a single node), N:M (any other transformation).
Currently, there is no mechanism that automatically detects transformation types, so this
runtime information needs to be propagated manually. See the example below:


.. doxygensnippet:: docs/articles_en/assets/snippets/ov_model_snippets.cpp
   :language: cpp
   :fragment: [ov:copy_runtime_info]

When a transformation has multiple fusions or decompositions, ``ov::copy_runtime_info`` must be called multiple times for each case.

.. note::

   ``copy_runtime_info`` removes ``rt_info`` from destination nodes. If you want to keep it,
   specify them in source nodes as following: ``copy_runtime_info({a, b, c}, {a, b})``

3. Constant Folding
+++++++++++++++++++

If your transformation inserts constant sub-graphs that need to be folded, do not forget
to use ``ov::pass::ConstantFolding()`` after your transformation or call constant folding directly for operation.
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

* Do not use deprecated OpenVINO™ API. Deprecated methods are marked with ``OPENVINO_DEPRECATED`` macro in their definition.
* Do not pass ``shared_ptr<Node>`` as input for another node if the type of the node is unknown
  or if it has multiple outputs. Instead, use explicit output ports.
* If you replace a node with another node that produces different shape, note that
  the new shape will not be propagated until the first ``validate_nodes_and_infer_types``
  call for ``ov::Model``. If you are using ``ov::pass::Manager``, it will automatically call
  this method after each transformation execution.
* Do not forget to call the ``ov::pass::ConstantFolding`` pass if your transformation creates constant subgraphs.
* Use latest OpSet if you are not developing downgrade transformation pass.
* When developing a callback for ``ov::pass::MatcherPass``, do not change nodes that come after the root node in the topological order.

.. _using_pass_manager:

Using pass manager
##################

``ov::pass::Manager`` is a container class that can store a list of transformations and execute them.
The main idea of this class is to have a high-level representation for grouped list of transformations.
It can register and apply any `transformation pass <#transformations-types>`__ on a model.
In addition, ``ov::pass::Manager`` has extended debug capabilities (find more information
in the `how to debug transformations <#how-to-debug-transformations>`__ section).

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

If you are using ``ov::pass::Manager`` to run sequence of transformations, you can get
additional debug capabilities by using the following environment variables:

.. code-block:: cpp

   OV_PROFILE_PASS_ENABLE=1 - enables performance measurement for each transformation and prints execution status
   OV_ENABLE_VISUALIZE_TRACING=1 -  enables visualization after each transformation. By default, it saves dot and svg files.


.. note::

   Make sure that you have dot installed on your machine; otherwise, it will silently save only dot file without svg file.

See Also
########

* :doc:`OpenVINO™ Model Representation <../../openvino-workflow/running-inference/model-representation>`
* :doc:`OpenVINO™ Extensions <../openvino-extensibility>`

