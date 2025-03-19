OpenVINO Matcher Pass
=====================


.. meta::
   :description: Learn how to create a pattern, implement a callback, register
                 the pattern and Matcher to execute MatcherPass transformation
                 on a model.

``ov::pass::MatcherPass``  is used for pattern-based transformations.

Template for MatcherPass transformation class

.. doxygensnippet:: docs/articles_en/assets/snippets/template_pattern_transformation.hpp
   :language: cpp
   :fragment: [graph_rewrite:template_transformation_hpp]

.. tab-set::

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/articles_en/assets/snippets/template_pattern_transformation.cpp
         :language: cpp
         :fragment: [graph_rewrite:template_transformation_cpp]

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_matcher_pass.py
         :language: py
         :fragment: [matcher_pass:ov_matcher_pass_py]

To use ``ov::pass::MatcherPass``, you need to complete these steps:

1. Create a pattern
2. Implement a callback
3. Register the pattern and Matcher
4. Execute MatcherPass

So let's go through each of these steps.

Create a pattern
################

Pattern is a single root ``ov::Model``. But the only difference is that you do not need to create a model object, you just need to create and connect opset or special pattern operations.
Then you need to take the last created operation and put it as a root of the pattern. This root node will be used as a root node in pattern matching.

.. note::
   Any nodes in a pattern that have no consumers and are not registered as root will not be used in pattern matching.

.. doxygensnippet:: docs/articles_en/assets/snippets/ov_model_snippets.cpp
   :language: cpp
   :fragment: [pattern:simple_example]

The ``Parameter`` operation in the example above has type and shape specified. These attributes are needed only to create Parameter operation class and will not be used in pattern matching.

For more pattern examples, refer to the `pattern matching section <#pattern-matching>`__.

Implement callback
##################

Callback is an action applied to every pattern entrance. In general, callback is the lambda function that takes Matcher object with detected subgraph.

.. doxygensnippet:: docs/articles_en/assets/snippets/ov_model_snippets.cpp
   :language: cpp
   :fragment: [pattern:callback_example]

The example above shows the callback structure and how Matcher can be used for accessing nodes detected by pattern.
Callback return value is ``true`` if root node was replaced and another pattern cannot be applied to the same root node; otherwise, it is ``false``.

.. note::

   It is not recommended to manipulate with nodes that are under root node. This may affect GraphRewrite execution as it is expected that all nodes that come after root node in topological order are valid and can be used in pattern matching.

MatcherPass also provides functionality that allows reporting of the newly created nodes that can be used in additional pattern matching.
If MatcherPass was registered in ``ov::pass::Manager`` or ``ov::pass::GraphRewrite``, these registered nodes will be added for additional pattern matching.
That means that matcher passes registered in ``ov::pass::GraphRewrite`` will be applied to these nodes.

The example below shows how single MatcherPass can fuse sequence of operations using the ``register_new_node`` method.

.. doxygensnippet:: docs/articles_en/assets/snippets/template_pattern_transformation.cpp
   :language: cpp
   :fragment: [matcher_pass:relu_fusion]

.. note::
   If you register multiple nodes, please add them in topological order. We do not topologically sort these nodes as it is a time-consuming operation.

Register pattern and Matcher
############################

The last step is to register Matcher and callback inside the MatcherPass pass. To do this, call the ``register_matcher`` method.

.. note::

   Only one matcher can be registered for a single MatcherPass class.

.. code-block:: cpp

   // Register matcher and callback
   register_matcher(m, callback);


Execute MatcherPass
###################

MatcherPass has multiple ways to be executed:

* Run on a single node - it can be useful if you want to run MatcherPass inside another transformation.

.. doxygensnippet:: docs/articles_en/assets/snippets/template_pattern_transformation.cpp
   :language: cpp
   :fragment: [matcher_pass:run_on_node]

* Run on ``ov::Model`` using GraphRewrite - this approach gives ability to run MatcherPass on whole ``ov::Model``. Moreover, multiple MatcherPass transformation can be registered in a single GraphRewite to be executed in a single graph traversal.

.. doxygensnippet:: docs/articles_en/assets/snippets/template_pattern_transformation.cpp
   :language: cpp
   :fragment: [matcher_pass:graph_rewrite]

* Run on ``ov::Model`` using ``ov::pass::Manager`` - this approach helps you to register MatcherPass for execution on ``ov::Model`` as another transformation types.

.. doxygensnippet:: docs/articles_en/assets/snippets/template_pattern_transformation.cpp
   :language: cpp
   :fragment: [matcher_pass:manager]


Pattern Matching
################

Sometimes patterns cannot be expressed via regular operations or it is too complicated.
For example, if you want to detect **Convolution->Add** sub-graph without specifying particular input type for Convolution operation or you want to create a pattern where some of operations can have different types.
And for these cases OpenVINO™ provides additional helpers to construct patterns for GraphRewrite transformations.

There are two main helpers:

1. ``ov::pass::pattern::any_input`` - helps to express inputs if their types are undefined.
2. ``ov::pass::pattern::wrap_type <T>`` - helps to express nodes of pattern without specifying node attributes.

Let's go through the example to have better understanding of how it works:

.. note::
   Node attributes do not participate in pattern matching and are needed only for operations creation. Only operation types participate in pattern matching.

The example below shows basic usage of ``ov::passpattern::any_input``.
Here we construct Multiply pattern with arbitrary first input and Constant as a second input.
Also as Multiply is commutative operation, it does not matter in which order we set inputs (any_input/Constant or Constant/any_input) because both cases will be matched.

.. doxygensnippet:: docs/articles_en/assets/snippets/ov_model_snippets.cpp
   :language: cpp
   :fragment: [pattern:label_example]

This example shows how we can construct a pattern when operation has arbitrary number of inputs.

.. doxygensnippet:: docs/articles_en/assets/snippets/ov_model_snippets.cpp
   :language: cpp
   :fragment: [pattern:concat_example]

This example shows how to use predicate to construct a pattern. Also it shows how to match pattern manually on given node.

.. doxygensnippet:: docs/articles_en/assets/snippets/ov_model_snippets.cpp
   :language: cpp
   :fragment: [pattern:predicate_example]

.. note::

   Be careful with manual matching because Matcher object holds matched nodes. To clear a match, use the m->clear_state() method.

See Also
########

* :doc:`OpenVINO™ Transformations <../transformation-api>`

