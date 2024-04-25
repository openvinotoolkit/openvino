.. {#openvino_docs_Extensibility_UG_patterns-python-api}

Overview of creating Transformation patterns using OpenVINO™ API
=============================================

.. meta::
   :description: Learn how to apply additional model optimizations or transform
                 unsupported subgraphs and operations, using OpenVINO™ Transformations API.

Pattern matching is an essential component of OpenVINO™ transformations. Before performing any transformation on a subgraph of a graph, we need to find the subgraph in the graph.
Here come patterns which serve as a searching utility to identify nodes we are going to work with in our transformation. In this article we are going to review the basics of pattern
creation using OpenVINO™ API and helping utilities we may use to facilitate working with them. Some examples will be intentionally simplified for ease of understanding. 

Though, before proceeding any further, we need to add some imports. That would import the operations we're going to use and additional utility that we are going to talk later in this guide.
Add the following lines to your file:

.. doxygensnippet:: docs/snippets/ov_patterns.py
   :language: cpp
   :fragment: [ov:imports]

This should be enough for our article.

What is a pattern and how to create one?
++++++++++++++++++++++++++++++++++++++++

Let's start with a very brief definition of pattern. In a nutshell, a pattern is a very simplified model comprised of nodes we want to match. It lacks some features of a model and generally cannot serve as one,
but this is not very important right now.

Suppose, we are having a very simple pattern consisting of 3 nodes and we want to find it in a given model.

.. image:: ./../../../_static/images/python-api.png

Let's create the model and the pattern:

.. doxygensnippet:: docs/snippets/ov_patterns.py
   :language: cpp
   :fragment: [ov:create_simple_model_and_pattern]

.. note:: We are using testing utilities in this article that just directly compare given sequences of nodes. In real-life everything is a little bit more complicated with many other things happening to find a pattern in a model, but we intentionally omit this details to focus on patterns and their functionality.

Our code already looks promissing, however in OpenVINO™ we usually don't create patterns using the same nodes we used for creating the model. Instead, we want to use so-called wrappers that provide us for additional functionality.
For the given case we would probably use ``WrapType`` and the code would look as following:

.. doxygensnippet:: docs/snippets/ov_patterns.py
   :language: cpp
   :fragment: [ov:create_simple_model_and_pattern_wrap_type]

1. WrapType
++++++++++++++++++++++++++++++++++++++++

``WrapType`` is a wrapper used to store one or many types to match them. As we already saw, it is possible to specify a single type in ``WrapType`` and use it for matching.
However, it is also possible to list all possible types for the given node. For example, you may do something like this:

.. doxygensnippet:: docs/snippets/ov_patterns.py
   :language: cpp
   :fragment: [ov:wrap_type_list]

As you may see, ``pattern_sig`` is created with the list ``["opset13.Relu", "opset13.Sigmoid"]`` which means it can either be a ``Relu`` or ``Sigmoid``. Pretty convenient, huh?
This is why matching the same pattern against different nodes becomes possible. Basically, we may think of ``WrapType`` as "one of listed". Note, that you may provide more than 2 types
to ``WrapType``.

If you want to have some additional checking for you node, you may create a predicate for it providing a function or a lambda. This function will be executed during
matching performing the additional validation specified in the logic of the function. For example, you may want to check the consumers count of a given node:

.. doxygensnippet:: docs/snippets/ov_patterns.py
   :language: cpp
   :fragment: [ov:wrap_type_predicate]

2. AnyInput 
++++++++++++++++++++++++++++++++++++++++
You have already seen ``AnyInput`` in the above examples. We use it when we don't really care about a specific input for a given node.

.. doxygensnippet:: docs/snippets/ov_patterns.py
   :language: cpp
   :fragment: [ov:any_input]

You may also create ``AnyInput()`` with a predicate, if you want some additional checks for you input. It would look similar to ``WrapType`` with a lambda or a function. Let's say we want to make sure the inputs has a rank of 4.

.. doxygensnippet:: docs/snippets/ov_patterns.py
   :language: cpp
   :fragment: [ov:any_input_predicate]

3. Or
++++++++++++++++++++++++++++++++++++++++
``Or`` is somewhat similar to ``WrapType``, however if ``WrapType`` can only match one of types provided in the list, ``Or`` is used to match different _branches_ of nodes.
It would be much easier to understand with a visualization. Let's say, we want to try to match the model against two different sequences of nodes. The ``Or`` type
facilitates this by creating 2 different branches (``Or`` supports more than 2 branches). It would look as following:

.. image:: ./../../../_static/images/or-branches.png

As you may see, the red branch will not match, however it will work perfectly fine for the blue one.
That's what it would look in code:

.. doxygensnippet:: docs/snippets/ov_patterns.py
   :language: cpp
   :fragment: [ov:or]

Note that matching will succeed for the first matching branch and the remaining ones will not be checked.

4. Optional
++++++++++++++++++++++++++++++++++++++++
``Optional`` is a bit tricky one. It allows to specify what node might be or might not present in the model. Under the hood
the pattern will create 2 branches using ``Or``: one with the optional node present, another one without it. That's what it would look like visually with the ``Optional``
unfolding into 2 branches:

.. image:: ./../../../_static/images/optional.png

The code would look as following for our model:

.. doxygensnippet:: docs/snippets/ov_patterns.py
   :language: cpp
   :fragment: [ov:optional_middle]

The ``Optional`` doesn't necessarily have to be in the middle of the pattern. It can be a top node and a root node.

Top node:

.. doxygensnippet:: docs/snippets/ov_patterns.py
   :language: cpp
   :fragment: [ov:optional_top]

Root node:

.. doxygensnippet:: docs/snippets/ov_patterns.py
   :language: cpp
   :fragment: [ov:optional_root]

```Optional``` also supports adding a predicate the same way ``WrapType`` and ``AnyInput`` do:

.. doxygensnippet:: docs/snippets/ov_patterns.py
   :language: cpp
   :fragment: [ov:optional_predicate]