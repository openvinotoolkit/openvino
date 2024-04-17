.. {#openvino_docs_transformations}

Overview of creating pattern using Python API
=============================================


.. meta::
   :description: Learn how to apply additional model optimizations or transform
                 unsupported subgraphs and operations, using OpenVINO™ Transformations API.


.. toctree::
   :maxdepth: 1
   :hidden:

   i don't know what to put here

Pattern matching is an essential component of OpenVINO™ transformations. Before performing any transformation on a subgraph of a graph, we need to find the subgraph in the graph.
Here come patterns which serve as a searching utility to identify nodes we are going to work with in our transformation. In this article we are going to review the basics of pattern
creation using Python API and helping utilities we may use to facilitate working with them. Some examples will be intentionally simplified for ease of understanding. 

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

Suppose, we are having a very simple pattern consisting of 2 nodes and we want to find it in a given model.

.. image:: ./../../_static/images/python-api.png

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

2. Any 
++++++++++++++++++++++++++++++++++++++++

3. Or
++++++++++++++++++++++++++++++++++++++++

4. Optional
++++++++++++++++++++++++++++++++++++++++