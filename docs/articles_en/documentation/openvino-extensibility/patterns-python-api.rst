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

What is a pattern and how to create one?
++++++++++++++++++++++++++++++++++++++++

Let's start with a very brief definition of pattern. In a nutshell, a pattern is a very simplified model comprised of nodes we want to match. It lacks some features of a model and generally cannot serve as one,
but this is not very important right now.

Suppose, we are having a very simple pattern consisting of 2 nodes and we want to find it in a given model.

.. image:: ./../../_static/images/python-api.png

Let's create the model and the pattern

.. doxygensnippet:: docs/snippets/ov_patterns.py
   :language: cpp
   :fragment: [ov:create_simple_model]
