
Transformation Patterns with OpenVINO API
==================================================

.. meta::
   :description: Learn how to apply additional model optimizations or transform
                 unsupported subgraphs and operations using OpenVINO™ Transformations API.

Pattern matching is an essential component of OpenVINO™ transformations. Before performing any transformation on a subgraph of a graph, it is necessary to find that subgraph in the graph.
Patterns serve as a searching utility to identify nodes intended for transformations. This article covers the basics of pattern
creation using OpenVINO™ API and helpful utilities to facilitate working with them. While this guide focuses on creating patterns, if you want to learn more about ``MatcherPass``, refer to the :doc:`OpenVINO Matcher Pass article <./matcher-pass>`. Note that some examples may be intentionally simplified for ease of understanding.

Before proceeding further, it is necessary to add some imports. These imports include the operations to be used and additional utilities described in this guide.
Add the following lines to your file:

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_patterns.py
         :language: python
         :fragment: [ov:imports]

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_patterns.cpp
         :language: cpp
         :fragment: [ov:imports]

Pattern Creation
+++++++++++++++++++++

A pattern is a simplified model comprised of nodes aimed to be matched. It lacks some features of a model and cannot function as one.

Consider a straightforward pattern consisting of three nodes to be found in a given model.

.. image:: ./../../../assets/images/simple_pattern_example.png

Let's create the model and the pattern:

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_patterns.py
         :language: python
         :fragment: [ov:create_simple_model_and_pattern]

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_patterns.cpp
         :language: cpp
         :fragment: [ov:create_simple_model_and_pattern]

.. note:: This example uses testing utilities that directly compare given sequences of nodes. In reality, the process of finding a pattern within a model is more complicated. However, to focus only on patterns and their functionality, these details are intentionally omitted.

Although the code is functional, in OpenVINO, patterns are typically not created using the same nodes as those used for creating the model. Instead, wrappers are preferred, providing additional functionality.
For the given case, ``WrapType`` is used and the code looks as following:

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_patterns.py
         :language: python
         :fragment: [ov:create_simple_model_and_pattern_wrap_type]

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_patterns.cpp
         :language: cpp
         :fragment: [ov:create_simple_model_and_pattern_wrap_type]

1. WrapType
++++++++++++++++++++++++++++++++++++++++

``WrapType`` is a wrapper used to store one or many types to match them. As demonstrated earlier, it is possible to specify a single type in ``WrapType`` and use it for matching.
However, you can also list all possible types for a given node, for example:

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_patterns.py
         :language: python
         :fragment: [ov:wrap_type_list]

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_patterns.cpp
         :language: cpp
         :fragment: [ov:wrap_type_list]

Note that ``pattern_sig`` is created with the list ``["opset13.Relu", "opset13.Sigmoid"]``, meaning it can be either a ``Relu`` or a ``Sigmoid``.
This feature enables matching the same pattern against different nodes. Essentially, ``WrapType`` can represent "one of listed" types. ``WrapType`` supports specifying more than two types.

To add additional checks for your node, create a predicate by providing a function or a lambda. This function will be executed during matching, performing the additional validation specified in the logic of the function. For example, you might want to check the consumers count of a given node:

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_patterns.py
         :language: python
         :fragment: [ov:wrap_type_predicate]

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_patterns.cpp
         :language: cpp
         :fragment: [ov:wrap_type_predicate]

2. AnyInput
++++++++++++++++++++++++++++++++++++++++
``AnyInput`` is used when there is no need to specify a particular input for a given node.

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_patterns.py
         :language: python
         :fragment: [ov:any_input]

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_patterns.cpp
         :language: cpp
         :fragment: [ov:any_input]

You can also create ``AnyInput()`` with a predicate, if you want additional checks for you input. It will look similar to ``WrapType`` with a lambda or a function. For example, to ensure that the input has a rank of 4:

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_patterns.py
         :language: python
         :fragment: [ov:any_input_predicate]

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_patterns.cpp
         :language: cpp
         :fragment: [ov:any_input_predicate]

3. Or
++++++++++++++++++++++++++++++++++++++++
``Or`` functions similar to ``WrapType``, however, while ``WrapType`` can only match one of the types provided in the list, ``Or`` is used to match different branches of nodes.
Suppose the goal is to match the model against two different sequences of nodes. The ``Or`` type
facilitates this by creating two different branches (``Or`` supports more than two branches), looking as follows:

.. image:: ./../../../assets/images/or_branches.png

The red branch will not match, but it will work perfectly for the blue one.
Here is how it looks in code:

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_patterns.py
         :language: python
         :fragment: [ov:pattern_or]

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_patterns.cpp
         :language: cpp
         :fragment: [ov:pattern_or]

Note that matching will succeed for the first matching branch and the remaining ones will not be checked.

4. Optional
++++++++++++++++++++++++++++++++++++++++
``Optional`` is a bit tricky. It allows specifying whether a node might be present or absent in the model. Under the hood,
the pattern will create two branches using ``Or``: one with the optional node present and another one without it. Here is what it would look like with the ``Optional``
unfolding into two branches:

.. image:: ./../../../assets/images/optional.png

The code for our model looks as follows:

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_patterns.py
         :language: python
         :fragment: [ov:pattern_optional_middle]

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_patterns.cpp
         :language: cpp
         :fragment: [ov:pattern_optional_middle]

The ``Optional`` does not necessarily have to be in the middle of the pattern. It can be a top node and a root node.


Top node:

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_patterns.py
         :language: python
         :fragment: [ov:pattern_optional_top]

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_patterns.cpp
         :language: cpp
         :fragment: [ov:pattern_optional_top]

Root node:

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_patterns.py
         :language: python
         :fragment: [ov:pattern_optional_root]

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_patterns.cpp
         :language: cpp
         :fragment: [ov:pattern_optional_root]

``Optional`` also supports adding a predicate the same way ``WrapType`` and ``AnyInput`` do:

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_patterns.py
         :language: python
         :fragment: [ov:optional_predicate]

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_patterns.cpp
         :language: cpp
         :fragment: [ov:optional_predicate]