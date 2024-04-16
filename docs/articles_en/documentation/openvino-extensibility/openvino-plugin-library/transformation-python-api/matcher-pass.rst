.. {#openvino_docs_Extensibility_UG_matcher_pass}

OpenVINO Model Pass Python API
==============================


.. meta::
   :description: Learn how to create a pattern, implement a callback, register
                 the pattern and Matcher to execute MatcherPass transformation
                 on a model.

``MatcherPass`` is used for pattern-based transformations.
To create transformation you need:

1. Create a pattern
2. Implement a callback
3. Register the pattern and Matcher

In the next example we define transformation that searches for ``Relu`` layer and inserts after it another
``Relu`` layer.

.. doxygensnippet:: docs/snippets/ov_matcher_pass.py
   :language: py
   :fragment: [matcher_pass:ov_matcher_pass_py]

The next example shows MatcherPass-based transformation usage.

.. doxygensnippet:: docs/snippets/ov_matcher_pass.py
   :language: py
   :fragment: [matcher_pass_full_example:ov_matcher_pass_py]

After running this code you will see the next: text
```
model ops :
parameter
result
relu

model ops :
parameter
result
relu
new_relu
```

In oder to run this script you need to export PYTHONPATH as the path to binary OpenVINO python models.
