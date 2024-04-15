.. {#openvino_docs_Extensibility_UG_model_pass}

OpenVINO Model Pass Python API
==============================


.. meta::
   :description: Learn how to use Model Pass transformation class to take entire
                 ov::Model as input and process it.

``ModelPass`` can be used as base class for transformation classes that take entire ``Model`` and proceed it.
To create transformation you need:

1. Define class with ``ModelPass`` as a parent
2. Redefine run_on_model method that will receive ``Model`` as an argument

.. doxygensnippet:: docs/snippets/ov_model_pass.py
   :language: py
   :fragment: [model_pass:ov_model_pass_py]

In this example we define transformation that prints all model operation names.

The next example shows ModelPass-based transformation usage.

.. doxygensnippet:: docs/snippets/ov_model_pass.py
   :language: py
   :fragment: [model_pass_full_example:ov_model_pass_py]

We create Model with Relu, Parameter and Result nodes. After running this code you will see names of these three nodes.
In oder to run this script you need to export PYTHONPATH as the path to binary OpenVINO python models.
