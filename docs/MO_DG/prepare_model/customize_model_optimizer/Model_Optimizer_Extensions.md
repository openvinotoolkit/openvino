# Model Optimizer Extensions {#openvino_docs_MO_DG_prepare_model_customize_model_optimizer_Model_Optimizer_Extensions}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   openvino_docs_MO_DG_prepare_model_customize_model_optimizer_Model_Optimizer_Extensions_Model_Optimizer_Operation
   openvino_docs_MO_DG_prepare_model_customize_model_optimizer_Model_Optimizer_Extensions_Model_Optimizer_Extractor
   openvino_docs_MO_DG_prepare_model_customize_model_optimizer_Model_Optimizer_Extensions_Model_Optimizer_Transformation_Extensions

@endsphinxdirective

Model Optimizer extensions enable you to inject some logic to the model conversion pipeline without changing the Model
Optimizer core code. There are three types of the Model Optimizer extensions:

1. [Model Optimizer operation](Model_Optimizer_Operation.md).
2. A [framework operation extractor](Model_Optimizer_Extractor.md).
3. A [model transformation](Model_Optimizer_Transformation_Extensions.md), which can be executed during front, middle or back phase of the model conversion.

An extension is just a plain text file with a Python code. The file should contain a class (or classes) inherited from
one of extension base classes. Extension files should be saved to a directory with the following structure:

```sh
./<MY_EXT>/
           ops/                  - custom operations
           front/                - framework independent front transformations
                 <FRAMEWORK_1>/  - front transformations for <FRAMEWORK_1> models only and extractors for <FRAMEWORK_1> operations
                 <FRAMEWORK_2>/  - front transformations for <FRAMEWORK_2> models only and extractors for <FRAMEWORK_2> operations
                 ...
           middle/               - middle transformations
           back/                 - back transformations
```

Model Optimizer uses the same layout internally to keep built-in extensions. The only exception is that the 
`mo/ops/` directory is also used as a source of the Model Optimizer operations due to historical reasons.

> **NOTE**: The name of a root directory with extensions should not be equal to "extensions" because it will result in a
> name conflict with the built-in Model Optimizer extensions.

> **NOTE**: Model Optimizer itself is built by using these extensions, so there is a huge number of examples of their
> usage in the Model Optimizer code.

## Additional Resources

* [Model Optimizer Extensibility](Customize_Model_Optimizer.md)
* [Graph Traversal and Modification Using Ports and Connections](Model_Optimizer_Ports_Connections.md)
* [Extending Model Optimizer with Caffe Python Layers](Extending_Model_Optimizer_with_Caffe_Python_Layers.md)
