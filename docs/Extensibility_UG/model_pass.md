# OpenVINO Model Pass {#openvino_docs_Extensibility_UG_model_pass}

`ov::pass::ModelPass` is used for transformations that take entire `ov::Model` as an input and process it.

Template for ModelPass transformation class

@snippet template_model_transformation.hpp model_pass:template_transformation_hpp

@snippet template_model_transformation.cpp model_pass:template_transformation_cpp

Using `ov::pass::ModelPass`, you need to override the `run_on_model` method where you will write the transformation code.
Return value is `true` if the original model has changed during transformation (new operation was added, or operations replacement was made, or node attributes were changed); otherwise, it is `false`.
Also `ov::pass::ModelPass` based transformations can be executed via `ov::pass::Manager`.

## See Also

* [OpenVINO™ Transformations](./ov_transformations.md)
