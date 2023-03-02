# Use Case - Integrate and Save Preprocessing Steps Into IR {#openvino_docs_OV_UG_Preprocess_Usecase_save}


Previous sections covered the topic of the [preprocessing steps](@ref openvino_docs_OV_UG_Preprocessing_Details) and the overview of [Layout](@ref openvino_docs_OV_UG_Layout_Overview) API.

For many applications, it is also important to minimize read/load time of a model. Therefore, performing integration of preprocessing steps every time on application startup, after `ov::runtime::Core::read_model`, may seem inconvenient. In such cases, once pre and postprocessing steps have been added, it can be useful to store new execution model to OpenVINO Intermediate Representation (OpenVINO IR, `.xml` format).

Most available preprocessing steps can also be performed via command-line options, using Model Optimizer. For details on such command-line options, refer to the [Optimizing Preprocessing Computation](../MO_DG/prepare_model/Additional_Optimizations.md).

## Code example - Saving Model with Preprocessing to OpenVINO IR

When some preprocessing steps cannot be integrated into the execution graph using Model Optimizer command-line options (for example, `YUV`->`RGB` color space conversion, `Resize`, etc.), it is possible to write a simple code which:
 - Reads the original model (OpenVINO IR, TensorFlow, ONNX, PaddlePaddle).
 - Adds the preprocessing/postprocessing steps.
 - Saves resulting model as IR (`.xml` and `.bin`).

Consider the example, where an original ONNX model takes one `float32` input with the `{1, 3, 224, 224}` shape, the `RGB` channel order, and mean/scale values applied. In contrast, the application provides `BGR` image buffer with a non-fixed size and input images as batches of two. Below is the model conversion code that can be applied in the model preparation script for such a case.

- Includes / Imports

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_preprocessing.cpp ov:preprocess:save_headers

@endsphinxtab

@sphinxtab{Python}

@snippet docs/snippets/ov_preprocessing.py ov:preprocess:save_headers

@endsphinxtab

@endsphinxtabset

- Preprocessing & Saving to the OpenVINO IR code.

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_preprocessing.cpp ov:preprocess:save

@endsphinxtab

@sphinxtab{Python}

@snippet docs/snippets/ov_preprocessing.py ov:preprocess:save

@endsphinxtab

@endsphinxtabset


## Application Code - Load Model to Target Device

After this, the application code can load a saved file and stop preprocessing. In this case, enable [model caching](./Model_caching_overview.md) to minimize load time when the cached model is available.

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_preprocessing.cpp ov:preprocess:save_load

@endsphinxtab

@sphinxtab{Python}

@snippet docs/snippets/ov_preprocessing.py ov:preprocess:save_load

@endsphinxtab

@endsphinxtabset


## Additional Resources
* [Preprocessing Details](@ref openvino_docs_OV_UG_Preprocessing_Details)
* [Layout API overview](@ref openvino_docs_OV_UG_Layout_Overview)
* [Model Optimizer - Optimize Preprocessing Computation](../MO_DG/prepare_model/Additional_Optimizations.md)
* [Model Caching Overview](./Model_caching_overview.md)
* The `ov::preprocess::PrePostProcessor` C++ class documentation
* The `ov::pass::Serialize` - pass to serialize model to XML/BIN
* The `ov::set_batch` - update batch dimension for a given model