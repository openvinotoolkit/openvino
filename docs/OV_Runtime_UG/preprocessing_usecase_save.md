# Use Case - Integrate and Save Preprocessing Steps Into IR {#openvino_docs_OV_UG_Preprocess_Usecase_save}

## Introduction

In previous sections we've covered how to add [preprocessing steps](./preprocessing_details.md) and got the overview of [Layout](./layout_overview.md) API.

For many applications it is also important to minimize model's read/load time, so performing integration of preprocessing steps every time on application startup after `ov::runtime::Core::read_model` may look not convenient. In such cases, after adding of Pre- and Post-processing steps it can be useful to store new execution model to Intermediate Representation (IR, .xml format).

Most part of existing preprocessing steps can also be performed via command line options using Model Optimizer tool. Refer to [Model Optimizer - Optimize Preprocessing Computation](../MO_DG/prepare_model/Additional_Optimizations.md) for details os such command line options.

## Code example - saving model with preprocessing to IR

In case if you have some preprocessing steps which can't be integrated into execution graph using Model Optimizer command line options (e.g. `YUV->RGB` color space conversion, Resize, etc.) it is possible to write simple code which:
 - Reads original model (IR, ONNX, Paddle)
 - Adds preprocessing/postprocessing steps
 - Saves resulting model as IR (.xml/.bin)

Let's consider the example, there is an original `ONNX` model which takes one `float32` input with shape `{1, 3, 224, 224}` with `RGB` channels order, with mean/scale values applied. User's application can provide `BGR` image buffer with not fixed size. Additionally, we'll also imagine that our application provides input images as batches, each batch contains 2 images. Here is how model conversion code may look like in your model preparation script

- Includes / Imports

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_preprocessing.cpp ov:preprocess:save_headers

@endsphinxtab

@sphinxtab{Python}

@snippet docs/snippets/ov_preprocessing.py ov:preprocess:save_headers

@endsphinxtab

@endsphinxtabset

- Preprocessing & Saving to IR code

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_preprocessing.cpp ov:preprocess:save

@endsphinxtab

@sphinxtab{Python}

@snippet docs/snippets/ov_preprocessing.py ov:preprocess:save

@endsphinxtab

@endsphinxtabset


## Application code - load model to target device

After this, your application's code can load saved file and don't perform preprocessing anymore. In this example we'll also enable [model caching](./Model_caching_overview.md) to minimize load time when cached model is available

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_preprocessing.cpp ov:preprocess:save_load

@endsphinxtab

@sphinxtab{Python}

@snippet docs/snippets/ov_preprocessing.py ov:preprocess:save_load

@endsphinxtab

@endsphinxtabset


## See Also
* [Preprocessing Details](./preprocessing_details.md)
* [Layout API overview](./layout_overview.md)
* [Model Optimizer - Optimize Preprocessing Computation](../MO_DG/prepare_model/Additional_Optimizations.md)
* [Model Caching Overview](./Model_caching_overview.md)
* <code>ov::preprocess::PrePostProcessor</code> C++ class documentation
* <code>ov::pass::Serialize</code> - pass to serialize model to XML/BIN
* <code>ov::set_batch</code> - update batch dimension for a given model