# Converting a PaddlePaddle* Model {#openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_Paddle}

## Convert a PaddlePaddle Model <a name="Convert_From_Paddle"></a>
To convert a PaddlePaddle model, use the `mo` script and specify the path to the input model `.pdmodel` file:

```sh
 mo --input_model <INPUT_MODEL>.pdmodel
```

### Example of Converting a PaddlePaddle Model
Below is the example command to convert yolo v3 PaddlePaddle network to OpenVINO IR network with Model Optimizer.

```sh
 mo --input_model=yolov3.pdmodel --input=image,im_shape,scale_factor --input_shape=[1,3,608,608],[1,2],[1,2] --reverse_input_channels --output=save_infer_model/scale_0.tmp_1,save_infer_model/scale_1.tmp_1
```

## Supported PaddlePaddle Layers
Refer to [Supported Framework Layers](../Supported_Frameworks_Layers.md) for the list of supported standard layers.

## Frequently Asked Questions (FAQ)

When Model Optimizer is unable to run to completion due to issues like typographical errors, incorrectly used options, etc., it provides explanatory messages. They describe the potential cause of the problem and give a link to the [Model Optimizer FAQ](../Model_Optimizer_FAQ.md), which provides instructions on how to resolve most issues. The FAQ also includes links to relevant sections in the Model Optimizer Developer Guide to help you understand what went wrong.

## See Also
[Model Conversion Tutorials](Convert_Model_Tutorials.md)
