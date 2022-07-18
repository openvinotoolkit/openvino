# Converting TensorFlow FaceNet Models {#openvino_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_FaceNet_From_Tensorflow}

[Public pretrained FaceNet models](https://github.com/davidsandberg/facenet#pre-trained-models) contain both training
and inference part of graph. Switch between this two states is manageable with placeholder value.
Intermediate Representation (IR) models are intended for inference, which means that train part is redundant.

There are two inputs in this network: boolean `phase_train` which manages state of the graph (train/infer) and
`batch_size` which is a part of batch joining pattern.


![FaceNet model view](../../../img/FaceNet.png)

## Converting a TensorFlow FaceNet Model to the IR

To generate a FaceNet OpenVINO model, feed a TensorFlow FaceNet model to Model Optimizer with the following parameters:
```sh
 mo
--input_model path_to_model/model_name.pb       \
--freeze_placeholder_with_value "phase_train->False"
```

The batch joining pattern transforms to a placeholder with the model default shape if `--input_shape` or `--batch`*/*`-b` are not
provided. Otherwise, the placeholder shape has custom parameters.

* `--freeze_placeholder_with_value "phase_train->False"` to switch graph to inference mode
* `--batch`*/*`-b` is applicable to override original network batch
* `--input_shape` is applicable with or without `--input`
* other options are applicable
