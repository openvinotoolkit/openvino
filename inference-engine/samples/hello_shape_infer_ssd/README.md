# Hello Shape Infer Sample

This topic demonstrates how to run the Hello Shape Infer SSD application, which does inference using object detection
networks like SSD-VGG. The sample shows how to use [Shape Inference feature](./docs/IE_DG/ShapeInference.md).

## Running

You can use the following command to do inference on Intel&reg; Processors on an image using a trained SSD network:
```sh
./hello_shape_infer_ssd <path_to_model>/ssd_300.xml <path_to_image>/500x500.bmp CPU 3
```

> **NOTE**: Before running the sample with a trained model, make sure the model is converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](./docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md).

### Outputs

The application renders an image with detected objects enclosed in rectangles. It outputs the list of classes
of the detected objects along with the respective confidence values and the coordinates of the
rectangles to the standard output stream.

## See Also
* [Using Inference Engine Samples](./docs/IE_DG/Samples_Overview.md)
