# Object Detection YOLO* V4 C++ Demo, API Performance Showcase

This demo showcases Object Detection with YOLO* V4 API.

> **NOTE:** This topic describes usage of C++ implementation of the Object Detection YOLO* V4 Demo API . 

Other demo objectives are:
* Video as input support via OpenCV*
* Visualization of the resulting bounding boxes and text labels (from the `.labels` file) or class number (if no file is provided)
* OpenCV provides resulting bounding boxes, labels, and other information.
You can copy and paste this code without pulling Open Model Zoo demos helpers into your application


## How It Works

On the start-up, the application reads command-line parameters and loads a network to the Inference
Engine. Upon getting a frame from the OpenCV VideoCapture, it performs inference and displays the results.

> **NOTE**: By default, Open Model Zoo demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the demo application or reconvert your model using the Model Optimizer tool with `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Reverse Input Channels** section of [Converting a Model Using General Conversion Parameters](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Converting_Model_General.html).

## Running

Running the application with the <code>-h</code> option yields the following usage message:
```
./object_detection_demo_yolov4 -h
InferenceEngine:
    API version ............ <version>
    Build .................. <number>

object_detection_demo_yolov4 [OPTION]
Options:

    -h                      Print a usage message.
    -i "<path>"             Required. Path to a folder with images or path to an image files: a .ubyte file for LeNetand a .bmp file for the other networks.
    -m "<path>"             Required. Path to an .xml file with a trained model.
      -l "<absolute_path>"    Required for MKLDNN (CPU)-targeted custom layers.Absolute path to a shared library with the kernels impl.
          Or
      -c "<absolute_path>"    Required for clDNN (GPU)-targeted custom kernels.Absolute path to the xml file with the kernels desc.
    -pp "<path>"            Path to a plugin folder.
    -d "<device>"           Specify the target device to infer on; CPU, GPU, FPGA or MYRIAD is acceptable. Sample will look for a suitable plugin for device specified (CPU by default)
    -nt "<integer>"         Number of top results (default 10)
    -pc                     Enables per-layer performance report
    -p_msg                  Enables messages from a plugin
    -t                      Optional. Probability threshold for detections.
    -iou_t                  Optional. Filtering intersection over union threshold for overlapping boxes.
```

Running the application with the empty list of options yields the usage message given above and an error message.

To run the demo, you can use public or pre-trained models. To download the pre-trained models, use the OpenVINO [Model Downloader](../../tools/downloader/README.md) or go to [https://download.01.org/opencv/](https://download.01.org/opencv/).

> **NOTE**: Before running the demo with a trained model, make sure the model is converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html).

You can use the following command to do inference on GPU with a pre-trained object detection model:
```sh
./object_detection_demo_yolov4 -i <path_to_image>/inputImage.bmp -m <path_to_model>/yolo_v4.xml -d CPU
```


## Demo Output

The application outputs an image (`object_detection_demo_yolov4_output.jpg`) with detected objects enclosed in rectangles. It outputs the list of classes of the detected objects along with the respective confidence values and the coordinates of the rectangles to the standard output stream.


## See Also
* [Using Open Model Zoo demos](../README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../tools/downloader/README.md)
