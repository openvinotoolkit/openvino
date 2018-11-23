# Object Detection Sample SSD

This topic demonstrates how to run the Object Detection sample application, which does inference using object detection 
networks like SSD-VGG on Intel® Processors and Intel® HD Graphics.

## Running

Running the application with the <code>-h</code> option yields the following usage message:
```sh
./object_detection_sample_ssd -h
InferenceEngine: 
    API version ............ <version>
    Build .................. <number>

object_detection_sample_ssd [OPTION]
Options:

    -h                      Print a usage message.
    -i "<path>"             Required. Path to an .bmp image.
    -m "<path>"             Required. Path to an .xml file with a trained model.
      -l "<absolute_path>"    Required for MKLDNN (CPU)-targeted custom layers. Absolute path to a shared library with the kernels impl.
          Or
      -c "<absolute_path>"    Required for clDNN (GPU)-targeted custom kernels. Absolute path to the xml file with the kernels desc.
    -pp "<path>"            Path to a plugin folder.
    -d "<device>"           Specify the target device to infer on; CPU, GPU, FPGA or MYRIAD is acceptable. Sample will look for a suitable plugin for device specified
    -pc                     Enables per-layer performance report
    -ni "<integer>"         Number of iterations (default 1)
    -p_msg                  Enables messages from a plugin

```

Running the application with the empty list of options yields the usage message given above and an error message.

To run the sample, you can use a set of pre-trained and optimized models delivered with the package or a Caffe* public model.

**NOTE**: A public model should be converted to the Inference Engine format (`.xml` + `.bin`) using the Model Optimizer tool. For Model Optimizer documentation, see https://software.intel.com/en-us/articles/OpenVINO-ModelOptimizer.

For example, to do inference on a CPU with the OpenVINO&trade; toolkit person detection SSD model (`<INSTAL_DIR>/deployment_tools/intel_models/person-detection-retail-00013`), run the following command:

```sh
./object_detection_sample_ssd -i <path_to_image>/inputImage.bmp -m person-detection-retail-0013.xml -d CPU
```

### Outputs

The application outputs an image (<code>out_0.bmp</code>) with detected objects enclosed in rectangles. It outputs the list of classes 
of the detected objects along with the respective confidence values and the coordinates of the 
rectangles to the standard output stream.

### How it works

Upon the start-up the sample application reads command line parameters and loads a network and an image to the Inference 
Engine plugin. When inference is done, the application creates an 
output image and outputs data to the standard output stream.

## See Also 
* [Using Inference Engine Samples](./docs/Inference_Engine_Developer_Guide/Samples_Overview.md)
