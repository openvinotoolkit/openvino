# Neural Style Transfer Sample {#InferenceEngineNeuralStyleTransferSampleApplication}

This topic demonstrates how to build and run the Neural Style Transfer sample (NST sample) application, which does
inference using models of style transfer topology.

## Running

Running the application with the <code>-h</code> option yields the following usage message:
```sh
./style_transfer_sample --help
InferenceEngine:
    API version ............ <version>
    Build .................. <number>

style_transfer_sample [OPTION]
Options:

    -h
                            Print a usage message.
    -i "<path1>" "<path2>"
                            Required. Path to a folder with images or path to an image files: a .ubyte file for LeNet
                            and a .bmp file for the other networks.
    -m "<path>"
                            Required. Path to an .xml file with a trained model.
        -l "<absolute_path>"
                            Optional. Absolute path to library with MKL-DNN (CPU) custom layers (*.so).
        Or
        -c "<absolute_path>"
                            Optional. Absolute path to clDNN (GPU) custom layers config (*.xml).
    -pp "<path>"
                            Path to a plugin folder.
    -p "<name>"
                            Plugin name. For example MKLDNNPlugin. If this parameter is pointed, the sample will look for this plugin only
    -d "<device>"
                            Specify the target device to infer on; CPU, GPU, FPGA or MYRIAD is acceptable. Sample will look for a suitable plugin for device specified
    -nt "<integer>"
                            Number of top results (default 10)
    -ni "<integer>"
                            Number of iterations (default 1)
    -pc
                            Enables per-layer performance report

```

Running the application with the empty list of options yields the usage message given above and an error message.

You can do inference on an image using a trained model of NST network on Intel&reg; Processors using the following command:
```sh
./style_transfer_sample -i <path_to_image>/cat.bmp -m <path_to_model>/1_decoder_FP32.xml
```

### Outputs

The application outputs an styled image(s) (<code>out(1).bmp</code>) which were redrawn in style of model which used for infer.
Style of output images depend on models which use for sample.

## See Also 
* [Using Inference Engine Samples](@ref SamplesOverview)

