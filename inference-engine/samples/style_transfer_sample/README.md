# Neural Style Transfer Sample

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

    -h                      Print a usage message.
    -i "<path>"             Required. Path to an .bmp image.
    -m "<path>"             Required. Path to an .xml file with a trained model.
    -pp "<path>"            Path to a plugin folder.
    -d "<device>"           Specify the target device to infer on; CPU, GPU, FPGA or MYRIAD is acceptable. Sample will look for a suitable plugin for device specified
    -ni "<integer>"         Number of iterations (default 1)
    -pc                     Enables per-layer performance report
    -mean_val_r,
    -mean_val_g,
    -mean_val_b             Mean values. Required if the model needs mean values for preprocessing and postprocessing

```

Running the application with the empty list of options yields the usage message given above and an error message.

You can do inference on an image using a trained model of NST network on Intel&reg; Processors using the following command:
```sh
./style_transfer_sample -i <path_to_image>/cat.bmp -m <path_to_model>/1_decoder_FP32.xml
```

> **NOTE**: Before running the sample with a trained model, make sure the model is converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](./docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md).

### Outputs

The application outputs an styled image(s) (<code>out(1).bmp</code>) which were redrawn in style of model which used for infer.
Style of output images depend on models which use for sample.

## See Also 
* [Using Inference Engine Samples](./docs/IE_DG/Samples_Overview.md)

