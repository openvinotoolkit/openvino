# LeNet Number Classifications Network Using Graph Builder API

This sample demonstrates how to execute inference using Inference Engine Graph Builder API to build a network on example of the LeNet classifications network.

XML file is not required for network building now. Inference Engine Graph Builder API allows building of a network "on the fly" from source code. The sample uses one-channel `ubyte` pictures as input.

## How It Works

Upon the start-up the sample reads command line parameters and builds a network using Graph Builder API and passed weights file.
Then, the application loads built network and an image to the Inference Engine plugin.

When inference is done, the application outputs inference results to the standard output stream.

> **NOTE**: This sample is implemented to support models with FP32 weights only.

> **NOTE**: By default, Inference Engine samples and demos expect input with BGR channels order. If you trained your model to work with RGB order, you need to manually rearrange the default channels order in the sample or demo application or reconvert your model using the Model Optimizer tool with `--reverse_input_channels` argument specified. For more information about the argument, refer to **When to Specify Input Shapes** section of [Converting a Model Using General Conversion Parameters](./docs/MO_DG/prepare_model/convert_model/Converting_Model_General.md).

## Running

Running the application with the `-h` option yields the following usage message:
```sh
./lenet_network_graph_builder -h
InferenceEngine:
    API version ............ <version>
    Build .................. <number>

lenet_network_graph_builder [OPTION]
Options:

    -h                      Print a usage message.
    -m "<path>"             Path to a .bin file with weights for trained model
    -i "<path>"             Required. Path to image or folder with images
    -d "<device>"           Specify the target device to infer on this. Sample will look for a suitable plugin for device specified. Default value is CPU
    -pp "<path>"            Path to a plugin folder
    -pc                     Enables per-layer performance report
    -nt "<integer>"         Number of top results. Default value is 10
    -ni "<integer>"         Number of iterations. Default value is 1

```

Running the application with empty list of options yields the usage message given above.

For example, to do inference of an ubyte image on a GPU run the following command:
```sh
./lenet_network_graph_builder -i <path_to_image> -m <path_to_weights_file> -d GPU
```

## Sample Output

By default the application outputs top-10 inference results for each infer request.
In addition to this information it will provide throughput value measured in frames per seconds.

## See Also
* [Using Inference Engine Samples](./docs/IE_DG/Samples_Overview.md)
