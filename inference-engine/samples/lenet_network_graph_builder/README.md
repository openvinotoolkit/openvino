# Lenet Number Classifications Network using Graph Builder API

This sample demonstrates how to execute inference using Inference Engine Graph Builder API to build a network on example of the LeNet classifications network.
XML file is not required for network building now. Inference Engine Graph Builder API allows building of a network "on the fly" from source code. The sample uses 1-channel ubyte pictures as input.
<br>

## Running

Running the application with the <code>-h</code> option yields the following usage message:
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
    -d "<device>"           Specify the target device to infer on this. Sample will look for a suitable plugin for device specified(default value is CPU)
    -pp "<path>"            Path to a plugin folder
    -pc                     Enables per-layer performance report
    -nt "<integer>"         Number of top results (default 10)
    -ni "<integer>"         Number of iterations (default 1)

```

Running the application with empty list of options yields the usage message given above.

For example, to do inference of an ubyte image on a GPU run the following command:
```sh
./lenet_network_graph_builder -i <path_to_image> -m <path_to_weights_file> -d GPU
```

### Outputs

By default the application outputs top-10 inference results for each infer request.
In addition to this information it will provide throughput value measured in frames per seconds.

### How it works

Upon the start-up the sample application reads command line parameters and loads a network and an image to the Inference 
Engine plugin. When inference is done, the application creates an 
output image and outputs data to the standard output stream.

Upon the start-up the sample reads command line parameters and builds a network using Graph Builder API and passed weights file.
Then, the application loads built network and an image to the Inference Engine plugin.

When inference is done, the application outputs inference results to the standard output stream.

## See Also
* [Using Inference Engine Samples](./docs/IE_DG/Samples_Overview.md)
