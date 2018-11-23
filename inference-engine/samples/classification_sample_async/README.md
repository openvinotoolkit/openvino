# Image Classification Sample Async

This sample demonstrates how to build and execute inference in pipelined mode on example of classifications networks.

The pipelined mode might increase the throghput of the pictures. The latency of one inference will be the same as for syncronious execution.
<br>
The throughput is increased due to follow reasons:
* Some plugins have heterogenity inside themselves. Transferring of data, execution on remote device, doigin pre-processing and post-processing on the host
* Using of explicit heterogenious plugin with execution of different parts of network on differnt devices

When two and more devices are involved in inference process of one picture, creation of several infer requests and starting of asynchronious inference allows to utilize devices the most efficient way.
If two devices are involved in execution, the most optimal value for -nireq option is 2
To do this efficiently, Classification Sample Async uses round-robin algorithm for infer requests. It starts execution for the current infer request and swith for the waiting of results for previous one. After finishing of wait, it switches infer requsts and repeat the procedure.

Another required aspect of seeing good throughput is number of iterations. Only having big number of iterations you can emulate the real application work and see performance

The batch mode is an independent attribute on the pipelined mode. Pipelined mode works efficiently with any batch size.

## Running

Running the application with the <code>-h</code> option yields the following usage message:
```sh
./classification_sample_async -h
InferenceEngine:
    API version ............ <version>
    Build .................. <number>

classification_sample_async [OPTION]
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
    -d "<device>"           
                            Specify the target device to infer on; CPU, GPU, FPGA or MYRIAD is acceptable. Sample will look for a suitable plugin for device specified
    -nt "<integer>"         
                            Number of top results (default 10)
    -ni "<integer>"         
                            Number of iterations (default 1)
    -pc                     
                            Enables per-layer performance report
    -nireq "<integer>"
                            Number of infer request for pipelined mode (default 1)
    -p_msg                  
                            Enables messages from a plugin

```

Running the application with the empty list of options yields the usage message given above and an error message.

You can do inference on an image using a trained AlexNet network on FPGA with fallback to Intel&reg; Processors using the following command:
```sh
./classification_sample_async -i <path_to_image>/cat.bmp -m <path_to_model>/alexnet_fp32.xml -nt 5 -d HETERO:FPGA,CPU -nireq 2 -ni 200
```

### Outputs

By default the application outputs top-10 inference results for each infer request.
In addition to this information it will provide throughput value measured in frames per seconds.

### How it works

Upon the start-up the sample application reads command line parameters and loads a network and an image to the Inference 
Engine plugin.
Then application creates several infer requests pointed in -nireq parameter and loads pictures for inference.

Then in the loop it starts inference for the current infer request and switch for waiting of another one. When results are ready, infer requests will be swapped.

When inference is done, the application outputs data to the standard output stream.

## See Also
* [Using Inference Engine Samples](./docs/Inference_Engine_Developer_Guide/Samples_Overview.md)
