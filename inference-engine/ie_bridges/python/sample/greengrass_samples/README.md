# This README demonstrates use of all GreenGrass samples
 
# GreenGrass Classification Sample

This topic demonstrates how to build and run the GreenGrass Image Classification sample application, which does inference using image classification networks like AlexNet and GoogLeNet on on Intel® Processors, Intel® HD Graphics and Intel® FPGA.

## Running

1. Modify the "accelerator" parameter inside the sample to deploy the sample on any accelerator option of your choice(CPU/GPU/FPGA)  
   For CPU, please specify "CPU"  
   For GPU, please specify "GPU"  
   For FPGA, please specify "HETERO:FPGA,CPU"  
2. Enable the option(s) on how output is displayed/consumed 
3. Now follow the instructions listed in the Greengrass-FaaS-User-Guide.pdf to create the lambda and deploy on edge device using Greengrass
 
### Outputs

The application publishes top-10 results on AWS IoT Cloud every second by default. For other output consumption options, please refer to Greengrass-FaaS-User-Guide.pdf

### How it works

Upon deployment,the sample application loads a network and an image to the Inference Engine plugin. When inference is done, the application publishes results to AWS IoT Cloud 

=====================================================================================================

# GreenGrass Object Detection Sample SSD

This topic demonstrates how to run the GreenGrass Object Detection SSD sample application, which does inference using object detection networks like Squeezenet-SSD on Intel® Processors, Intel® HD Graphics and Intel® FPGA.

## Running

1. Modify the "accelerator" parameter inside the sample to deploy the sample on any accelerator option of your choice(CPU/GPU/FPGA)  
   For CPU, please specify "CPU"  
   For GPU, please specify "GPU"  
   For FPGA, please specify "HETERO:FPGA,CPU"  
2. Enable the option(s) on how output is displayed/consumed 
3. Set the variable is_async_mode to 'True' for Asynchronous execution and 'False' for Synchronous execution
3. Now follow the instructions listed in the Greengrass-FaaS-User-Guide.pdf to create the lambda and deploy on edge device using Greengrass
 
### Outputs

The application publishes detection outputs such as class label, class confidence, and bounding box coordinates on AWS IoT Cloud every second. For other output consumption options, please refer to Greengrass-FaaS-User-Guide.pdf  

### How it works

Upon deployment,the sample application loads a network and an image to the Inference Engine plugin. When inference is done, the application publishes results to AWS IoT Cloud 
 


