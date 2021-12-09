#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120

#include <ie_core.hpp>
#include <CL/cl2.hpp>
#include <gpu/gpu_context_api_ocl.hpp>


int main() {
using namespace InferenceEngine;
//! [part0]


// ...


// initialize the core and load the network 
InferenceEngine::Core ie;
auto net = ie.ReadNetwork("network.xml");
auto exec_net = ie.LoadNetwork(net, "GPU");


// obtain the RemoteContext pointer from the executable network object
auto cldnn_context = exec_net.GetContext();
// obtain the OpenCL context handle from the RemoteContext,
// get device info and create a queue
cl::Context ctx = std::dynamic_pointer_cast<cl::Context>(cldnn_context);
_device = cl::Device(_context.getInfo<CL_CONTEXT_DEVICES>()[0].get(), true);
cl::CommandQueue _queue;
cl_command_queue_properties props = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
_queue = cl::CommandQueue(_context, _device, props);


// create the OpenCL buffer within the obtained context
cl::Buffer shared_buffer(ctx, CL_MEM_READ_WRITE, image_size * num_channels, NULL, &err);
// wrap the buffer into RemoteBlob
auto shared_blob = gpu::make_shared_blob(input_info->getTensorDesc(), cldnn_context, shared_buffer);


// ...
// execute user kernel
cl::Kernel kernel(program, kernelName.c_str());
kernel.setArg(0, shared_buffer);
queue.enqueueNDRangeKernel(kernel,
                           cl::NDRange(0),
                           cl::NDRange(image_size),
                           cl::NDRange(1),
                           0, // wait events *
                           &profileEvent);
queue.finish();
// ...


// pass results to the inference
inf_req_shared.SetBlob(input_name, shared_blob);
inf_req_shared.Infer();
//! [part0]

return 0;
}
