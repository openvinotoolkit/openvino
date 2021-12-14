#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120

#include <ie_core.hpp>
#include <CL/cl2.hpp>
#include <gpu/gpu_context_api_ocl.hpp>


int main() {
using namespace InferenceEngine;
//! [part0]


// ...


// initialize the core and read the network
InferenceEngine::Core ie;
auto net = ie.ReadNetwork("network.xml");

// initialize opencl context and create queue
cl::Context ctx = get_my_OpenCL_context();
cl::CommandQueue queue = get_my_OpenCL_queue();

// share the queue with GPU plugin and compile ExecutableNetwork
auto remote_context = gpu::make_shared_context(ie, "GPU", queue.get());
auto exec_net_shared = ie.LoadNetwork(net, remote_context);

// create the OpenCL buffers within the context
cl::Buffer shared_in_buffer(ctx, CL_MEM_READ_WRITE, image_size * num_channels, NULL, &err);
cl::Buffer shared_out_buffer(ctx, CL_MEM_READ_WRITE, image_size * num_channels, NULL, &err);
// wrap in and out buffers into RemoteBlob and set them to infer request
auto shared_in_blob = gpu::make_shared_blob(input_info->getTensorDesc(), remote_context, shared_in_buffer);
auto shared_out_blob = gpu::make_shared_blob(out_data->getTensorDesc(), remote_context, shared_out_buffer);
auto infer_request = exec_net_shared.CreateInferRequest();
infer_request.SetBlob(input_name, shared_in_blob);
infer_request.SetBlob(output_name, shared_out_blob);

// ...
// execute user kernel
cl::Kernel kernel_preproc(program, kernel_name_preproc.c_str());
kernel_preproc.setArg(0, shared_in_buffer);
queue.enqueueNDRangeKernel(kernel_preproc,
                           cl::NDRange(0),
                           cl::NDRange(image_size),
                           cl::NDRange(1),
                           nullptr,  // wait events *
                           &profileEvent);
// Blocking clFinish() call is not required, but this barrier is added to the queue to guarantee that user kernel is finished
// before any inference primitive is started
queue.enqueueBarrierWithWaitList(nullptr, nullptr);
// ...

// pass results to the inference
// since the remote context is created with queue sharing, StartAsync() guarantees that scheduling is finished
infer_request.StartAsync();

// execute some postprocessing kernel.
// infer_request.Wait() is not called, synchonization between inference and post-processing is done via
// enqueueBarrierWithWaitList call.
cl::Kernel kernel_postproc(program, kernel_name_postproc.c_str());
kernel_postproc.setArg(0, shared_out_buffer);
queue.enqueueBarrierWithWaitList(nullptr, nullptr);
queue.enqueueNDRangeKernel(kernel_postproc,
                           cl::NDRange(0),
                           cl::NDRange(image_size),
                           cl::NDRange(1),
                           nullptr,  // wait events *
                           &profileEvent);

// Wait for pipeline completion
queue.finish();
//! [part0]

return 0;
}
