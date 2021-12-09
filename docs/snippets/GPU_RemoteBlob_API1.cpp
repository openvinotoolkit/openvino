#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120

#include <ie_core.hpp>
#include <CL/cl2.hpp>
#include <gpu/gpu_context_api_ocl.hpp>


int main() {
using namespace InferenceEngine;
//! [part1]

// ...

cl::Context ctx = get_my_OpenCL_context();

// share the context with GPU plugin and compile ExecutableNetwork
auto remote_context = gpu::make_shared_context(ie, "GPU", ocl_instance->_context.get());
auto exec_net_shared = ie.LoadNetwork(net, remote_context);
auto inf_req_shared = exec_net_shared.CreateInferRequest();


// ...
// do OpenCL processing stuff
// ...

// run the inference
inf_req_shared.Infer();
//! [part1]

return 0;
}
