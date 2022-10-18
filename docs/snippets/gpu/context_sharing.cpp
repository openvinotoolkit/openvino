#include <openvino/runtime/core.hpp>
#include <openvino/runtime/intel_gpu/ocl/ocl.hpp>

cl::Context get_ocl_context(); // a function which returns cl context created on the app side

int main() {
{
    //! [context_sharing_get_from_ov]

    // ...

    // initialize the core and load the network
    ov::Core core;
    auto model = core.read_model("model.xml");
    auto compiled_model = core.compile_model(model, "GPU");
    auto infer_request = compiled_model.create_infer_request();


    // obtain the RemoteContext from the compiled model object and cast it to ClContext
    auto gpu_context = compiled_model.get_context().as<ov::intel_gpu::ocl::ClContext>();
    // obtain the OpenCL context handle from the RemoteContext,
    // get device info and create a queue
    cl::Context cl_context = gpu_context;
    cl::Device device = cl::Device(cl_context.getInfo<CL_CONTEXT_DEVICES>()[0].get(), true);
    cl_command_queue_properties props = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
    cl::CommandQueue queue = cl::CommandQueue(cl_context, device, props);

    // create the OpenCL buffer within the obtained context
    auto input = model->get_parameters().at(0);
    auto input_size = ov::shape_size(input->get_shape());
    cl_int err;
    cl::Buffer shared_buffer(cl_context, CL_MEM_READ_WRITE, input_size, NULL, &err);
    // wrap the buffer into RemoteBlob
    auto shared_blob = gpu_context.create_tensor(input->get_element_type(), input->get_shape(), shared_buffer);

    // ...
    // execute user kernel
    cl::Program program;
    cl::Kernel kernel(program, "user_kernel");
    kernel.setArg(0, shared_buffer);
    queue.enqueueNDRangeKernel(kernel,
                               cl::NDRange(0),
                               cl::NDRange(input_size),
                               cl::NDRange(1),
                               nullptr,
                               nullptr);
    queue.finish();
    // ...
    // pass results to the inference
    infer_request.set_tensor(input, shared_blob);
    infer_request.infer();
    //! [context_sharing_get_from_ov]
}
{

    //! [context_sharing_user_handle]
    cl::Context ctx = get_ocl_context();

    ov::Core core;
    auto model = core.read_model("model.xml");

    // share the context with GPU plugin and compile ExecutableNetwork
    auto remote_context = ov::intel_gpu::ocl::ClContext(core, ctx.get());
    auto exec_net_shared = core.compile_model(model, remote_context);
    auto inf_req_shared = exec_net_shared.create_infer_request();


    // ...
    // do OpenCL processing stuff
    // ...

    // run the inference
    inf_req_shared.infer();
    //! [context_sharing_user_handle]
}
    return 0;
}
