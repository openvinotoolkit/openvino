#include <openvino/runtime/core.hpp>
#include <openvino/runtime/intel_gpu/ocl/ocl.hpp>


cl::CommandQueue get_ocl_queue(); // a function which returns cl queue created on the app side
cl::Context get_ocl_context(); // a function which returns cl context created on the app side

int main() {
    //! [queue_sharing]

    // ...

    // initialize the core and read the model
    ov::Core core;
    auto model = core.read_model("model.xml");

    // get opencl queue object
    cl::CommandQueue queue = get_ocl_queue();
    cl::Context cl_context = get_ocl_context();

    // share the queue with GPU plugin and compile model
    auto remote_context = ov::intel_gpu::ocl::ClContext(core, queue.get());
    auto exec_net_shared = core.compile_model(model, remote_context);

    auto input = model->get_parameters().at(0);
    auto input_size = ov::shape_size(input->get_shape());
    auto output = model->get_results().at(0);
    auto output_size = ov::shape_size(output->get_shape());
    cl_int err;

    // create the OpenCL buffers within the context
    cl::Buffer shared_in_buffer(cl_context, CL_MEM_READ_WRITE, input_size, NULL, &err);
    cl::Buffer shared_out_buffer(cl_context, CL_MEM_READ_WRITE, output_size, NULL, &err);
    // wrap in and out buffers into RemoteTensor and set them to infer request
    auto shared_in_blob = remote_context.create_tensor(input->get_element_type(), input->get_shape(), shared_in_buffer);
    auto shared_out_blob = remote_context.create_tensor(output->get_element_type(), output->get_shape(), shared_out_buffer);
    auto infer_request = exec_net_shared.create_infer_request();
    infer_request.set_tensor(input, shared_in_blob);
    infer_request.set_tensor(output, shared_out_blob);

    // ...
    // execute user kernel
    cl::Program program;
    cl::Kernel kernel_preproc(program, "user_kernel_preproc");
    kernel_preproc.setArg(0, shared_in_buffer);
    queue.enqueueNDRangeKernel(kernel_preproc,
                               cl::NDRange(0),
                               cl::NDRange(input_size),
                               cl::NDRange(1),
                               nullptr,
                               nullptr);
    // Blocking clFinish() call is not required, but this barrier is added to the queue to guarantee that user kernel is finished
    // before any inference primitive is started
    queue.enqueueBarrierWithWaitList(nullptr, nullptr);
    // ...

    // pass results to the inference
    // since the remote context is created with queue sharing, start_async() guarantees that scheduling is finished
    infer_request.start_async();

    // execute some postprocessing kernel.
    // infer_request.wait() is not called, synchonization between inference and post-processing is done via
    // enqueueBarrierWithWaitList call.
    cl::Kernel kernel_postproc(program, "user_kernel_postproc");
    kernel_postproc.setArg(0, shared_out_buffer);
    queue.enqueueBarrierWithWaitList(nullptr, nullptr);
    queue.enqueueNDRangeKernel(kernel_postproc,
                               cl::NDRange(0),
                               cl::NDRange(output_size),
                               cl::NDRange(1),
                               nullptr,
                               nullptr);

    // Wait for pipeline completion
    queue.finish();
    //! [queue_sharing]

    return 0;
}
