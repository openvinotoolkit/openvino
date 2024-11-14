#include <openvino/runtime/intel_gpu/ocl/ocl.hpp>
#include <openvino/runtime/intel_gpu/properties.hpp>
#include <openvino/core/preprocess/pre_post_process.hpp>

ov::intel_gpu::ocl::ClImage2DTensor get_y_tensor();
ov::intel_gpu::ocl::ClImage2DTensor get_uv_tensor();

int main() {
    ov::Core core;
    auto model = core.read_model("model.xml");

    //! [init_preproc]
    using namespace ov::preprocess;
    auto p = PrePostProcessor(model);
    p.input().tensor().set_element_type(ov::element::u8)
                      .set_layout("NHWC")
                      .set_memory_type(ov::intel_gpu::memory_type::surface);
    p.input().model().set_layout("NCHW");
    auto model_with_preproc = p.build();
    //! [init_preproc]

    auto compiled_model = core.compile_model(model_with_preproc, "GPU");
    auto remote_context = compiled_model.get_context().as<ov::intel_gpu::ocl::ClContext>();
    auto input = model->input(0);
    auto infer_request = compiled_model.create_infer_request();

{
    //! [single_batch]
    cl::Image2D img_y_plane;
    auto input_y = model_with_preproc->input(0);
    auto remote_y_tensor = remote_context.create_tensor(input_y.get_element_type(), input.get_shape(), img_y_plane);
    infer_request.set_tensor(input_y.get_any_name(), remote_y_tensor);
    infer_request.infer();
    //! [single_batch]
}

{
    //! [batched_case]
    cl::Image2D img_y_plane_0, img_y_plane_l;
    auto input_y = model_with_preproc->input(0);
    auto remote_y_tensor_0 = remote_context.create_tensor(input_y.get_element_type(), input.get_shape(), img_y_plane_0);
    auto remote_y_tensor_1 = remote_context.create_tensor(input_y.get_element_type(), input.get_shape(), img_y_plane_l);
    std::vector<ov::Tensor> y_tensors = {remote_y_tensor_0, remote_y_tensor_1};
    infer_request.set_tensors(input_y.get_any_name(), y_tensors);
    infer_request.infer();
    //! [batched_case]
}
    return 0;
}
