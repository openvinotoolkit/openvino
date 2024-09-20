#include <openvino/runtime/core.hpp>
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
                      .set_color_format(ov::preprocess::ColorFormat::NV12_TWO_PLANES, {"y", "uv"})
                      .set_memory_type(ov::intel_gpu::memory_type::surface);
    p.input().preprocess().convert_color(ov::preprocess::ColorFormat::BGR);
    p.input().model().set_layout("NCHW");
    auto model_with_preproc = p.build();
    //! [init_preproc]

    auto compiled_model = core.compile_model(model, "GPU");
    auto context = compiled_model.get_context().as<ov::intel_gpu::ocl::ClContext>();
    auto input = model->get_parameters().at(0);
    auto infer_request = compiled_model.create_infer_request();

{
    //! [single_batch]
    auto input0 = model_with_preproc->get_parameters().at(0);
    auto input1 = model_with_preproc->get_parameters().at(1);
    ov::intel_gpu::ocl::ClImage2DTensor y_tensor = get_y_tensor();
    ov::intel_gpu::ocl::ClImage2DTensor uv_tensor = get_uv_tensor();
    infer_request.set_tensor(input0->get_friendly_name(), y_tensor);
    infer_request.set_tensor(input1->get_friendly_name(), uv_tensor);
    infer_request.infer();
    //! [single_batch]
}

{
    auto y_tensor_0 = get_y_tensor();
    auto y_tensor_1 = get_y_tensor();
    auto uv_tensor_0 = get_uv_tensor();
    auto uv_tensor_1 = get_uv_tensor();
    //! [batched_case]
    auto input0 = model_with_preproc->get_parameters().at(0);
    auto input1 = model_with_preproc->get_parameters().at(1);
    std::vector<ov::Tensor> y_tensors = {y_tensor_0, y_tensor_1};
    std::vector<ov::Tensor> uv_tensors = {uv_tensor_0, uv_tensor_1};
    infer_request.set_tensors(input0->get_friendly_name(), y_tensors);
    infer_request.set_tensors(input1->get_friendly_name(), uv_tensors);
    infer_request.infer();
    //! [batched_case]
}
    return 0;
}
