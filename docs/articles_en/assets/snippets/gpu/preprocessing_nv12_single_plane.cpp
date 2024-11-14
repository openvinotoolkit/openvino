#include <openvino/runtime/core.hpp>
#include <openvino/runtime/intel_gpu/ocl/ocl.hpp>
#include <openvino/runtime/intel_gpu/properties.hpp>
#include <openvino/core/preprocess/pre_post_process.hpp>

ov::intel_gpu::ocl::ClImage2DTensor get_yuv_tensor();

int main() {
    ov::Core core;
    auto model = core.read_model("model.xml");

    //! [init_preproc]
    using namespace ov::preprocess;
    auto p = PrePostProcessor(model);
    p.input().tensor().set_element_type(ov::element::u8)
                      .set_color_format(ColorFormat::NV12_SINGLE_PLANE)
                      .set_memory_type(ov::intel_gpu::memory_type::surface);
    p.input().preprocess().convert_color(ov::preprocess::ColorFormat::BGR);
    p.input().model().set_layout("NCHW");
    auto model_with_preproc = p.build();
    //! [init_preproc]

    auto compiled_model = core.compile_model(model_with_preproc, "GPU");
    auto context = compiled_model.get_context().as<ov::intel_gpu::ocl::ClContext>();
    auto infer_request = compiled_model.create_infer_request();

{
    //! [single_batch]
    auto input_yuv = model_with_preproc->input(0);
    ov::intel_gpu::ocl::ClImage2DTensor yuv_tensor = get_yuv_tensor();
    infer_request.set_tensor(input_yuv.get_any_name(), yuv_tensor);
    infer_request.infer();
    //! [single_batch]
}

{
    auto yuv_tensor_0 = get_yuv_tensor();
    auto yuv_tensor_1 = get_yuv_tensor();
    //! [batched_case]
    auto input_yuv = model_with_preproc->input(0);
    std::vector<ov::Tensor> yuv_tensors = {yuv_tensor_0, yuv_tensor_1};
    infer_request.set_tensors(input_yuv.get_any_name(), yuv_tensors);
    infer_request.infer();
    //! [batched_case]
}
    return 0;
}
