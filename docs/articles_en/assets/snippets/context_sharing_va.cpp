#ifdef ENABLE_LIBVA
#include <openvino/runtime/core.hpp>
#include <openvino/runtime/intel_gpu/ocl/va.hpp>
#include <openvino/runtime/intel_gpu/properties.hpp>
#include <openvino/core/preprocess/pre_post_process.hpp>

VADisplay get_va_display();
VASurfaceID decode_va_surface();

int main() {
    // initialize the objects
    ov::Core core;
    auto model = core.read_model("model.xml");


    // ...


    //! [context_sharing_va]

    // ...

    using namespace ov::preprocess;
    auto p = PrePostProcessor(model);
    p.input().tensor().set_element_type(ov::element::u8)
                      .set_color_format(ov::preprocess::ColorFormat::NV12_TWO_PLANES, {"y", "uv"})
                      .set_memory_type(ov::intel_gpu::memory_type::surface);
    p.input().preprocess().convert_color(ov::preprocess::ColorFormat::BGR);
    p.input().model().set_layout("NCHW");
    model = p.build();

    VADisplay disp = get_va_display();
    // create the shared context object
    auto shared_va_context = ov::intel_gpu::ocl::VAContext(core, disp);
    // compile model within a shared context
    auto compiled_model = core.compile_model(model, shared_va_context);

    auto input0 = model->get_parameters().at(0);
    auto input1 = model->get_parameters().at(1);

    auto shape = input0->get_shape();
    auto width = shape[1];
    auto height = shape[2];

    // execute decoding and obtain decoded surface handle
    VASurfaceID va_surface = decode_va_surface();
    //     ...
    //wrap decoder output into RemoteBlobs and set it as inference input
    auto nv12_blob = shared_va_context.create_tensor_nv12(height, width, va_surface);

    auto infer_request = compiled_model.create_infer_request();
    infer_request.set_tensor(input0->get_friendly_name(), nv12_blob.first);
    infer_request.set_tensor(input1->get_friendly_name(), nv12_blob.second);
    infer_request.start_async();
    infer_request.wait();
    //! [context_sharing_va]

    return 0;
}
#endif  // ENABLE_LIBVA
