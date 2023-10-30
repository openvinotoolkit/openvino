// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <utility>
#include <vector>
#include <memory>
#include <fcntl.h>

#include "common_test_utils/test_assertions.hpp"
#include "openvino/core/except.hpp"

#include "openvino/runtime/core.hpp"
#include "openvino/runtime/intel_gpu/properties.hpp"

#include <remote_blob_tests/remote_blob_helpers.hpp>
#include <common_test_utils/test_common.hpp>
#include <functional_test_utils/plugin_cache.hpp>
#include "base/ov_behavior_test_utils.hpp"
#include "ov_models/subgraph_builders.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "openvino/core/preprocess/pre_post_process.hpp"
#include "transformations/utils/utils.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"

#if defined(__unix__) && defined(ENABLE_LIBVA) && defined(ENABLE_LIBVA_DRM)

#include <openvino/runtime/intel_gpu/ocl/va.hpp>
#include <va/va_drm.h>
class VASurfaceSharing_Test : public ov::test::TestsCommon {
protected:
    std::shared_ptr<ngraph::Function> fn_ptr;

    void SetUp() override {
        fn_ptr = ngraph::builder::subgraph::makeSplitMultiConvConcat();
    }
};

TEST_F(VASurfaceSharing_Test, NV12toBGR_image_two_planes_surface_sharing) {
#if defined(ANDROID)
    GTEST_SKIP();
#endif
    const int height = 640;
    const int width = 640;

    // ------------------------------------------------------
    // Prepare input data
    //ov::Tensor fake_image_data_y = ov::test::utils::create_and_fill_tensor(ov::element::u8, {1, height, width, 1}, 50, 0, 1);
    //ov::Tensor fake_image_data_uv = ov::test::utils::create_and_fill_tensor(ov::element::u8, {1, height / 2, width / 2, 2}, 256, 0, 1);

    auto ie = ov::Core();

    // ------------------------------------------------------
    // inference using remote tensor
    auto fn_ptr_remote = ngraph::builder::subgraph::makeConvPoolRelu({1, 3, height, width});

    using namespace ov::preprocess;
    auto p = PrePostProcessor(fn_ptr_remote);
    p.input().tensor().set_element_type(ov::element::u8)
                      .set_color_format(ColorFormat::NV12_TWO_PLANES, {"y", "uv"})
                      .set_memory_type(ov::intel_gpu::memory_type::surface);
    p.input().preprocess().convert_color(ColorFormat::BGR);
    p.input().model().set_layout("NCHW");
    auto function = p.build();


    auto create_va_display = [&]() {
        int32_t fd = open("/dev/dri/renderD128", O_RDWR);
        VADisplay va_dpy = vaGetDisplayDRM(fd);
        int major = 0, minor = 0;
        vaInitialize(va_dpy, &major, &minor);
        return va_dpy;
    };

    auto alloc_va_surface = [&](VADisplay va_display, int width, int height) {
        VASurfaceID va_surface;
        VASurfaceAttrib surface_attrib{};
        surface_attrib.type = VASurfaceAttribPixelFormat;
        surface_attrib.flags = VA_SURFACE_ATTRIB_SETTABLE;
        surface_attrib.value.type = VAGenericValueTypeInteger;
        surface_attrib.value.value.i = VA_FOURCC_NV12;
        vaCreateSurfaces(va_display, VA_RT_FORMAT_YUV420, width, height, &va_surface,
                        1, &surface_attrib, 1);
        return va_surface;
    };

    auto param_input_y = fn_ptr_remote->get_parameters().at(0);
    auto param_input_uv = fn_ptr_remote->get_parameters().at(1);

    VADisplay disp = create_va_display();
    std::cout << "VA display creation." << std::endl;
    auto shared_va_context = ov::intel_gpu::ocl::VAContext(ie, disp);
    auto exec_net_b = ie.compile_model(function, shared_va_context);
    auto inf_req_remote = exec_net_b.create_infer_request();

    VASurfaceID va_surface = alloc_va_surface(disp, width, height);
    std::cout << "VA surface creation." << std::endl;

    for (int i = 0; i < 5; ++i) {
        std::cout << "Iteration: " << i << " Started. " << "Surface id: " << va_surface << std::endl;

        ov::AnyMap tensor_params = {{ov::intel_gpu::shared_mem_type.name(), ov::intel_gpu::SharedMemType::VA_SURFACE},
                                {ov::intel_gpu::dev_object_handle.name(), va_surface},
                                {ov::intel_gpu::va_plane.name(), uint32_t(0)}};
        auto y_tensor = shared_va_context.create_tensor(ov::element::u8, {1, height, width, 1}, tensor_params);
        tensor_params[ov::intel_gpu::va_plane.name()] = uint32_t(1);
        auto uv_tensor = shared_va_context.create_tensor(ov::element::u8, {1, height / 2, width / 2, 2}, tensor_params);

        inf_req_remote.set_tensor(*param_input_y->output(0).get_tensor().get_names().begin(), y_tensor);
        inf_req_remote.set_tensor(*param_input_uv->output(0).get_tensor().get_names().begin(), uv_tensor);

        inf_req_remote.infer();
        std::cout << "Iteration: " << i << " Completed" << std::endl;
    }

    auto output_tensor_shared = inf_req_remote.get_tensor(function->get_results().at(0));
    ASSERT_NO_THROW(output_tensor_shared.data());
}

#endif