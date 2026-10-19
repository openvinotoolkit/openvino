// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#if (defined(OV_GPU_WITH_OCL_RT) || defined(OV_GPU_WITH_ZE_RT)) && !defined(_WIN32) && defined(ENABLE_LIBVA) && defined(ENABLE_LIBVA_DRM)

#include <algorithm>
#include <numeric>
#include <string>
#include <vector>

#include "common_test_utils/subgraph_builders/conv_pool_relu.hpp"
#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/test_common.hpp"
#include "common_test_utils/test_constants.hpp"
#include "openvino/core/preprocess/pre_post_process.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/intel_gpu/properties.hpp"
#include "openvino/runtime/intel_gpu/remote_properties.hpp"

#include "remote_tensor_tests/helpers.hpp"

using namespace ::testing;

namespace {
constexpr size_t surface_width = 320;
constexpr size_t surface_height = 240;

bool has_gpu_device() {
    ov::Core core;
    const auto devices = core.get_available_devices();
    return std::any_of(devices.begin(), devices.end(), [](const std::string& device) {
        return device.find(ov::test::utils::DEVICE_GPU) != std::string::npos;
    });
}
}  // namespace

class OVRemoteTensorVA_Test : public ov::test::TestsCommon {
protected:
    VADevice va_device;
    VASurfaceID surface = VA_INVALID_SURFACE;

    void SetUp() override {
        if (!va_device.is_valid())
            GTEST_SKIP() << "VA-API display is not available on this host";

        if (!has_gpu_device())
            GTEST_SKIP() << "GPU plugin has no available device";

        surface = va_device.create_nv12_surface(surface_width, surface_height);
        if (surface == VA_INVALID_SURFACE)
            GTEST_SKIP() << "VA-API driver can't allocate an NV12 surface";
    }

    void TearDown() override {
        va_device.destroy_surface(surface);
    }
};

TEST_F(OVRemoteTensorVA_Test, smoke_va_context_from_display) {
    ov::Core core;
    ov::intel_gpu::ocl::VAContext context(core, va_device.get());

    OV_ASSERT_NO_THROW(ov::intel_gpu::ocl::VAContext::type_check(context));
    ASSERT_EQ(static_cast<VADisplay>(context), va_device.get());
    // The shared context is expected to expose the OpenCL context created on top of the VA display
    ASSERT_NE(context.get(), nullptr);
    ASSERT_EQ(context.get_device_name().find(ov::test::utils::DEVICE_GPU), 0);
}

TEST_F(OVRemoteTensorVA_Test, smoke_create_tensor_nv12) {
    ov::Core core;
    ov::intel_gpu::ocl::VAContext context(core, va_device.get());

    auto nv12 = context.create_tensor_nv12(surface_height, surface_width, surface);
    auto& tensor_y = nv12.first;
    auto& tensor_uv = nv12.second;

    OV_ASSERT_NO_THROW(ov::intel_gpu::ocl::VASurfaceTensor::type_check(tensor_y));
    OV_ASSERT_NO_THROW(ov::intel_gpu::ocl::VASurfaceTensor::type_check(tensor_uv));

    ASSERT_EQ(tensor_y.get_element_type(), ov::element::u8);
    ASSERT_EQ(tensor_uv.get_element_type(), ov::element::u8);
    ASSERT_EQ(tensor_y.get_shape(), ov::Shape({1, surface_height, surface_width, 1}));
    ASSERT_EQ(tensor_uv.get_shape(), ov::Shape({1, surface_height / 2, surface_width / 2, 2}));

    // Both planes refer to the same surface, but to different memory objects
    ASSERT_EQ(static_cast<VASurfaceID>(tensor_y), surface);
    ASSERT_EQ(static_cast<VASurfaceID>(tensor_uv), surface);
    ASSERT_EQ(tensor_y.plane(), 0);
    ASSERT_EQ(tensor_uv.plane(), 1);
    ASSERT_NE(tensor_y.get(), nullptr);
    ASSERT_NE(tensor_uv.get(), nullptr);
    ASSERT_NE(tensor_y.get(), tensor_uv.get());
}

TEST_F(OVRemoteTensorVA_Test, smoke_distinct_surfaces_get_distinct_memory) {
    ov::Core core;
    ov::intel_gpu::ocl::VAContext context(core, va_device.get());

    VASurfaceID other_surface = va_device.create_nv12_surface(surface_width, surface_height);
    ASSERT_NE(other_surface, VA_INVALID_SURFACE);
    ASSERT_NE(other_surface, surface);

    {
        auto nv12 = context.create_tensor_nv12(surface_height, surface_width, surface);
        auto other_nv12 = context.create_tensor_nv12(surface_height, surface_width, other_surface);

        ASSERT_EQ(static_cast<VASurfaceID>(other_nv12.first), other_surface);
        ASSERT_NE(nv12.first.get(), other_nv12.first.get());
        ASSERT_NE(nv12.second.get(), other_nv12.second.get());
    }

    va_device.destroy_surface(other_surface);
}

TEST_F(OVRemoteTensorVA_Test, smoke_repeated_import_of_the_same_surface) {
    ov::Core core;
    ov::intel_gpu::ocl::VAContext context(core, va_device.get());

    // Every iteration releases the previously imported memory while the surface stays alive
    const size_t total_run_number = 4;
    for (size_t i = 0; i < total_run_number; i++) {
        auto nv12 = context.create_tensor_nv12(surface_height, surface_width, surface);
        ASSERT_NE(nv12.first.get(), nullptr);
        ASSERT_NE(nv12.second.get(), nullptr);
        ASSERT_EQ(static_cast<VASurfaceID>(nv12.first), surface);
    }
}

TEST_F(OVRemoteTensorVA_Test, smoke_surface_import_is_cached_while_alive) {
    ov::Core core;
    ov::intel_gpu::ocl::VAContext context(core, va_device.get());

    auto first = context.create_tensor_nv12(surface_height, surface_width, surface);
    auto second = context.create_tensor_nv12(surface_height, surface_width, surface);

    ASSERT_EQ(first.first.get(), second.first.get());
    ASSERT_EQ(first.second.get(), second.second.get());

    // Both imports still describe the very same surface
    ASSERT_EQ(static_cast<VASurfaceID>(second.first), static_cast<VASurfaceID>(first.first));
    ASSERT_EQ(second.first.get_shape(), first.first.get_shape());
}

TEST_F(OVRemoteTensorVA_Test, smoke_nv12_surface_inference) {
    auto model = ov::test::utils::make_conv_pool_relu({1, 3, surface_height, surface_width});

    using namespace ov::preprocess;
    auto p = PrePostProcessor(model);
    p.input().tensor().set_element_type(ov::element::u8)
                      .set_color_format(ColorFormat::NV12_TWO_PLANES, {"y", "uv"})
                      .set_memory_type(ov::intel_gpu::memory_type::surface);
    p.input().preprocess().convert_color(ColorFormat::BGR);
    p.input().model().set_layout("NCHW");
    model = p.build();

    auto param_input_y = model->get_parameters().at(0);
    auto param_input_uv = model->get_parameters().at(1);

    ov::Core core;
    ov::intel_gpu::ocl::VAContext context(core, va_device.get());
    auto compiled_model = core.compile_model(model, context);
    auto request = compiled_model.create_infer_request();

    const size_t iteration_count = 4;
    for (size_t i = 0; i < iteration_count; i++) {
        auto nv12 = context.create_tensor_nv12(surface_height, surface_width, surface);
        request.set_tensor(param_input_y, nv12.first);
        request.set_tensor(param_input_uv, nv12.second);

        OV_ASSERT_NO_THROW(request.infer());

        auto output_tensor = request.get_tensor(model->get_results().at(0));
        ASSERT_EQ(output_tensor.get_shape(), model->get_results().at(0)->get_shape());
    }
}

// Regression test for VA surface cache collisions.
TEST_F(OVRemoteTensorVA_Test, smoke_recycled_surface_id_is_reimported) {
    ov::Core core;
    ov::intel_gpu::ocl::VAContext context(core, va_device.get());

    auto parameter = std::make_shared<ov::op::v0::Parameter>(
        ov::element::u8,
        ov::Shape{1, 3, surface_height, surface_width});
    auto model = std::make_shared<ov::Model>(ov::OutputVector{parameter}, ov::ParameterVector{parameter});

    using namespace ov::preprocess;
    auto preprocessor = PrePostProcessor(model);
    preprocessor.input().tensor().set_element_type(ov::element::u8)
                                .set_color_format(ColorFormat::NV12_TWO_PLANES, {"y", "uv"})
                                .set_memory_type(ov::intel_gpu::memory_type::surface);
    preprocessor.input().preprocess().convert_color(ColorFormat::BGR);
    preprocessor.input().model().set_layout("NCHW");
    model = preprocessor.build();

    auto compiled_model = core.compile_model(model, context);
    auto request = compiled_model.create_infer_request();
    const auto infer_mean = [&](VASurfaceID input_surface) {
        auto nv12 = context.create_tensor_nv12(surface_height, surface_width, input_surface);
        request.set_input_tensor(0, nv12.first);
        request.set_input_tensor(1, nv12.second);
        request.infer();

        auto output = request.get_output_tensor();
        const auto* output_data = output.data<const uint8_t>();
        return std::accumulate(output_data, output_data + output.get_size(), 0.0) / output.get_size();
    };

    const VASurfaceID original_id = surface;
    ASSERT_TRUE(va_device.fill_nv12_surface(surface, surface_width, surface_height, 16, 128, 128));
    const auto original_mean = infer_mean(surface);

    VASurfaceID replacement_surface = va_device.create_nv12_surface(surface_width, surface_height);
    ASSERT_NE(replacement_surface, VA_INVALID_SURFACE);
    ASSERT_NE(replacement_surface, original_id);
    ASSERT_TRUE(va_device.fill_nv12_surface(replacement_surface, surface_width, surface_height, 128, 128, 128));
    infer_mean(replacement_surface);

    // Keep the request and compiled model alive, but process a replacement input before destroying the old surface.
    // This releases the request and plugin references to the old import, so its weak cache entry must expire.
    bool is_recycled = false;
    const size_t max_attempts = 8;
    for (size_t attempt = 0; attempt < max_attempts && !is_recycled; attempt++) {
        va_device.destroy_surface(surface);
        surface = va_device.create_nv12_surface(surface_width, surface_height);
        ASSERT_NE(surface, VA_INVALID_SURFACE);
        is_recycled = surface == original_id;
    }

    if (!is_recycled) {
        va_device.destroy_surface(replacement_surface);
        GTEST_SKIP() << "VA-API did not reassign the released surface id, cache aliasing can't be exercised";
    }

    ASSERT_TRUE(va_device.fill_nv12_surface(surface, surface_width, surface_height, 235, 128, 128));
    const auto reimported_mean = infer_mean(surface);
    va_device.destroy_surface(replacement_surface);

    ASSERT_GT(reimported_mean, original_mean + 200.0);
}

#endif  // (OV_GPU_WITH_OCL_RT || OV_GPU_WITH_ZE_RT) && !_WIN32 && ENABLE_LIBVA && ENABLE_LIBVA_DRM
