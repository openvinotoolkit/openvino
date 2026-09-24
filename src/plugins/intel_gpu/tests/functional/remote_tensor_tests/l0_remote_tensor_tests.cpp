// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef OV_GPU_WITH_ZE_RT

#include <algorithm>
#include <memory>

#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/intel_gpu/properties.hpp"
#include "openvino/runtime/intel_gpu/remote_properties.hpp"
#include "openvino/runtime/intel_gpu/ze/ze.hpp"
#include "openvino/runtime/remote_context.hpp"
#include "openvino/util/memory.hpp"
#include "openvino/util/mmap_object.hpp"

#include "common_test_utils/test_common.hpp"

namespace {
std::shared_ptr<ov::Model> make_l0_copy_model(const ov::Shape& shape) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
    auto result = std::make_shared<ov::op::v0::Result>(input);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input});
}
}  // namespace

TEST(GpuRemoteTensorL0, smoke_allocAlignedCPUMemory) {
    ov::Core core;
    const std::string target_device = "GPU";
    const auto cacheline_size = core.get_property(target_device, ov::intel_gpu::cacheline_size);
    ASSERT_GT(cacheline_size, 0u);
    const auto page_size = static_cast<size_t>(ov::util::get_system_page_size());
    ASSERT_GT(page_size, 0u);
    ASSERT_EQ(page_size % cacheline_size, 0u);

    const size_t float_size = sizeof(float);
    const ov::Shape shape{page_size / float_size};
    const size_t element_count = ov::shape_size(shape);
    const size_t byte_size = element_count * sizeof(float);

    void* input_ptr = ov::util::aligned_alloc(byte_size, page_size);
    void* output_ptr = ov::util::aligned_alloc(byte_size, page_size);

    std::fill_n(static_cast<float*>(input_ptr), element_count, 2.0f);
    std::fill_n(static_cast<float*>(output_ptr), element_count, 0.0f);

    {
        auto context = core.get_default_context(target_device).as<ov::intel_gpu::ze::ZeContext>();
        auto remote_input_tensor =
            context.create_tensor(ov::element::f32,
                                  shape,
                                  ov::intel_gpu::VirtualAddressMemory(input_ptr,
                                                                      static_cast<int64_t>(byte_size),
                                                                      ov::intel_gpu::AccessMode::READ));
        auto remote_output_tensor =
            context.create_tensor(ov::element::f32,
                                  shape,
                                  ov::intel_gpu::VirtualAddressMemory(output_ptr, static_cast<int64_t>(byte_size)));
        ASSERT_TRUE(remote_input_tensor.is<ov::intel_gpu::ze::ZeRemoteTensor>());
        ASSERT_TRUE(remote_output_tensor.is<ov::intel_gpu::ze::ZeRemoteTensor>());
        ASSERT_NE(remote_input_tensor.get(), nullptr);
        ASSERT_NE(remote_output_tensor.get(), nullptr);

        auto model = make_l0_copy_model(shape);
        auto compiled = core.compile_model(model, context);
        auto infer_req = compiled.create_infer_request();
        infer_req.set_tensor(compiled.input(), remote_input_tensor);
        infer_req.set_tensor(compiled.output(), remote_output_tensor);
        infer_req.infer();

        for (size_t i = 0; i < element_count; ++i) {
            EXPECT_FLOAT_EQ(static_cast<float*>(output_ptr)[i], 2.0f) << "Mismatch at index " << i;
        }
    }

    ov::util::aligned_free(input_ptr);
    ov::util::aligned_free(output_ptr);
}

#endif  // OV_GPU_WITH_ZE_RT
