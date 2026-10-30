// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifdef OV_GPU_WITH_OCL_RT

#    include <gtest/gtest.h>

#    include <algorithm>

#    include "openvino/op/constant.hpp"
#    include "openvino/op/convert.hpp"
#    include "openvino/op/multiply.hpp"
#    include "openvino/op/parameter.hpp"
#    include "openvino/runtime/core.hpp"
#    include "openvino/runtime/intel_gpu/ocl/ocl.hpp"
#    include "openvino/runtime/properties.hpp"

namespace {
class DynamicOutputRebindTest : public ::testing::TestWithParam<bool> {};

TEST_P(DynamicOutputRebindTest, SharedProducerWithNewOutputBindings) {
    using namespace ov;
    auto input = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{1, 1, -1, 8});
    auto product = std::make_shared<op::v1::Multiply>(input, op::v0::Constant::create(element::f16, Shape{}, {2}));
    auto converted = std::make_shared<op::v0::Convert>(product, element::f32);
    auto model = std::make_shared<Model>(OutputVector{converted, product}, ParameterVector{input});
    Core core;
    auto compiled = core.compile_model(model, "GPU", hint::inference_precision(element::dynamic));
    if (!compiled.get_context().is<intel_gpu::ocl::ClContext>())
        GTEST_SKIP() << "This test uses OpenCL USM output tensors";
    auto context = compiled.get_context().as<intel_gpu::ocl::ClContext>();
    auto request = compiled.create_infer_request();
    const Shape capacity{1, 1, 3, 8};
    auto result_storage = context.create_usm_device_tensor(element::f32, capacity);
    auto product_storage = context.create_usm_device_tensor(element::f16, capacity);

    for (size_t tokens : {3u, 1u, 2u}) {
        SCOPED_TRACE(tokens);
        const Shape shape{1, 1, tokens, 8};
        Tensor source(element::f16, shape);
        std::fill_n(source.data<float16>(), source.get_size(), float16(0.25f));
        // Exercise both new allocations and changed capacity at the same pointer.
        auto result = GetParam() ? context.create_usm_device_tensor(element::f32, capacity) : context.create_tensor(element::f32, shape, result_storage.get());
        auto raw = GetParam() ? context.create_usm_device_tensor(element::f16, capacity) : context.create_tensor(element::f16, shape, product_storage.get());
        result.set_shape(shape);
        raw.set_shape(shape);
        request.set_input_tensor(source);
        request.set_output_tensor(0, result);
        request.set_output_tensor(1, raw);
        ASSERT_NO_THROW(request.infer());
        Tensor result_host(element::f32, shape);
        Tensor product_host(element::f16, shape);
        result.copy_to(result_host);
        raw.copy_to(product_host);
        for (size_t i = 0; i < result_host.get_size(); ++i) {
            EXPECT_EQ(result_host.data<float>()[i], 0.5f);
            EXPECT_EQ(float(product_host.data<float16>()[i]), 0.5f);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(smoke, DynamicOutputRebindTest, ::testing::Values(false, true));
}  // namespace

#endif  // OV_GPU_WITH_OCL_RT
