// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <sstream>

#include "openvino/core/type/float16.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/normalize_l2.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/properties.hpp"

TEST(OVDynamicOutputRebindTest, EmptyProducerThenNewDestination) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{1, 1, -1, 8});
    auto factor = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{}, {2});
    auto product = std::make_shared<ov::op::v1::Multiply>(input, factor);
    auto converted = std::make_shared<ov::op::v0::Convert>(product, ov::element::f32);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{converted, product}, ov::ParameterVector{input});
    ov::Core core;
    const auto precision = ov::hint::inference_precision(ov::element::dynamic);
    auto compiled = core.compile_model(model, "GPU", precision);
    std::stringstream blob;
    compiled.export_model(blob);
    auto imported = core.import_model(blob, "GPU", precision);

    for (auto executable : {compiled, imported}) {
        auto request = executable.create_infer_request();
        for (size_t tokens : {0u, 3u, 0u, 1u, 5u}) {
            SCOPED_TRACE(tokens);
            ov::Shape shape{1, 1, tokens, 8};
            ov::Tensor source(ov::element::f16, shape);
            for (size_t i = 0; i < source.get_size(); ++i)
                source.data<ov::float16>()[i] = ov::float16(static_cast<float>(i) - 12.f);
            request.set_input_tensor(source);
            ov::Tensor destination(ov::element::f32, shape);
            // After an empty inference, the producer has no output allocation.
            // Rebinding its Result must wait for the next inference to allocate it.
            ASSERT_NO_THROW(request.set_output_tensor(0, destination));
            request.set_output_tensor(1, ov::Tensor(ov::element::f16, shape));
            ASSERT_NO_THROW(request.infer());
            EXPECT_EQ(request.get_output_tensor(0).get_shape(), shape);
            for (size_t i = 0; i < destination.get_size(); ++i)
                EXPECT_EQ(destination.data<float>()[i], 2.f * (static_cast<float>(i) - 12.f));
        }
    }
}

TEST(OVDynamicOutputRebindTest, SameInputAndShapeWithNewDestination) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 1, -1, 8});
    std::vector<float> weights(64, 0.f);
    for (size_t i = 0; i < 8; ++i)
        weights[i * 8 + i] = static_cast<float>(i + 1);
    auto matrix = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{8, 8}, weights);
    auto product = std::make_shared<ov::op::v0::MatMul>(input, matrix, false, false);
    auto axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
    auto normalized = std::make_shared<ov::op::v0::NormalizeL2>(product, axes, 1e-6f, ov::op::EpsMode::ADD);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{normalized}, ov::ParameterVector{input});
    ov::Core core;
    const auto precision = ov::hint::inference_precision(ov::element::dynamic);
    auto compiled = core.compile_model(model, "GPU", precision);
    std::stringstream blob;
    compiled.export_model(blob);
    auto imported = core.import_model(blob, "GPU", precision);
    for (auto executable : {compiled, imported}) {
        auto request = executable.create_infer_request();
        const ov::Shape shape{1, 1, 3, 8};
        ov::Tensor source(ov::element::f32, shape);
        for (size_t i = 0; i < source.get_size(); ++i)
            source.data<float>()[i] = static_cast<float>(i + 1);
        request.set_input_tensor(source);
        for (size_t iteration = 0; iteration < 3; ++iteration) {
            ov::Tensor destination(ov::element::f32, shape);
            std::fill_n(destination.data<float>(), destination.get_size(), -99.f);
            request.set_output_tensor(destination);
            request.infer();
            for (size_t row = 0; row < 3; ++row) {
                float squares = 1e-6f;
                for (size_t column = 0; column < 8; ++column) {
                    const float value = source.data<float>()[row * 8 + column] * static_cast<float>(column + 1);
                    squares += value * value;
                }
                for (size_t column = 0; column < 8; ++column) {
                    const float value = source.data<float>()[row * 8 + column] * static_cast<float>(column + 1);
                    EXPECT_NEAR(destination.data<float>()[row * 8 + column], value / std::sqrt(squares), 1e-5f);
                }
            }
        }
    }
}
