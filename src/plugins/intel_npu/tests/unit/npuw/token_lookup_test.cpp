// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <sstream>
#include <vector>

#include "openvino/op/ops.hpp"
#include "openvino/openvino.hpp"

namespace {

std::shared_ptr<ov::Model> make_lookup(bool contextual = false) {
    auto ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{-1, -1});
    ids->output(0).set_names({"input"});
    std::vector<float> values(32 * 4);
    std::iota(values.begin(), values.end(), 0.0f);
    auto vocabulary = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{32, 4}, values);
    auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
    ov::Output<ov::Node> result = std::make_shared<ov::op::v8::Gather>(vocabulary, ids, axis);
    if (contextual) {
        result = std::make_shared<ov::op::v1::Add>(result,
                                                   ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {1.0f}));
    } else {
        result = std::make_shared<ov::op::v1::Multiply>(
            result,
            ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {0.5f}));
    }
    result.get_tensor().set_names({"inputs_embeds"});
    return std::make_shared<ov::Model>(ov::OutputVector{result}, ov::ParameterVector{ids});
}

ov::AnyMap properties() {
    return {{"NPU_USE_NPUW", "YES"},
            {"NPUW_TOKEN_LOOKUP", "YES"},
            {"NPUW_DEVICES", "CPU"},
            {"NPUW_ONLINE_PIPELINE", "NONE"}};
}

void check_lengths(ov::CompiledModel& compiled) {
    auto request = compiled.create_infer_request();
    for (const std::size_t length : {1u, 63u, 64u, 65u, 129u, 1u}) {
        ov::Tensor ids{ov::element::i64, {1, length}};
        for (std::size_t i = 0; i < length; ++i) {
            ids.data<int64_t>()[i] = static_cast<int64_t>((i * 7) % 32);
        }
        request.set_input_tensor(ids);
        ASSERT_NO_THROW(request.infer());
        const auto output = request.get_output_tensor();
        ASSERT_EQ(output.get_shape(), (ov::Shape{1, length, 4}));
        ASSERT_EQ(output.get_element_type(), ov::element::f32);
        for (std::size_t i = 0; i < length; ++i) {
            for (std::size_t j = 0; j < 4; ++j) {
                EXPECT_FLOAT_EQ(output.data<const float>()[i * 4 + j],
                                static_cast<float>(ids.data<int64_t>()[i] * 4 + j) * 0.5f);
            }
        }
    }
}

TEST(NPUWTokenLookupTest, HandlesVariableLengthAndRepeatedRequests) {
    ov::Core core;
    auto compiled = core.compile_model(make_lookup(), "NPU", properties());
    EXPECT_TRUE(compiled.get_property("NPUW_TOKEN_LOOKUP").as<bool>());
    check_lengths(compiled);

    auto request = compiled.create_infer_request();
    ov::Tensor invalid{ov::element::i64, {1, 1}};
    for (int64_t token : {-1, 32}) {
        invalid.data<int64_t>()[0] = token;
        request.set_input_tensor(invalid);
        EXPECT_THROW(request.infer(), ov::Exception);
    }
    EXPECT_THROW(request.set_input_tensor(ov::Tensor(ov::element::i32, {1, 1})), ov::Exception);
    request.set_input_tensor(ov::Tensor(ov::element::i64, {2, 1}));
    EXPECT_THROW(request.infer(), ov::Exception);
    request.set_input_tensor(ov::Tensor(ov::element::i64, {1, 0}));
    EXPECT_THROW(request.infer(), ov::Exception);

    std::vector<int64_t> interleaved{1, -1, 2, -1, 3, -1};
    ov::Tensor strided{ov::element::i64, {1, 3}, interleaved.data(), ov::Strides{48, 16}};
    request.set_input_tensor(strided);
    ASSERT_NO_THROW(request.infer());
    const auto output = request.get_output_tensor();
    ASSERT_EQ(output.get_shape(), (ov::Shape{1, 3, 4}));
    EXPECT_FLOAT_EQ(output.data<const float>()[0], 2.0f);
    EXPECT_FLOAT_EQ(output.data<const float>()[4], 4.0f);
    EXPECT_FLOAT_EQ(output.data<const float>()[8], 6.0f);

    std::vector<float> strided_output(24, -1.0f);
    ov::Tensor bound_output{ov::element::f32, {1, 3, 4}, strided_output.data(), ov::Strides{96, 32, 8}};
    request.set_output_tensor(bound_output);
    ASSERT_NO_THROW(request.infer());
    EXPECT_FLOAT_EQ(strided_output[0], 2.0f);
    EXPECT_FLOAT_EQ(strided_output[8], 4.0f);
    EXPECT_FLOAT_EQ(strided_output[16], 6.0f);
    EXPECT_FLOAT_EQ(strided_output[1], -1.0f);
}

TEST(NPUWTokenLookupTest, RejectsContextualGraphs) {
    ov::Core core;
    EXPECT_THROW(core.compile_model(make_lookup(true), "NPU", properties()), ov::Exception);
}

TEST(NPUWTokenLookupTest, ExportsAndImportsDynamicInterface) {
    ov::Core core;
    auto config = properties();
    config[ov::cache_mode.name()] = ov::CacheMode::OPTIMIZE_SPEED;
    auto compiled = core.compile_model(make_lookup(), "NPU", config);
    std::stringstream blob;
    compiled.export_model(blob);
    auto imported = core.import_model(blob, "NPU", properties());
    EXPECT_TRUE(imported.get_property("NPUW_TOKEN_LOOKUP").as<bool>());
    check_lengths(imported);
}

}  // namespace
