// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "model_serializer.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"

namespace {

std::string serializeIOInfoOfRank(const size_t rank) {
    auto weights = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, std::vector<float>{1.0f});
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape(rank, 1));
    auto add = std::make_shared<ov::op::v1::Add>(input, weights);
    const auto model = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{input}, "rank_model");

    return ::intel_npu::compiler_utils::serializeIOInfo(model, true);
}

}  // namespace

TEST(ModelSerializerTest, SerializeIOInfoUsesNtdchwForSixDimensionalTensors) {
    const std::string ioInfo = serializeIOInfoOfRank(6);

    EXPECT_NE(ioInfo.find("--inputs_layouts=\"0:NTDCHW\""), std::string::npos);
    EXPECT_NE(ioInfo.find("--outputs_layouts=\"0:NTDCHW\""), std::string::npos);
}

TEST(ModelSerializerTest, SerializeIOInfoUsesBlockedAboveSixDimensions) {
    const std::string ioInfo = serializeIOInfoOfRank(7);

    EXPECT_NE(ioInfo.find("--inputs_layouts=\"0:BLOCKED\""), std::string::npos);
    EXPECT_NE(ioInfo.find("--outputs_layouts=\"0:BLOCKED\""), std::string::npos);
}
