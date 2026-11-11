// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "model_serializer.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/result.hpp"

namespace {

std::shared_ptr<ov::Model> make_model(const ov::Shape& shape) {
    const auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
    input->set_friendly_name("input");
    const auto relu = std::make_shared<ov::op::v0::Relu>(input);
    return std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(relu)},
                                       ov::ParameterVector{input});
}

}  // namespace

TEST(NpuModelSerializerTest, supportedRanksUseLegacyLayoutStrings) {
    EXPECT_NE(intel_npu::compiler_utils::serializeIOInfo(make_model({1, 3, 8, 8}), true).find("NCHW"),
              std::string::npos);
    EXPECT_NE(intel_npu::compiler_utils::serializeIOInfo(make_model({1, 3, 1, 8, 8}), true).find("NCDHW"),
              std::string::npos);
}

TEST(NpuModelSerializerTest, rankAboveFiveIsRejectedWithAClearMessage) {
    try {
        intel_npu::compiler_utils::serializeIOInfo(make_model({1, 3, 1, 1, 8, 8}), true);
        FAIL() << "serializeIOInfo is expected to throw for a rank-6 tensor";
    } catch (const ov::Exception& exc) {
        const std::string message = exc.what();
        EXPECT_NE(message.find("at most 5 dimensions"), std::string::npos) << message;
        EXPECT_NE(message.find("rank 6"), std::string::npos) << message;
        EXPECT_NE(message.find("input 'input'"), std::string::npos) << message;
    }
}
