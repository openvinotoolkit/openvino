// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/paged_attention.hpp"

#include "base_reference_test.hpp"
#include "gtest/gtest.h"
#include "openvino/op/parameter.hpp"
#include "openvino/opsets/opset13.hpp"
#include "openvino/pass/manager.hpp"

namespace {
using PagedAttentionParams =
    std::tuple<ov::element::Type, std::vector<ov::Shape>, int32_t, int32_t, int32_t, std::string>;

class ReferencePagedAttention : public testing::TestWithParam<PagedAttentionParams>,
                                public reference_tests::CommonReferenceTest {
public:
    void SetUp() override {
        const auto& params = GetParam();
        const auto& inType = std::get<0>(params);
        const auto& inputShapes = std::get<1>(params);
        scale = std::get<2>(params);
        sliding_window = std::get<3>(params);
        max_context_len = std::get<4>(params);
        targetDevice = std::get<5>(params);

        function = CreateFunction(params);
        inputData = GenerateInputData(inputShapes, inType);
        refOutData = GenerateRefData(inputShapes, inType);
    }

    static std::string getTestCaseName(const testing::TestParamInfo<PagedAttentionParams>& obj) {
        std::ostringstream name;
        const auto& inType = std::get<0>(obj.param);
        const auto& inputShapes = std::get<1>(obj.param);
        name << "netPRC=" << inType << "_";
        name << "IS=";
        for (const auto& shape : inputShapes) {
            name << shape << "_";
        }
        name << "scale=" << std::get<2>(obj.param) << "_";
        name << "sliding_window=" << std::get<3>(obj.param) << "_";
        name << "max_context_len=" << std::get<4>(obj.param) << "_";
        name << "trgDev=" << std::get<5>(obj.param);
        return name.str();
    }

private:
    static std::shared_ptr<ov::Model> CreateFunction(const PagedAttentionParams& params) {
        const auto& inType = std::get<0>(params);
        const auto& inputShapes = std::get<1>(params);
        int32_t scale = std::get<2>(params);
        int32_t sliding_window = std::get<3>(params);
        int32_t max_context_len = std::get<4>(params);

        ov::ParameterVector inputParams;
        inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inType, inputShapes[0]));  // query
        inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inType, inputShapes[1]));  // key
        inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inType, inputShapes[2]));  // value
        inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inType, inputShapes[3]));  // key_cache
        inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inType, inputShapes[4]));  // value_cache

        ov::OutputVector inputs;
        for (auto& input : inputParams) {
            inputs.push_back(input);
        }
        auto paged_attn = std::make_shared<ov::opset13::PagedAttention>(inputs, scale, sliding_window, max_context_len);
        return std::make_shared<ov::Model>(paged_attn->outputs(), inputParams);
    }

    std::vector<reference_tests::Tensor> GenerateInputData(const std::vector<ov::Shape>& inputShapes,
                                                           const ov::element::Type& inType) {
        std::vector<reference_tests::Tensor> inputData;
        for (const auto& shape : inputShapes) {
            inputData.emplace_back(shape, inType, std::vector<float>(shape_size(shape), 1.0f));
        }
        return inputData;
    }

    std::vector<reference_tests::Tensor> GenerateRefData(const std::vector<ov::Shape>& inputShapes,
                                                         const ov::element::Type& inType) {
        // Generate reference data based on the expected output
        // This is a placeholder implementation
        std::vector<reference_tests::Tensor> refData;
        for (const auto& shape : inputShapes) {
            refData.emplace_back(shape, inType, std::vector<float>(shape_size(shape), 1.0f));
        }
        return refData;
    }

    int32_t scale;
    int32_t sliding_window;
    int32_t max_context_len;
    std::string targetDevice;
};

TEST_P(ReferencePagedAttention, CompareWithRefs) {
    Exec();
}

}  // namespace