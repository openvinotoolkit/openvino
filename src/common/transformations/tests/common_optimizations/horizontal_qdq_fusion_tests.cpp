// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/common_optimizations/horizontal_qdq_fusion.hpp"

using namespace ov;
using namespace testing;

namespace {

std::shared_ptr<ov::Node> make_fq_converts(const std::shared_ptr<ov::op::v0::Parameter>& data) {
    auto fq = std::make_shared<ov::op::v0::FakeQuantize>(data,
                                                         ov::op::v0::Constant::create(element::f32, Shape{}, {-128.f}),
                                                         ov::op::v0::Constant::create(element::f32, Shape{}, {127.f}),
                                                         ov::op::v0::Constant::create(element::f32, Shape{}, {-128.f}),
                                                         ov::op::v0::Constant::create(element::f32, Shape{}, {127.f}),
                                                         256);
    auto conv1 = std::make_shared<ov::op::v0::Convert>(fq, element::i8);
    return std::make_shared<ov::op::v0::Convert>(conv1, element::f32);
}

std::shared_ptr<ov::Node> make_dequantize(const std::shared_ptr<ov::Node>& data,
                                          float scale,
                                          std::optional<float> zp = std::nullopt) {
    auto input = data;
    if (zp.has_value()) {
        input =
            std::make_shared<ov::op::v1::Subtract>(data, ov::op::v0::Constant::create(element::f32, Shape{}, {*zp}));
    }
    return std::make_shared<ov::op::v1::Multiply>(input, ov::op::v0::Constant::create(element::f32, Shape{}, {scale}));
}

}  // namespace

struct DQParams {
    float scale_value;
    std::optional<float> shift_value;

    bool operator==(const DQParams& other) const {
        return scale_value == other.scale_value && shift_value == other.shift_value;
    }
};

struct HorizontalQDQFusionTestParam {
    std::string name;
    std::vector<DQParams> branches;
};

class HorizontalQDQFusionTest : public TransformationTestsF,
                                public testing::WithParamInterface<HorizontalQDQFusionTestParam> {
public:
    void SetUp() override {
        TransformationTestsF::SetUp();
    }

    static std::string getTestCaseName(const testing::TestParamInfo<HorizontalQDQFusionTestParam>& info) {
        return info.param.name;
    }
};

TEST_P(HorizontalQDQFusionTest, CompareFunctions) {
    const auto& params = GetParam();

    // Build the model: one DQ branch per entry in params.branches.
    {
        auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 4, 4});
        auto quantize = make_fq_converts(data);
        ResultVector results;
        for (const auto& p : params.branches) {
            auto dq = make_dequantize(quantize, p.scale_value, p.shift_value);
            results.push_back(std::make_shared<ov::op::v0::Result>(dq));
        }
        model = std::make_shared<ov::Model>(results, ParameterVector{data});
        manager.register_pass<ov::pass::HorizontalQDQFusion>();
    }

    {
        auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 4, 4});
        auto quantize = make_fq_converts(data);

        // first occurrence of each unique DQParams -> its DQ node.
        std::vector<std::pair<DQParams, std::shared_ptr<ov::Node>>> dq_cache;
        ResultVector results;
        for (const auto& p : params.branches) {
            auto it = std::find_if(dq_cache.begin(), dq_cache.end(), [&p](const auto& e) {
                return e.first == p;
            });
            std::shared_ptr<ov::Node> dq;
            if (it == dq_cache.end()) {
                dq = make_dequantize(quantize, p.scale_value, p.shift_value);
                dq_cache.emplace_back(p, dq);
            } else {
                dq = it->second;
            }
            results.push_back(std::make_shared<ov::op::v0::Result>(dq));
        }
        model_ref = std::make_shared<ov::Model>(results, ParameterVector{data});
    }
}

const std::vector<HorizontalQDQFusionTestParam> test_params{
    HorizontalQDQFusionTestParam{"TwoIdenticalBranches_SubMul", {{0.1f, 5.f}, {0.1f, 5.f}}},
    HorizontalQDQFusionTestParam{"TwoIdenticalBranches_MulOnly", {{0.5f, std::nullopt}, {0.5f, std::nullopt}}},
    HorizontalQDQFusionTestParam{"ThreeBranches_TwoIdentical", {{0.1f, 5.f}, {0.1f, 5.f}, {0.2f, 10.f}}},
    HorizontalQDQFusionTestParam{"FourBranches_TwoIdentical", {{0.1f, 5.f}, {0.1f, 5.f}, {0.2f, 10.f}, {0.2f, 10.f}}},
    HorizontalQDQFusionTestParam{"DifferentScales_NoFusion", {{0.1f, 5.f}, {0.2f, 5.f}}},
    HorizontalQDQFusionTestParam{"DifferentZeroPoints_NoFusion", {{0.1f, 5.f}, {0.1f, 7.f}}},
    HorizontalQDQFusionTestParam{"MismatchedStructure_NoFusion", {{0.1f, 5.f}, {0.1f, std::nullopt}}},
    HorizontalQDQFusionTestParam{"SingleConsumer_NoFusion", {{0.1f, 5.f}}}};

INSTANTIATE_TEST_SUITE_P(HorizontalQDQFusion,
                         HorizontalQDQFusionTest,
                         testing::ValuesIn(test_params),
                         HorizontalQDQFusionTest::getTestCaseName);
