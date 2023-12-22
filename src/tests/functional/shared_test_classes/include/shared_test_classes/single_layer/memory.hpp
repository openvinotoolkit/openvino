// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "ngraph/opsets/opset6.hpp"
#include "ngraph/opsets/opset3.hpp"

#include "shared_test_classes/base/layer_test_utils.hpp"

namespace LayerTestsDefinitions {

using MemoryTestParams = std::tuple<
        ngraph::helpers::MemoryTransformation,   // Apply Memory transformation
        int64_t,                            // iterationCount
        InferenceEngine::SizeVector,        // inputShape
        InferenceEngine::Precision,         // netPrecision
        std::string                         // targetDevice
>;

class MemoryTest : public testing::WithParamInterface<MemoryTestParams>, virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MemoryTestParams> &obj);
    void Run() override;

protected:
    std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>> CalculateRefs() override;
    void SetUp() override;
    void Infer() override;
    virtual std::shared_ptr<ov::op::util::ReadValueBase> CreateReadValueOp(
            const ov::Output<ov::Node>& value, const std::shared_ptr<ov::op::util::Variable>& variable) const {
        return std::make_shared<ov::op::v6::ReadValue>(value, variable);
    }
    virtual std::shared_ptr<ov::op::util::AssignBase> CreateAssignOp(
            const ov::Output<ov::Node>& value, const std::shared_ptr<ov::op::util::Variable>& variable) const {
        return std::make_shared<ov::op::v6::Assign>(value, variable);
    }

    virtual void CreateCommonFunc();

    ov::element::Type ngPrc;
    ov::Shape inputShape;

private:
    void CreateTIFunc();
    void ApplyLowLatency();

    InferenceEngine::Precision netPrecision;
    ngraph::EvaluationContext eval_context;
    ngraph::helpers::MemoryTransformation transformation;

    int64_t iteration_count;
};

class MemoryTestV3 : public MemoryTest {
protected:
    std::shared_ptr<ov::op::util::ReadValueBase> CreateReadValueOp(
            const ov::Output<ov::Node>& value, const std::shared_ptr<ov::op::util::Variable>& variable) const override {
        return std::make_shared<ov::op::v3::ReadValue>(value, variable->get_info().variable_id);
    }

    std::shared_ptr<ov::op::util::AssignBase> CreateAssignOp(
            const ov::Output<ov::Node>& value, const std::shared_ptr<ov::op::util::Variable>& variable) const override {
        return std::make_shared<ov::op::v3::Assign>(value, variable->get_info().variable_id);
    }
};

}  // namespace LayerTestsDefinitions
