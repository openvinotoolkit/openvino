// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rms_norm.hpp"

#include "gtest/gtest.h"
#include "openvino/core/shape.hpp"
#include "openvino/op/constant.hpp"
#include "ov_ops/rms.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"
#include "openvino/pass/manager.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

std::string RMSNormLayerCPUTest::getTestCaseName(const testing::TestParamInfo<RMSNormCPUTestParams>& obj) {
    CPUSpecificParams cpuParams;
    ElementType inType;
    std::vector<InputShape> inputShapes;
    std::string targetDevice;
    std::tie(inType, inputShapes, targetDevice, cpuParams) = obj.param;

    std::ostringstream result;
    result << "netPRC=" << inType << "_";
    result << "IS=";
    for (const auto& inputShape : inputShapes) {
        result << ov::test::utils::partialShape2str({inputShape.first}) << "_";
    }
    result << "TS=";
    for (const auto& shapes : inputShapes) {
        for (const auto& shape : shapes.second) {
            result << ov::test::utils::vec2str(shape);
            result << "_";
        }
    }
    result << "trgDev=" << targetDevice;
    result << CPUTestsBase::getTestCaseName(cpuParams);

    return result.str();
}

template <typename IT, typename T>
void strided_iota(IT first, size_t n, T value, T stride) {
    for (size_t i = 0; i < n; i++) {
        *first++ = value;
        value += stride;
    }
}

void RMSNormLayerCPUTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    inputs.clear();
    auto create_input = [this](std::shared_ptr<ov::op::v0::Parameter> param, ov::Shape shape, float val) {
        if (param->get_element_type() == element::i32) {
            ov::Tensor t{ov::element::i32, shape};
            auto size = shape[0];
            auto* p = static_cast<int*>(t.data());
            auto start = static_cast<int>(val);
            for (size_t i = 0; i < size; i++) {
                p[i] = (start + i) % size;
            }
            inputs.insert({param, t});
        } else if (param->get_element_type() == element::f32) {
            ov::Tensor t{ov::element::f32, shape};
            strided_iota(static_cast<float*>(t.data()), t.get_size(), val, 0.1f);
            inputs.insert({param, t});
        } else if (param->get_element_type() == element::f16) {
            ov::Tensor t{ov::element::f16, shape};
            strided_iota(static_cast<ov::float16*>(t.data()), t.get_size(), val, 0.1f);
            inputs.insert({param, t});
        } else {
            OPENVINO_ASSERT(param->get_element_type() == element::bf16);
            ov::Tensor t{ov::element::bf16, shape};
            strided_iota(static_cast<ov::bfloat16*>(t.data()), t.get_size(), val, 0.1f);
            inputs.insert({param, t});
        }
    };
    create_input(function->get_parameters()[0], targetInputStaticShapes[0], 1.0f);
    create_input(function->get_parameters()[1], targetInputStaticShapes[1], 0.0f);
    for (size_t i = 0; i < targetInputStaticShapes[1].size() - 1; i++) {
        if (targetInputStaticShapes[1][i] != 1) {
            // decomposed rms expected
            m_rms_decomposed = true;
            break;
        }
    }
}

void RMSNormLayerCPUTest::SetUp() {
    ElementType inType;
    CPUSpecificParams cpuParams;
    std::vector<InputShape> inputShapes;
    std::tie(inType, inputShapes, targetDevice, cpuParams) = this->GetParam();

    std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
    if (selectedType.empty()) {
        selectedType = getPrimitiveType();
    }

    rel_threshold = 0.001f;
    if (inType == ElementType::bf16) {
        rel_threshold = 2e-2f;
    }
    selectedType = makeSelectedTypeStr(selectedType, inType);
    init_input_shapes(inputShapes);
    ov::ParameterVector inputParams;
    // data, scale
    auto data = std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes[0]);
    inputParams.push_back(data);
    auto scale = std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes[1]);
    inputParams.push_back(scale);
    auto rms = std::make_shared<ov::op::internal::RMS>(data, scale, 0.1f);
    rms->set_friendly_name("rms");
    function = makeNgraphFunction(inType, inputParams, rms, "rms");
}

TEST_P(RMSNormLayerCPUTest, CompareWithRefs) {
    run();
    CheckNumberOfNodesWithType(compiledModel, "RMS", m_rms_decomposed ? 0 : 1);
}

}  // namespace test
}  // namespace ov
