// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "string_tensor_unpack.hpp"
#include "utils/cpu_test_utils.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/precision_support.h"

using namespace CPUTestUtils;

namespace ov {
namespace test {
namespace StringTensorUnpack {
std::string StringTensorUnpackLayerCPUTest::getTestCaseName(testing::TestParamInfo<StringTensorUnpackLayerCPUTestParamsSet> obj) {
        StringTensorUnpackLayerTestParams basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = obj.param;
        std::string td;
        StringTensorUnpackSpecificParams StringTensorUnpackPar;
        std::tie(StringTensorUnpackPar, td) = basicParamsSet;

        InputShape inputShape;
        std::tie(inputShape) = StringTensorUnpackPar;
        std::ostringstream result;

        result << ov::test::utils::partialShape2str({ inputShape.first }) << "_";
        result << "TS=";
        result << "(";
        for (const auto& targetShape : inputShape.second) {
            result << ov::test::utils::vec2str(targetShape) << "_";
        }
        result << ")";
        result << CPUTestsBase::getTestCaseName(cpuParams);

        return result.str();
    }

void StringTensorUnpackLayerCPUTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
        inputs.clear();
        const auto& funcInputs = function->inputs();

        ov::Tensor data_tensor;
        const auto& dataShape = targetInputStaticShapes.front();
        ov::test::utils::InputGenerateData in_data;
        in_data.start_from = 0;
        in_data.range = 10;
        data_tensor = ov::test::utils::create_and_fill_tensor(ov::element::string, dataShape, in_data);
        inputs.insert({ funcInputs[0].get_node_shared_ptr(), data_tensor });
    }

void StringTensorUnpackLayerCPUTest::SetUp() {
        StringTensorUnpackLayerTestParams basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = this->GetParam();
        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

        StringTensorUnpackSpecificParams StringTensorUnpackParams;
        std::tie(StringTensorUnpackParams, targetDevice) = basicParamsSet;

        InputShape dataInputShape;
        std::tie(dataInputShape) = StringTensorUnpackParams;

        init_input_shapes({dataInputShape});
        auto dataParameter = std::make_shared<ov::op::v0::Parameter>(ov::element::string, inputDynamicShapes[0]);
        auto StringTensorUnpack = std::make_shared<ov::op::v15::StringTensorUnpack>(dataParameter);

        ov::ParameterVector params{ dataParameter };
        function = makeNgraphFunction(ov::element::string, params, StringTensorUnpack, "StringTensorUnpack");
}

TEST_P(StringTensorUnpackLayerCPUTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "StringTensorUnpack");
}

const std::vector<StringTensorUnpackSpecificParams> StringTensorUnpackParamsVector = {
    StringTensorUnpackSpecificParams {
        InputShape{{}, {{3}}}
    },
    StringTensorUnpackSpecificParams {
        InputShape{{}, {{1, 1, 13}}}
    },
    StringTensorUnpackSpecificParams {
        InputShape{{}, {{100, 10}}}
    },
    StringTensorUnpackSpecificParams {
        InputShape{{}, {{3, 8, 12, 8, 4}}}
    },
    StringTensorUnpackSpecificParams {
        InputShape{{-1}, {{3}}}
    },
    StringTensorUnpackSpecificParams {
        InputShape{{-1, -1, -1}, {{1, 1, 13}}}
    },
    StringTensorUnpackSpecificParams {
        InputShape{{-1, 10}, {{100, 10}}}
    },
    StringTensorUnpackSpecificParams {
        InputShape{{-1, {7, 9}, 12, {1, 20}, -1}, {{3, 8, 12, 8, 4}, {21, 7, 12, 1, 3}, {4, 9, 12, 20, 1}}}
    },
    StringTensorUnpackSpecificParams {
        InputShape{{3, -1, {3, 8}}, {{3, 1, 3}, {3, 2, 8}}}
    },
};

}  // namespace StringTensorUnpack
}  // namespace test
}  // namespace ov
