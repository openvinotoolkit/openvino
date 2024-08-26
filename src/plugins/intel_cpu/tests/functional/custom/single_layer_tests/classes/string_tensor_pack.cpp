// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "string_tensor_pack.hpp"
#include "utils/cpu_test_utils.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/precision_support.h"

using namespace CPUTestUtils;

namespace ov {
namespace test {
namespace StringTensorPack {
std::string StringTensorPackLayerCPUTest::getTestCaseName(testing::TestParamInfo<StringTensorPackLayerCPUTestParamsSet> obj) {
        StringTensorPackLayerTestParams basicParamsSet;
        CPUSpecificParams cpuParams;
        StringTensorPackSpecificParams StringTensorPackPar;
        std::tie(basicParamsSet, cpuParams) = obj.param;
        std::string td;
        ElementType indicesPrecision;
        std::tie(StringTensorPackPar, indicesPrecision, td) = basicParamsSet;

        InputShape indicesShape;
        InputShape symbolsShape;
        std::tie(indicesShape, symbolsShape) = StringTensorPackPar;
        std::ostringstream result;

        result << ov::test::utils::partialShape2str({ indicesShape.first }) << "_";
        result << "TS=";
        result << "(";
        for (const auto& targetShape : indicesShape.second) {
            result << ov::test::utils::vec2str(targetShape) << "_";
        }
        result << ")";
        result << "_symbolsShape=";
        result << "(";
        for (const auto& targetShape : symbolsShape.second) {
            result << ov::test::utils::vec2str(targetShape) << "_";
        }
        result << ")";
        result << "_indicesPrecision=" << indicesPrecision;
        result << CPUTestsBase::getTestCaseName(cpuParams);

        return result.str();
    }

void StringTensorPackLayerCPUTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        const auto indicesType = funcInputs[0].get_element_type();
        const auto& indicesShape = targetInputStaticShapes[0];
        const auto& symbolsShape = targetInputStaticShapes[2];

        const ov::Tensor beginsTensor = ov::test::utils::create_and_fill_tensor_consistently(indicesType, indicesShape, symbolsShape[0], 0, 3);
        const ov::Tensor endsTensor = ov::test::utils::create_and_fill_tensor_consistently(indicesType, indicesShape, symbolsShape[0], 3, 3);
        inputs.insert({ funcInputs[0].get_node_shared_ptr(), beginsTensor });
        inputs.insert({ funcInputs[1].get_node_shared_ptr(), endsTensor });

        ov::Tensor symbolsTensor;
        ov::test::utils::InputGenerateData in_symbol_data;
        in_symbol_data.start_from = 0;
        in_symbol_data.range = 10;
        symbolsTensor = ov::test::utils::create_and_fill_tensor(ov::element::u8, symbolsShape, in_symbol_data);
        inputs.insert({ funcInputs[2].get_node_shared_ptr(), symbolsTensor });
    }

void StringTensorPackLayerCPUTest::SetUp() {
        StringTensorPackLayerTestParams basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = this->GetParam();
        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

        StringTensorPackSpecificParams StringTensorPackParams;
        ElementType indicesPrecision;
        std::tie(StringTensorPackParams, indicesPrecision, targetDevice) = basicParamsSet;

        InputShape indicesShape;
        InputShape symbolsShape;
        std::tie(indicesShape, symbolsShape) = StringTensorPackParams;

        init_input_shapes({indicesShape, indicesShape, symbolsShape});
        auto beginsParameter = std::make_shared<ov::op::v0::Parameter>(indicesPrecision, inputDynamicShapes[0]);
        auto endsParameter = std::make_shared<ov::op::v0::Parameter>(indicesPrecision, inputDynamicShapes[1]);
        auto symbolsParameter = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, inputDynamicShapes[2]);
        auto StringTensorPack = std::make_shared<ov::op::v15::StringTensorPack>(beginsParameter, endsParameter, symbolsParameter);

        ov::ParameterVector params{ beginsParameter, endsParameter, symbolsParameter };
        function = makeNgraphFunction(ov::element::string, params, StringTensorPack, "StringTensorPack");
}

TEST_P(StringTensorPackLayerCPUTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "StringTensorPack");
}

const std::vector<StringTensorPackSpecificParams> StringTensorPackParamsVector = {
    StringTensorPackSpecificParams {
        InputShape{{}, {{3}}},                                                      // begins/ends shape
        InputShape{{}, {{9}}},                                                      // utf-8 encoded symbols shape
    },
    StringTensorPackSpecificParams {
        InputShape{{}, {{1, 3, 4}}},                                                // begins/ends shape
        InputShape{{}, {{108}}},                                                    // utf-8 encoded symbols shape
    },
    StringTensorPackSpecificParams {
        InputShape{{}, {{1, 1, 1, 2}}},                                             // begins/ends shape
        InputShape{{}, {{67}}},                                                     // utf-8 encoded symbols shape
    },
    StringTensorPackSpecificParams {
        InputShape{{-1, -1, -1}, {{1, 3, 4}}},                                      // begins/ends shape
        InputShape{{-1}, {{108}}},                                                  // utf-8 encoded symbols shape
    },
    StringTensorPackSpecificParams {
        InputShape{{-1, {1, 4}, {2, 3}}, {{1, 2, 2}, {1, 1, 3}}},                   // begins/ends shape
        InputShape{{{50, 100}}, {{67}}},                                            // utf-8 encoded symbols shape
    },
    StringTensorPackSpecificParams {
        InputShape{{-1, {1, 4}, {1, 4}}, {{1, 1, 4}, {1, 4, 1}}},                   // begins/ends shape
        InputShape{{{50, 100}}, {{50}, {75}, {100}}},                               // utf-8 encoded symbols shape
    },
};

}  // namespace StringTensorPack
}  // namespace test
}  // namespace ov
