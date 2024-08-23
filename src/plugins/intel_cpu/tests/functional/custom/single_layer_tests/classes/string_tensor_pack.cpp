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
        std::vector<int64_t> begins;
        std::vector<int64_t> ends;
        std::tie(indicesShape, symbolsShape, begins, ends) = StringTensorPackPar;
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

        ov::Tensor data_tensor;
        const auto& dataShape = targetInputStaticShapes[2];
        ov::test::utils::InputGenerateData in_data;
        in_data.start_from = 0;
        in_data.range = 10;
        data_tensor = ov::test::utils::create_and_fill_tensor(ov::element::u8, dataShape, in_data);
        inputs.insert({ funcInputs[0].get_node_shared_ptr(), data_tensor });
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
        std::vector<int64_t> begins;
        std::vector<int64_t> ends;
        std::tie(indicesShape, symbolsShape, begins, ends) = StringTensorPackParams;

        init_input_shapes({indicesShape, indicesShape, symbolsShape});
        auto symbolsParameter = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, inputDynamicShapes[2]);
        auto beginsConst = std::make_shared<ov::op::v0::Constant>(indicesPrecision, indicesShape.second.front(), begins);
        auto endsConst = std::make_shared<ov::op::v0::Constant>(indicesPrecision, indicesShape.second.front(), ends);
        auto StringTensorPack = std::make_shared<ov::op::v15::StringTensorPack>(beginsConst, endsConst, symbolsParameter);

        ov::ParameterVector params{ symbolsParameter };
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
        std::vector<int64_t>{1, 4, 6},                                              // begins values
        std::vector<int64_t>{4, 6, 8}                                               // ends values
    },
    StringTensorPackSpecificParams {
        InputShape{{}, {{1, 3, 4}}},                                                // begins/ends shape
        InputShape{{}, {{108}}},                                                    // utf-8 encoded symbols shape
        std::vector<int64_t>{7, 28, 30, 30, 41, 42, 50, 50, 74, 76, 80, 92},        // begins values
        std::vector<int64_t>{28, 30, 30, 41, 42, 50, 50, 74, 76, 80, 92, 108}       // ends values
    },
    StringTensorPackSpecificParams {
        InputShape{{}, {{1, 1, 1, 2}}},                                             // begins/ends shape
        InputShape{{}, {{67}}},                                                     // utf-8 encoded symbols shape
        std::vector<int64_t>{30, 31},                                               // begins values
        std::vector<int64_t>{31, 31}                                                // ends values
    },
    StringTensorPackSpecificParams {
        InputShape{{-1, -1, -1}, {{1, 3, 4}}},                                      // begins/ends shape
        InputShape{{-1}, {{108}}},                                                  // utf-8 encoded symbols shape
        std::vector<int64_t>{7, 28, 30, 30, 41, 42, 50, 50, 74, 76, 80, 92},        // begins values
        std::vector<int64_t>{28, 30, 30, 41, 42, 50, 50, 74, 76, 80, 92, 108}       // ends values
    },
    StringTensorPackSpecificParams {
        InputShape{{-1, {1, 4}, {2, 3}}, {{1, 2, 2}, {1, 1, 3}}},                   // begins/ends shape
        InputShape{{{50, 100}}, {{67}}},                                            // utf-8 encoded symbols shape
        std::vector<int64_t>{30, 31, 56, 60},                                       // begins values
        std::vector<int64_t>{31, 56, 60, 67}                                        // ends values
    },
    StringTensorPackSpecificParams {
        InputShape{{-1, {1, 4}, {1, 4}}, {{1, 1, 4}, {1, 4, 1}}},                   // begins/ends shape
        InputShape{{{50, 100}}, {{50}, {75}, {100}}},                               // utf-8 encoded symbols shape
        std::vector<int64_t>{30, 31, 46, 50},                                       // begins values
        std::vector<int64_t>{31, 46, 50, 50}                                        // ends values
    },
};

}  // namespace StringTensorPack
}  // namespace test
}  // namespace ov
