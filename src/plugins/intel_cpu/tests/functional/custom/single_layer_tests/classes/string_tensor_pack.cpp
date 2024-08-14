// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "string_tensor_unpack.hpp"
#include "utils/cpu_test_utils.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/precision_support.h"

//using namespace CPUTestUtils;
//
//namespace ov {
//namespace test {
//namespace StringTensorUnpack {
//std::string StringTensorUnpackLayerCPUTest::getTestCaseName(testing::TestParamInfo<StringTensorUnpackLayerCPUTestParamsSet> obj) {
//        StringTensorUnpackLayerTestParams basicParamsSet;
//        CPUSpecificParams cpuParams;
//        std::tie(basicParamsSet, cpuParams) = obj.param;
//        std::string td;
//        ElementType netPrecision;
//        StringTensorUnpackSpecificParams StringTensorUnpackPar;
//        std::tie(StringTensorUnpackPar, netPrecision, td) = basicParamsSet;
//
//        InputShape inputShape;
//        std::vector<std::string> data;
//        std::tie(inputShape, data) = StringTensorUnpackPar;
//        std::ostringstream result;
//
//        result << netPrecision << "_IS=";
//        result << ov::test::utils::partialShape2str({ inputShape.first }) << "_";
//        result << "TS=";
//        result << "(";
//        for (const auto& targetShape : inputShape.second) {
//            result << ov::test::utils::vec2str(targetShape) << "_";
//        }
//        result << ")_";
//        result << "data=" << ov::test::utils::vec2str(data) << "_";
//        result << "dataPrecision=" << netPrecision << "_";
//        result << CPUTestsBase::getTestCaseName(cpuParams);
//
//        return result.str();
//    }
//
//void StringTensorUnpackLayerCPUTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
//        inputs.clear();
//        const auto& funcInputs = function->inputs();
//
//        ov::Tensor data_tensor;
//        const auto& dataPrecision = funcInputs[0].get_element_type();
//        const auto& dataShape = targetInputStaticShapes.front();
//        ov::test::utils::InputGenerateData in_data;
//        in_data.start_from = 0;
//        in_data.range = 10;
//        data_tensor = ov::test::utils::create_and_fill_tensor(ov::element::string, dataShape, in_data);
//        inputs.insert({ funcInputs[0].get_node_shared_ptr(), data_tensor });
//    }
//
//void StringTensorUnpackLayerCPUTest::SetUp() {
//        StringTensorUnpackLayerTestParams basicParamsSet;
//        CPUSpecificParams cpuParams;
//        std::tie(basicParamsSet, cpuParams) = this->GetParam();
//        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
//
//        StringTensorUnpackSpecificParams StringTensorUnpackParams;
//        ElementType inputPrecision;
//        std::tie(StringTensorUnpackParams, inputPrecision, targetDevice) = basicParamsSet;
//
//        InputShape dataInputShape;
//        std::vector<std::string> data;
//        std::tie(dataInputShape, data) = StringTensorUnpackParams;
//
//        init_input_shapes({dataInputShape});
//        auto dataParameter = std::make_shared<ov::op::v0::Parameter>(ov::element::string, inputDynamicShapes[0]);
//        auto StringTensorUnpack = std::make_shared<ov::op::v15::StringTensorUnpack>(dataParameter);
//
//        ov::ParameterVector params{ dataParameter };
//        function = makeNgraphFunction(inputPrecision, params, StringTensorUnpack, "StringTensorUnpack");
//}
//
//TEST_P(StringTensorUnpackLayerCPUTest, CompareWithRefs) {
//    run();
//    CheckPluginRelatedResults(compiledModel, "StringTensorUnpack");
//}
//
//const std::vector<StringTensorUnpackSpecificParams> StringTensorUnpackParamsVector = {
//    StringTensorUnpackSpecificParams {
//        InputShape{{}, {{1, 12, 9}}},
//        std::vector<int64_t>{4, 4},
//        std::vector<int64_t>{2, 2},
//        ov::Strides{1, 1},
//        ov::Strides{1, 1},
//        ov::Shape{0, 0},
//        ov::Shape{0, 0}
//    }
//};
//
//}  // namespace StringTensorUnpack
//}  // namespace test
//}  // namespace ov
//