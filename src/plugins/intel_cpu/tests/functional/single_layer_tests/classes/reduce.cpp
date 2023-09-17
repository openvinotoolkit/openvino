// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reduce.hpp"

#include "gtest/gtest.h"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "test_utils/cpu_test_utils.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ngraph::helpers;
using namespace ov::test;

namespace CPULayerTestsDefinitions {

std::string ReduceCPULayerTest::getTestCaseName(testing::TestParamInfo<ReduceLayerCPUTestParamSet> obj) {
    basicReduceParams basicParams;
    CPUSpecificParams cpuParams;
    fusingSpecificParams fusingParams;
    std::map<std::string, ov::element::Type> additionalConfig;
    std::tie(basicParams, cpuParams, fusingParams, additionalConfig) = obj.param;

    std::vector<int> axes;
    ov::test::utils::OpType opType;
    bool keepDims;
    ngraph::helpers::ReductionType reductionType;
    ElementType netPrecision, inPrc, outPrc;
    std::vector<InputShape> inputShapes;

    std::tie(axes, opType, keepDims, reductionType, netPrecision, inPrc, outPrc, inputShapes) = basicParams;

    std::ostringstream result;
    result << "IS=(";
    for (const auto& shape : inputShapes) {
        result << ov::test::utils::partialShape2str({shape.first}) << "_";
    }
    result << ")_TS=(";
    for (const auto& shape : inputShapes) {
        for (const auto& item : shape.second) {
            result << ov::test::utils::vec2str(item) << "_";
        }
    }
    result << ")_axes=" << ov::test::utils::vec2str(axes) << "_";
    result << "opType=" << opType << "_";
    result << "type=" << reductionType << "_";
    if (keepDims)
        result << "KeepDims=true_";
    else
        result << "KeepDims=false_";
    result << "netPRC=" << netPrecision << "_";
    result << "inPRC=" << inPrc << "_";
    result << "outPRC=" << outPrc << "_";

    if (!additionalConfig.empty()) {
        result << "PluginConf";
        for (auto& item : additionalConfig) {
            result << "_" << item.first << "=" << item.second.get_type_name();
        }
    }

    result << CPUTestsBase::getTestCaseName(cpuParams);
    result << CpuTestWithFusing::getTestCaseName(fusingParams);

    return result.str();
}

void ReduceCPULayerTest::SetUp() {
    targetDevice = ov::test::utils::DEVICE_CPU;

    basicReduceParams basicParams;
    CPUSpecificParams cpuParams;
    fusingSpecificParams fusingParams;
    std::map<std::string, ov::element::Type> additionalConfig;
    std::tie(basicParams, cpuParams, fusingParams, additionalConfig) = this->GetParam();

    std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
    std::tie(postOpMgrPtr, fusedOps) = fusingParams;

    std::vector<int> axes;
    ov::test::utils::OpType opType;
    bool keepDims;
    ElementType inPrc, outPrc;
    std::vector<InputShape> inputShapes;

    std::tie(axes, opType, keepDims, reductionType, netPrecision, inPrc, outPrc, inputShapes) = basicParams;
    if (netPrecision == ElementType::boolean) {
        inPrc = outPrc = netPrecision;
    }

    configuration.insert(additionalConfig.begin(), additionalConfig.end());
    updateSelectedType(getPrimitiveType(), netPrecision == ElementType::boolean ? ElementType::i8 : netPrecision, configuration);

    init_input_shapes(inputShapes);

    ov::ParameterVector params;
    for (auto&& shape : inputDynamicShapes) {
        params.push_back(std::make_shared<ov::op::v0::Parameter>(netPrecision, shape));
    }
    auto paramOuts =
        ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

    std::vector<size_t> shapeAxes;
    switch (opType) {
    case ov::test::utils::OpType::SCALAR:
        if (axes.size() > 1)
            FAIL() << "In reduce op if op type is scalar, 'axis' input's must contain 1 element";
        break;
    case ov::test::utils::OpType::VECTOR:
        shapeAxes.push_back(axes.size());
        break;
    default:
        FAIL() << "Reduce op doesn't support operation type: " << opType;
    }
    auto reductionAxesNode = std::dynamic_pointer_cast<ngraph::Node>(
        std::make_shared<ngraph::opset3::Constant>(ngraph::element::Type_t::i64, ngraph::Shape(shapeAxes), axes));

    const auto reduce = ngraph::builder::makeReduce(paramOuts[0], reductionAxesNode, keepDims, reductionType);

    // hybrid layouts
    if (inFmts.size() != 0 && outFmts.size() == 0) {
        size_t outShapeSize = inputDynamicShapes[0].size() - axes.size();
        switch (outShapeSize) {
        case 0:
        case 1:
            outFmts.push_back(x);
            break;
        case 2:
            outFmts.push_back(nc);
            break;
        case 3:
            outFmts.push_back(tnc);
            break;
        case 4:
            outFmts.push_back(nchw);
            break;
        default:
            FAIL() << "Invaid outShapeSize: " << outShapeSize;
        }
    }

    function = makeNgraphFunction(netPrecision, params, reduce, "Reduce");
}

void ReduceCPULayerTest::generate_inputs(const std::vector<ngraph::Shape>& targetInputStaticShapes) {
    inputs.clear();
    const auto& funcInputs = function->inputs();
    for (size_t i = 0; i < funcInputs.size(); ++i) {
        const auto& funcInput = funcInputs[i];
        ov::Tensor tensor;
        if (reductionType == ngraph::helpers::ReductionType::Prod) {
            tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(),
                                                             targetInputStaticShapes[i],
                                                             10,
                                                             5);
            if (netPrecision == ElementType::f32) {
                auto* rawBlobDataPtr = static_cast<float*>(tensor.data());
                for (size_t i = 0; i < tensor.get_size(); ++i) {
                    rawBlobDataPtr[i] /= 10.f;
                }
            } else if (netPrecision == ElementType::f16) {
                auto *rawBlobDataPtr = static_cast<ngraph::float16 *>(tensor.data());
                for (size_t i = 0; i < tensor.get_size(); ++i) {
                    rawBlobDataPtr[i] /= 10.f;
                }
            } else if (netPrecision == ElementType::bf16) {
                auto* rawBlobDataPtr = static_cast<ngraph::bfloat16*>(tensor.data());
                for (size_t i = 0; i < tensor.get_size(); ++i) {
                    rawBlobDataPtr[i] /= 10.f;
                }
            }
        } else {
            tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i]);
        }

        inputs.insert({funcInput.get_node_shared_ptr(), tensor});
    }
}

TEST_P(ReduceCPULayerTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "Reduce");
}

namespace Reduce {

const std::vector<bool>& keepDims() {
    static const std::vector<bool> keepDims = {
            true,
            false,
    };
    return keepDims;
}

const std::vector<std::vector<int>>& axes() {
    static const std::vector<std::vector<int>> axes = {
            {0},
            {1},
            {2},
            {3}
    };
    return axes;
}

const std::vector<std::vector<int>>& axesND() {
    static const std::vector<std::vector<int>> axesND = {
            {0, 1},
            {0, 2},
            {0, 3},
            {1, 2},
            {1, 3},
            {2, 3},
            {0, 1, 2},
            {0, 1, 3},
            {0, 2, 3},
            {1, 2, 3},
            {0, 1, 2, 3}
    };
    return axesND;
}

const std::vector<ov::test::utils::OpType>& opTypes() {
    static const std::vector<ov::test::utils::OpType> opTypes = {
            ov::test::utils::OpType::SCALAR,
            ov::test::utils::OpType::VECTOR,
    };
    return opTypes;
}

const std::vector<ngraph::helpers::ReductionType>& reductionTypes() {
    static const std::vector<ngraph::helpers::ReductionType> reductionTypes = {
            ngraph::helpers::ReductionType::Mean,
            ngraph::helpers::ReductionType::Max,
            ngraph::helpers::ReductionType::Sum,
            ngraph::helpers::ReductionType::Min,
            ngraph::helpers::ReductionType::Prod,
            ngraph::helpers::ReductionType::L1,
            ngraph::helpers::ReductionType::L2,
    };
    return reductionTypes;
}

const std::vector<ElementType>& inpOutPrc() {
    static const std::vector<ElementType> inpOutPrc = {ElementType::f32};
    return inpOutPrc;
}

const std::vector<std::map<std::string, ov::element::Type>> additionalConfig() {
    static const std::vector<std::map<std::string, ov::element::Type>> additionalConfig = {
        {{ov::hint::inference_precision.name(), ov::element::f32}},
        {{ov::hint::inference_precision.name(), ov::element::bf16}},
// ARM doesn't support FP16 for now
#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)
        {{ov::hint::inference_precision.name(), ov::element::f16}},
#endif
    };
    return additionalConfig;
}

const std::vector<std::map<std::string, ov::element::Type>> additionalConfigFP32() {
    static const std::vector<std::map<std::string, ov::element::Type>> additionalConfig = {
        {{ov::hint::inference_precision.name(), ov::element::f32}}
    };
    return additionalConfig;
}

const std::vector<ngraph::helpers::ReductionType>& reductionTypesInt32() {
    static const std::vector<ngraph::helpers::ReductionType> reductionTypesInt32 = {
            ngraph::helpers::ReductionType::Sum,
            ngraph::helpers::ReductionType::Min,
            ngraph::helpers::ReductionType::Max,
            ngraph::helpers::ReductionType::L1,
    };
    return reductionTypesInt32;
}

}  // namespace Reduce
}  // namespace CPULayerTestsDefinitions
