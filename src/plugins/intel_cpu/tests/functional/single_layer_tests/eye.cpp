/// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset9.hpp>
#include <common_test_utils/ov_tensor_utils.hpp>
#include "test_utils/cpu_test_utils.hpp"
#include "ov_models/builders.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ov::test;

namespace CPULayerTestsDefinitions {
namespace {
    std::vector<InputShape> inputShape;
    std::vector<int> outBatchShape;
    int rowNum, colNum;
    int shift;
}  // namespace

using EyeLayerTestParams = std::tuple<
        std::vector<InputShape>,    // eye shape
        std::vector<int>,           // output batch shape
        std::vector<int>,           // eye params (rows, cols, diag_shift)
        ElementType,                // Net precision
        TargetDevice>;              // Device name

using EyeLayerCPUTestParamsSet = std::tuple<
        CPULayerTestsDefinitions::EyeLayerTestParams,
        CPUSpecificParams>;

class EyeLayerCPUTest : public testing::WithParamInterface<EyeLayerCPUTestParamsSet>,
                            virtual public SubgraphBaseTest, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<EyeLayerCPUTestParamsSet> obj) {
        CPULayerTestsDefinitions::EyeLayerTestParams basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = obj.param;
        std::string td;
        ElementType netPr;
        std::vector<int> eyePar;
        std::tie(inputShape, outBatchShape, eyePar, netPr, td) = basicParamsSet;
        std::ostringstream result;
        result << "EyeTest_";
        result << "IS=(";
        for (const auto& shape : inputShape) {
            result << ov::test::utils::partialShape2str({shape.first}) << "_";
        }
        result << ")_TS=(";
        for (const auto& shape : inputShape) {
            for (const auto& item : shape.second) {
                result << ov::test::utils::vec2str(item) << "_";
            }
        }
        result << "rowNum=" << eyePar[0] << "_";
        result << "colNum=" << eyePar[1] << "_";
        result << "diagShift=" << eyePar[2] << "_";
        result << "batchShape=" << ov::test::utils::vec2str(outBatchShape) << "_";
        result << netPr << "_";
        result << CPUTestsBase::getTestCaseName(cpuParams) << "_";
        result << std::to_string(obj.index);
        return result.str();
    }
protected:
    void SetUp() override {
        CPULayerTestsDefinitions::EyeLayerTestParams basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = this->GetParam();
        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

        ElementType netPrecision;
        std::vector<int> eyePar;
        std::tie(inputShape, outBatchShape, eyePar, netPrecision, targetDevice) = basicParamsSet;
        rowNum = eyePar[0];
        colNum = eyePar[1];
        shift = eyePar[2];

        init_input_shapes(inputShape);

        selectedType = std::string("ref_I32");
        function = createFunction();
    }

    std::shared_ptr<ngraph::Function> createFunction() {
        ov::ParameterVector inputParams;
        for (auto&& shape : inputDynamicShapes) {
            inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(ov::element::i32, shape));
        }
        auto rowsPar = inputParams[0];
        rowsPar->set_friendly_name("rows");
        auto colsPar = inputParams[1];
        colsPar->set_friendly_name("cols");
        auto diagPar = inputParams[2];
        diagPar->set_friendly_name("diagInd");
        if (inputParams.size() == 4) {
            auto batchShapePar = inputParams[3];
            batchShapePar->set_friendly_name("batchShape");
            auto eyelikeBatchShape = std::make_shared<ngraph::op::v9::Eye>(rowsPar, colsPar, diagPar, batchShapePar, ngraph::element::i32);
            eyelikeBatchShape->get_rt_info() = getCPUInfo();
            return makeNgraphFunction(ngraph::element::i32, inputParams, eyelikeBatchShape, "Eye");
        } else {
            auto eyelikePure = std::make_shared<ngraph::op::v9::Eye>(rowsPar, colsPar, diagPar, ngraph::element::i32);
            eyelikePure->get_rt_info() = getCPUInfo();
            return makeNgraphFunction(ngraph::element::i32, inputParams, eyelikePure, "Eye");
        }
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        for (size_t i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            ov::Tensor tensor;
            if (i == 3) {  // batch shape
                tensor = ov::Tensor(funcInput.get_element_type(), targetInputStaticShapes[i]);
                int *batchShapePtr = tensor.data<int>();
                // Spec: batch_shape - 1D tensor with non-negative values of type T_NUM defines leading batch dimensions of output shape
                EXPECT_EQ(targetInputStaticShapes[i].size(), 1);
                EXPECT_EQ(targetInputStaticShapes[i][0], outBatchShape.size());
                for (size_t j = 0; j < targetInputStaticShapes[i][0]; j++) {
                    batchShapePtr[j] = outBatchShape[j];
                }
            } else {
                tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i], 1,
                    (i == 0 ? rowNum : (i == 1 ? colNum : shift)));
            }
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }
};

TEST_P(EyeLayerCPUTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "Eye");
}

namespace {

const std::vector<ElementType> netPrecisions = {
    ElementType::f32, ElementType::bf16, ElementType::i32,
    ElementType::i8, ElementType::u8
};

const std::vector<std::vector<int>> eyePars = {  // rows, cols, diag_shift
    {3, 3, 0},
    {3, 4, 1},
    {4, 3, 1},
    {3, 4, 0},
    {4, 3, 0},
    {3, 4, -1},
    {4, 3, -1},
    {3, 4, 10},
    {4, 4, -2},
    {0, 0, 0}
};

// dummy parameter to prevent empty set of test cases
const std::vector<std::vector<int>> emptyBatchShape = {{0}};

const std::vector<std::vector<int>> batchShapes1D = {
    {3}, {2}, {1}, {0}
};
const std::vector<std::vector<int>> batchShapes2D = {
    {3, 2}, {2, 1}, {0, 0}
};
// Ticket: 85127
// const std::vector<std::vector<int>> batchShapes3D = {
//     {3, 2, 1}, {1, 1, 1}
// };

INSTANTIATE_TEST_SUITE_P(smoke_Eye2D_PureScalar_Test, EyeLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         ::testing::ValuesIn(static_shapes_to_test_representation(
                                             std::vector<std::vector<ov::Shape>> {{{}, {}, {}}})),
                                         ::testing::ValuesIn(emptyBatchShape),
                                         ::testing::ValuesIn(eyePars),
                                         ::testing::ValuesIn(netPrecisions),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::Values(CPUSpecificParams{{}, {}, {}, {}})),
                         EyeLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Eye2D_WithNonScalar_Test, EyeLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         ::testing::ValuesIn(static_shapes_to_test_representation(
                                             std::vector<std::vector<ov::Shape>> {{{1}, {1}, {1}}})),
                                         ::testing::ValuesIn(emptyBatchShape),
                                         ::testing::ValuesIn(eyePars),
                                         ::testing::ValuesIn(netPrecisions),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::Values(CPUSpecificParams{{}, {}, {}, {}})),
                         EyeLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Eye_1DBatch_Test, EyeLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         ::testing::ValuesIn(static_shapes_to_test_representation(
                                             std::vector<std::vector<ov::Shape>> {{{}, {}, {}, {1}}})),
                                         ::testing::ValuesIn(batchShapes1D),
                                         ::testing::ValuesIn(eyePars),
                                         ::testing::ValuesIn(netPrecisions),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::Values(CPUSpecificParams{{}, {}, {}, {}})),
                         EyeLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Eye_2DBatch_Test, EyeLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         ::testing::ValuesIn(static_shapes_to_test_representation(
                                             std::vector<std::vector<ov::Shape>> {{{}, {}, {}, {2}}})),
                                         ::testing::ValuesIn(batchShapes2D),
                                         ::testing::ValuesIn(eyePars),
                                         ::testing::ValuesIn(netPrecisions),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::Values(CPUSpecificParams{{}, {}, {}, {}})),
                         EyeLayerCPUTest::getTestCaseName);

// Ticket: 85127
// INSTANTIATE_TEST_SUITE_P(smoke_Eye_3DBatch_Test, EyeLayerCPUTest,
//                          ::testing::Combine(
//                                  ::testing::Combine(
//                                          ::testing::ValuesIn(static_shapes_to_test_representation(
//                                              std::vector<std::vector<ov::Shape>> {{{}, {}, {}, {3}}})),
//                                          ::testing::ValuesIn(batchShapes3D),
//                                          ::testing::ValuesIn(eyePars),
//                                          ::testing::ValuesIn(netPrecisions),
//                                          ::testing::Values(ov::test::utils::DEVICE_CPU)),
//                                  ::testing::Values(CPUSpecificParams{{}, {}, {}, {}})),
//                          EyeLayerCPUTest::getTestCaseName);

const std::vector<std::vector<InputShape>> dynShapes = {
        {
            {{-1}, {{1}, {1}}},  // input 0
            {{-1}, {{1}, {1}}},  // input 1
            {{-1}, {{1}, {1}}}   // input 2
        },
};

const std::vector<std::vector<InputShape>> dynShapesWith2DBatches = {
        {
            {{-1}, {{1}, {1}, {1}}},  // input 0
            {{-1}, {{1}, {1}, {1}}},  // input 1
            {{-1}, {{1}, {1}, {1}}},  // input 2
            {{2}, {{2}, {2}, {2}}}    // input 3
        },
};

// Ticket: 85127
// const std::vector<std::vector<InputShape>> dynShapesWith3DBatches = {
//         {
//             {{-1}, {{1}, {1}, {1}}},  // input 0
//             {{-1}, {{1}, {1}, {1}}},  // input 1
//             {{-1}, {{1}, {1}, {1}}},  // input 2
//             {{3}, {{3}, {3}, {3}}}    // input 3
//         },
// };

INSTANTIATE_TEST_SUITE_P(smoke_Eye_Dynamic_Test, EyeLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         ::testing::ValuesIn(dynShapes),
                                         ::testing::ValuesIn(emptyBatchShape),
                                         ::testing::ValuesIn(eyePars),
                                         ::testing::ValuesIn(netPrecisions),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::Values(CPUSpecificParams{{}, {}, {}, {}})),
                         EyeLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Eye_With2DBatchShape_Dynamic_Test, EyeLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         ::testing::ValuesIn(dynShapesWith2DBatches),
                                         ::testing::ValuesIn(batchShapes2D),
                                         ::testing::ValuesIn(eyePars),
                                         ::testing::ValuesIn(netPrecisions),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::Values(CPUSpecificParams{{}, {}, {}, {}})),
                         EyeLayerCPUTest::getTestCaseName);

// Ticket: 85127
// INSTANTIATE_TEST_SUITE_P(smoke_Eye_With3DBatchShape_Dynamic_Test, EyeLayerCPUTest,
//                          ::testing::Combine(
//                                  ::testing::Combine(
//                                          ::testing::ValuesIn(dynShapesWith3DBatches),
//                                          ::testing::ValuesIn(batchShapes3D),
//                                          ::testing::ValuesIn(eyePars),
//                                          ::testing::ValuesIn(netPrecisions),
//                                          ::testing::Values(ov::test::utils::DEVICE_CPU)),
//                                  ::testing::Values(CPUSpecificParams{{}, {}, {}, {}})),
//                          EyeLayerCPUTest::getTestCaseName);
} // namespace
} // namespace CPULayerTestsDefinitions
