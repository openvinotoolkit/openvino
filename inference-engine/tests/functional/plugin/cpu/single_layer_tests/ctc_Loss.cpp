// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <functional_test_utils/ov_tensor_utils.hpp>
#include <ngraph_functions/builders.hpp>
#include <string>
#include <tuple>
#include <vector>

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "test_utils/cpu_test_utils.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ov::test;

namespace CPULayerTestsDefinitions {

using CTCLossShapeParams = std::pair<std::vector<ngraph::PartialShape>, std::vector<std::vector<ngraph::Shape>>>;

using CTCLossLayerCPUTestParams = std::tuple<CTCLossShapeParams,            // dynamic, static shape[N, T, C]
                                            bool,                           // preprocessCollapseRepeated
                                            bool,                           // ctcMergeRepeated
                                            bool,                           // unique
                                            ElementType,                    // fp precision for logits
                                            ElementType                     // int precision for label and length
                                            >;
ngraph::ParameterVector makeDynamicParams(const std::vector<ElementType>& types,
                                          const std::vector<ov::PartialShape>& shapes) {
    ngraph::ParameterVector outs;
    NGRAPH_CHECK(types.size() == shapes.size());
    for (size_t i = 0; i < types.size(); i++) {
        auto paramNode = std::make_shared<ov::op::v0::Parameter>(types[i], shapes[i]);
        outs.push_back(paramNode);
    }
    return outs;
}

class CTCLossLayerCPUTest : public testing::WithParamInterface<CTCLossLayerCPUTestParams>, virtual public SubgraphBaseTest, public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<CTCLossLayerCPUTestParams>& obj) {
        CTCLossShapeParams shapes;
        bool preprocessCollapseRepeated;
        bool ctcMergeRepeated;
        bool unique;
        ElementType fPrecision;
        ElementType iPrecision;
        std::tie(shapes, preprocessCollapseRepeated, ctcMergeRepeated, unique, fPrecision, iPrecision) = obj.param;
        std::ostringstream results;
        results << "IS=" << CommonTestUtils::partialShape2str({shapes.first}) << "_";
        results << "TS=";
        for (auto& shapes : shapes.second) {
            for (auto& shape : shapes) {
                size_t N = shape[0];
                size_t T = shape[1];
                size_t C = shape[2];
                results << "{" << N << "," << T << "," << C << "}"
                    << "_";
            }
        }
        results << "preprocessCollapseRepeated=" << preprocessCollapseRepeated << "_";
        results << "ctcMergeRepeated=" << ctcMergeRepeated << "_";
        results << "unique=" << unique << "_";

        results << "fPrecision=" << fPrecision << "_";
        results << "iPrecision=" << iPrecision << "_";

        return results.str();
    }

protected:
    void SetUp() override {
        CTCLossShapeParams shapes;
        bool preprocessCollapseRepeated;
        bool ctcMergeRepeated;
        bool unique;
        ElementType fPrecision;
        ElementType iPrecision;
        std::tie(shapes, preprocessCollapseRepeated, ctcMergeRepeated, unique, fPrecision, iPrecision) = GetParam();

        targetDevice = CommonTestUtils::DEVICE_CPU;
        selectedType = std::string("ref_any_FP32");

        for (auto& shapes : shapes.second) {
            for (auto& shape : shapes) {
                size_t N = shape[0];
                size_t T = shape[1];
                size_t C = shape[2];
                targetStaticShapes.push_back({{N, T, C}, {N}, {N, T}, {N}});
            }
        }

        auto inputDynamicShapesValues = shapes.first.front();
        ov::PartialShape shapeN{inputDynamicShapesValues[0]};
        ov::PartialShape shapeNT{inputDynamicShapesValues[0], inputDynamicShapesValues[1]};
        ov::PartialShape shapeNTC{inputDynamicShapesValues[0], inputDynamicShapesValues[1], inputDynamicShapesValues[2]};
        inputDynamicShapes = {shapeNTC, shapeN, shapeNT, shapeN};

        std::vector<ElementType> types{fPrecision, iPrecision, iPrecision, iPrecision};
        std::vector<ov::PartialShape> partialShapes{inputDynamicShapesValues, shapeN, shapeNT, shapeN};

        auto params = makeDynamicParams(types, partialShapes);
        auto bank = ngraph::op::Constant::create(ngraph::element::i64, ngraph::Shape{ }, {7});

        auto ctcLoss = std::make_shared<ngraph::opset4::CTCLoss>(params[0], params[1], params[2],
            params[3], bank, preprocessCollapseRepeated, ctcMergeRepeated, unique);
        ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(ctcLoss)};
        function = std::make_shared<ngraph::Function>(results, params, "CTCLossLayerCPUTest");
    };

    void generate_inputs(const std::vector<ngraph::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        const auto& dataShape = targetInputStaticShapes[0];
        const auto N = dataShape[0];
        const auto T = dataShape[1];
        const auto C = dataShape[2];
        ngraph::Shape shapeN{dataShape[0]};
        ngraph::Shape shapeNT{dataShape[0], dataShape[1]};

        std::mt19937 gen(42);
        std::uniform_int_distribution<unsigned long> dist(1, T);
        std::vector<int32_t> logitLength(N, 0);
        for (int n = 0; n < N; n++) {
            const int len = dist(gen);
            logitLength[n] = len;
        }
        for (int i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            ov::runtime::Tensor tensor;
            if (i == 0) {
                tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(),
                                                                 dataShape,
                                                                 10,
                                                                 0,
                                                                 10);
            } else if (i == 1) {
                tensor = ov::runtime::Tensor{funcInput.get_element_type(), {shapeN}};
                if (funcInput.get_element_type() == ElementType::i32) {
                    auto begin = tensor.data<int32_t>();
                    std::copy(logitLength.begin(), logitLength.end(), begin);
                } else if (funcInput.get_element_type() == ElementType::i64) {
                    auto begin = tensor.data<int64_t>();
                    std::copy(logitLength.begin(), logitLength.end(), begin);
                }
            } else if (i == 2) {
                std::mt19937 genLable(42);
                std::uniform_int_distribution<unsigned long> distLabel(0, C - 2);
                std::vector<int32_t> labels(N * T, 0);
                for (int n = 0; n < N * T; n++) {
                    const int len = distLabel(genLable);
                    labels[n] = len;
                }
                tensor = ov::runtime::Tensor{funcInput.get_element_type(), {shapeNT}};
                if (funcInput.get_element_type() == ElementType::i32) {
                    auto begin = tensor.data<int32_t>();
                    std::copy(labels.begin(), labels.end(), begin);
                } else if (funcInput.get_element_type() == ElementType::i64) {
                    auto begin = tensor.data<int64_t>();
                    std::copy(labels.begin(), labels.end(), begin);
                }
            } else if (i == 3) {
                std::mt19937 gen(1);
                std::uniform_int_distribution<unsigned long> dist(1, T);

                std::vector<int32_t> labelLength(N, 0);
                for (int n = 0; n < N; n++) {
                    const int len = dist(gen);
                    labelLength[n] = std::min(len, logitLength[n] - 1);
                }

                tensor = ov::runtime::Tensor{funcInput.get_element_type(), {shapeN}};
                if (funcInput.get_element_type() == ElementType::i32) {
                    auto begin = tensor.data<int32_t>();
                    std::copy(labelLength.begin(), labelLength.end(), begin);
                } else if (funcInput.get_element_type() == ElementType::i64) {
                    auto begin = tensor.data<int64_t>();
                    std::copy(labelLength.begin(), labelLength.end(), begin);
                }
            }
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }
};

TEST_P(CTCLossLayerCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
    CheckPluginRelatedResults(executableNetwork, "CTCLoss");
}

namespace {
    const std::vector<ElementType> fPrecisions = {
            ElementType::f32,
            // ElementType::f16
    };
    const std::vector<ElementType> iPrecisions = {
            ElementType::i32,
            ElementType::i64
    };

    const std::vector<bool> preprocessCollapseRepeated = {true, false};
    const std::vector<bool> ctcMergeRepeated = {true, false};
    const std::vector<bool> unique = {true, false};

    const std::vector<CTCLossShapeParams> shapes = {
        {
            // dynamic
            {
                {-1, -1, -1},
            },
            // target
            {
                {{3, 6, 8}, {2, 5, 8}, {5, 6, 8}}
            }
        },
        {
            // dynamic
            {
                {-1, 10, 8},
            },
            // target
            {
                {{5, 10, 8}, {2, 10, 8}}
            }
        }
    };

const auto basicCases = ::testing::Combine(::testing::ValuesIn(shapes),
                                           ::testing::ValuesIn(preprocessCollapseRepeated),
                                           ::testing::ValuesIn(ctcMergeRepeated),
                                           ::testing::ValuesIn(unique),
                                           ::testing::ValuesIn(fPrecisions),
                                           ::testing::ValuesIn(iPrecisions));

INSTANTIATE_TEST_SUITE_P(smoke_CTCLossCPU,
                         CTCLossLayerCPUTest,
                         basicCases,
                         CTCLossLayerCPUTest::getTestCaseName);
}  // namespace

}  // namespace CPULayerTestsDefinitions