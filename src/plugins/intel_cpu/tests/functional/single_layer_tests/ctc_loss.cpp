// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <common_test_utils/ov_tensor_utils.hpp>
#include <ov_models/builders.hpp>
#include <string>
#include <tuple>
#include <vector>

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "test_utils/cpu_test_utils.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ov::test;

namespace CPULayerTestsDefinitions {

// N,T,C
using CTCLossShapeParams = std::pair<std::vector<ngraph::PartialShape>, std::vector<std::vector<ngraph::Shape>>>;

using CTCLossLayerCPUTestParams = std::tuple<CTCLossShapeParams,            // [N, T, C]
                                            int,                            // blank value
                                            bool,                           // preprocessCollapseRepeated
                                            bool,                           // ctcMergeRepeated
                                            bool,                           // unique
                                            ngraph::element::Type,          // fp precision for logits
                                            ngraph::element::Type           // int precision for label and length
                                            >;

class CTCLossLayerCPUTest : public testing::WithParamInterface<CTCLossLayerCPUTestParams>, virtual public SubgraphBaseTest, public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<CTCLossLayerCPUTestParams>& obj) {
        CTCLossShapeParams shapes;
        int blank;
        bool preprocessCollapseRepeated;
        bool ctcMergeRepeated;
        bool unique;
        ngraph::element::Type fPrecision;
        ngraph::element::Type iPrecision;
        std::tie(shapes, blank, preprocessCollapseRepeated, ctcMergeRepeated, unique, fPrecision, iPrecision) = obj.param;
        std::ostringstream results;
        results << "IS=" << ov::test::utils::partialShape2str({shapes.first}) << "_";
        results << "TS=";
        for (std::vector<ngraph::Shape>& staticShapes : shapes.second) {
            for (ngraph::Shape& shape : staticShapes) {
                size_t N = shape[0];
                size_t T = shape[1];
                size_t C = shape[2];
                results << "{" << N << "," << T << "," << C << "}"
                    << "_";
            }
        }
        results << "blank=" << blank << "_";
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
        ngraph::element::Type fPrecision;
        ngraph::element::Type iPrecision;
        std::tie(shapes, blank, preprocessCollapseRepeated, ctcMergeRepeated, unique, fPrecision, iPrecision) = GetParam();

        targetDevice = ov::test::utils::DEVICE_CPU;
        selectedType = std::string("ref_any_FP32");

        for (std::vector<ngraph::Shape>& staticShapes : shapes.second) {
            for (ngraph::Shape& shape : staticShapes) {
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

        std::vector<ngraph::element::Type> types{fPrecision, iPrecision, iPrecision, iPrecision};
        std::vector<ov::PartialShape> partialShapes{inputDynamicShapesValues, shapeN, shapeNT, shapeN};

        ov::ParameterVector params;
        for (size_t i = 0; i < types.size(); i++) {
            auto param_node = std::make_shared<ov::op::v0::Parameter>(types[i], partialShapes[i]);
            params.push_back(param_node);
        }
        auto bankNode = ngraph::op::Constant::create(ngraph::element::i64, ngraph::Shape{ }, {blank});

        auto ctcLoss = std::make_shared<ngraph::opset4::CTCLoss>(params[0], params[1], params[2],
            params[3], bankNode, preprocessCollapseRepeated, ctcMergeRepeated, unique);
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
        ngraph::Shape shapeN{N};
        ngraph::Shape shapeNT{N, T};

        std::mt19937 gen(42);
        std::uniform_int_distribution<unsigned long> dist(1, T);
        std::vector<int32_t> logitLength(N, 0);
        for (size_t n = 0; n < N; n++) {
            logitLength[n] = dist(gen);
        }
        for (size_t i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            ov::Tensor tensor;
            if (i == 0) {
                tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(),
                                                                 dataShape,
                                                                 10,
                                                                 0,
                                                                 10);
            } else if (i == 1) {
                tensor = ov::Tensor{funcInput.get_element_type(), {shapeN}};
                if (funcInput.get_element_type() == ElementType::i32) {
                    auto begin = tensor.data<int32_t>();
                    std::copy(logitLength.begin(), logitLength.end(), begin);
                } else if (funcInput.get_element_type() == ElementType::i64) {
                    auto begin = tensor.data<int64_t>();
                    std::copy(logitLength.begin(), logitLength.end(), begin);
                }
            } else if (i == 2) {
                std::mt19937 genLable(42);
                std::uniform_int_distribution<unsigned long> distLabel(0, C - 1);
                std::vector<int32_t> labels(N * T, 0);
                for (size_t n = 0; n < N * T; n++) {
                    int value;
                    // make sure blank not be inclded in labels
                    while ((value = distLabel(genLable)) == blank) {}
                    labels[n] = value;
                }
                tensor = ov::Tensor{funcInput.get_element_type(), {shapeNT}};
                if (funcInput.get_element_type() == ElementType::i32) {
                    auto begin = tensor.data<int32_t>();
                    std::copy(labels.begin(), labels.end(), begin);
                } else if (funcInput.get_element_type() == ElementType::i64) {
                    auto begin = tensor.data<int64_t>();
                    std::copy(labels.begin(), labels.end(), begin);
                }
            } else if (i == 3) {
                std::mt19937 gen(24);
                std::uniform_int_distribution<unsigned long> dist(1, T);

                std::vector<int32_t> labelLength(N, 0);
                for (size_t n = 0; n < N; n++) {
                    const int len = dist(gen);
                    // make sure lableLen <= logitLen
                    labelLength[n] = std::min(len, logitLength[n]);
                }

                tensor = ov::Tensor{funcInput.get_element_type(), {shapeN}};
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

private:
    int blank;
};

TEST_P(CTCLossLayerCPUTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "CTCLoss");
}

namespace {
    const ngraph::element::TypeVector fPrecisions = {
        ngraph::element::f32
        // ngraph::element::f16
    };

    const ngraph::element::TypeVector iPrecisions = {
        ngraph::element::i32,
        ngraph::element::i64
    };

    const std::vector<bool> preprocessCollapseRepeated = {true, false};
    const std::vector<bool> ctcMergeRepeated = {true, false};
    const std::vector<bool> unique = {true, false};

    const std::vector<CTCLossShapeParams> shapes = {
        {
            // dynamic undifined
            {
                {-1, -1, -1},
            },
            // target
            {
                {{3, 6, 8}, {2, 5, 6}, {5, 6, 10}}
            }
        },
        {
            // dynamic lower/upper bound
            {
                {{1, 10}, {5, 10}, {6, 12}},
            },
            // target
            {
                {{1, 5, 6}, {10, 10, 12}, {5, 7, 8}}
            }
        },
    };

    const std::vector<int> blanks = {
        0, 2, 5
    };

const auto basicCases = ::testing::Combine(::testing::ValuesIn(shapes),
                                           ::testing::ValuesIn(blanks),
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
