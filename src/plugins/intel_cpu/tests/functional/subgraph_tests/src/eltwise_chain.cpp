// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include <debug.h>
#include <shared_test_classes/base/ov_subgraph.hpp>
#include <ov_models/builders.hpp>
#include "common_test_utils/common_utils.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>
#include "functional_test_utils/skip_tests_config.hpp"
#include "test_utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;
using ngraph::helpers::EltwiseTypes;
using namespace ov::test;

namespace CPUSubgraphTestsDefinitions {

typedef std::tuple<
        std::vector<InputShape>, // Input shapes
        ngraph::helpers::InputLayerType,                                                       // Secondary input type
        std::vector<ElementType>,                                                              // Input precisions
        std::vector<EltwiseTypes>,                                                             // Eltwise operations
        bool,                                                                                  // With quantization
        std::string                                                                            // Device name
> EltwiseChainTuple;

class EltwiseChainTest : public testing::WithParamInterface<EltwiseChainTuple>,
                         virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<EltwiseChainTuple> &obj) {
        std::vector<InputShape> inputShapes;
        ngraph::helpers::InputLayerType secondaryInputType;
        std::vector<ElementType> inputPrecisions;
        std::vector<EltwiseTypes> eltwiseOpTypes;
        bool withQuantization;
        std::string targetName;
        std::tie(inputShapes, secondaryInputType, inputPrecisions, eltwiseOpTypes, withQuantization, targetName) = obj.param;
        std::ostringstream results;

        results << "IS=(";
        for (const auto& shape : inputShapes) {
            results << ov::test::utils::partialShape2str({shape.first}) << "_";
        }
        results << ")_TS=(";
        for (const auto& shape : inputShapes) {
            for (const auto& item : shape.second) {
                results << ov::test::utils::vec2str(item) << "_";
            }
        }
        for (size_t i = 0; i < inputPrecisions.size(); i++) {
            results << "InPRC" << std::to_string(i) << "=" << inputPrecisions[i] << "_";
        }
        for (size_t i = 0; i < eltwiseOpTypes.size(); i++) {
            results << "Op" << std::to_string(i) << "=" << eltwiseOpTypes[i] << "_";
        }
        results << "secondaryInputType=" << secondaryInputType << "_";
        results << "WithQuant=" << withQuantization << "_";
        results << "targetDevice=" << targetName;

        return results.str();
    }

    void generate_inputs(const std::vector<ngraph::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        for (size_t i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            ov::Tensor tensor;
            tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i], 10, 1, 1);
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }

protected:
    void SetUp() override {
        abs_threshold = 0.1f;

        std::vector<InputShape> inputShapes;
        ngraph::helpers::InputLayerType secondaryInputType;
        std::vector<ElementType> inputPrecisions;
        std::vector<EltwiseTypes> eltwiseOpTypes;
        bool withQuantization;
        std::tie(inputShapes, secondaryInputType, inputPrecisions, eltwiseOpTypes, withQuantization, targetDevice) = this->GetParam();

        init_input_shapes(inputShapes);

        ngraph::ParameterVector ngraphParam;
        std::vector<std::shared_ptr<ngraph::Node>> ngraphInputs;
        if (secondaryInputType == ngraph::helpers::InputLayerType::PARAMETER) {
            for (size_t i = 0; i < inputDynamicShapes.size(); i++) {
                ngraphParam.push_back(std::make_shared<ngraph::opset1::Parameter>(inputPrecisions[i], inputDynamicShapes[i]));
                ngraphInputs.push_back(ngraphParam.back());
            }
        } else {
            ngraphParam = ov::ParameterVector {std::make_shared<ov::op::v0::Parameter>(inputPrecisions[0], inputDynamicShapes.front())};
            for (size_t i = 1; i < inputPrecisions.size(); i++) {
                std::vector<float> ngraphInput1Data(ngraph::shape_size(targetStaticShapes[0][i]));
                ngraphInputs.push_back(ngraph::builder::makeConstant(inputPrecisions[i], targetStaticShapes[0][i],
                                                                     ngraphInput1Data, true));
            }
        }

        if (withQuantization) {
            std::vector<std::shared_ptr<ngraph::Node>> eltwiseOps;
            eltwiseOps.push_back(ngraph::builder::makeEltwise(ngraphParam[0], ngraphInputs[0], eltwiseOpTypes[0]));
            for (size_t i = 1; i < eltwiseOpTypes.size() - 1; i++) {
                eltwiseOps.push_back(ngraph::builder::makeEltwise(eltwiseOps[eltwiseOps.size() - 1], ngraphInputs[i], eltwiseOpTypes[i]));
            }

            std::vector<size_t> constShape(targetStaticShapes[0][0].size(), 1);
            constShape[1] = targetStaticShapes[0][0][1];
            auto fq = ngraph::builder::makeFakeQuantize(eltwiseOps[eltwiseOps.size() - 1],
                                                        ::ngraph::element::Type(::ngraph::element::Type_t::f32),
                                                        256, constShape);

            eltwiseOps.push_back(ngraph::builder::makeEltwise(fq, ngraphInputs[eltwiseOpTypes.size() - 1], eltwiseOpTypes[eltwiseOpTypes.size() - 1]));

            ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(eltwiseOps[eltwiseOps.size() - 1])};
            function = std::make_shared<ngraph::Function>(results, ngraphParam, "eltwise_chain_fq");
        } else {
            std::vector<std::shared_ptr<ngraph::Node>> eltwiseOps;
            eltwiseOps.push_back(ngraph::builder::makeEltwise(ngraphParam[0], ngraphInputs[0], eltwiseOpTypes[0]));
            for (size_t i = 1; i < eltwiseOpTypes.size(); i++) {
                eltwiseOps.push_back(ngraph::builder::makeEltwise(eltwiseOps[eltwiseOps.size() - 1], ngraphInputs[i], eltwiseOpTypes[i]));
            }

            ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(eltwiseOps[eltwiseOps.size() - 1])};
            function = std::make_shared<ngraph::Function>(results, ngraphParam, "eltwise_chain");
        }
    }
};

TEST_P(EltwiseChainTest, CompareWithRefs) {
    run();
}

namespace {

std::vector<std::vector<ngraph::Shape>> inputShapes = {
    {{1, 1, 2, 3}, {1, 1, 2, 3}, {1, 1, 2, 3}, {1, 1, 2, 3}},
    {{1, 48, 5, 6}, {1, 48, 1, 1}, {1, 48, 5, 6}, {1, 1, 5, 6}},
    {{1, 72, 28, 28}, {1, 72, 1, 1}, {1, 72, 1, 1}, {1, 72, 1, 1}},
    {{2, 33, 5, 5}, {2, 33, 5, 5}, {2, 33, 1, 5}, {2, 33, 5, 5}},
    {{1, 2, 3}, {3}, {3}, {3}},
    {{1, 12, 5, 5}, {5, 5}, {12, 5, 5}, {1}},
    {{3, 12, 5, 5}, {1, 12, 5, 1}, {3, 1, 1, 1}, {3, 12, 5, 5}},
    {{1, 1, 1, 1}, {1, 12, 5, 1}, {3, 12, 1, 5}, {3, 12, 5, 1}},
    {{1, 1, 1, 1, 6}, {1, 12, 5, 1, 6}, {3, 12, 1, 5, 1}, {3, 12, 5, 1, 1}}
};

std::vector<std::vector<ElementType>> inputPrecisions = {
        { ElementType::f32, ElementType::f32, ElementType::f32, ElementType::f32 },
        { ElementType::i32, ElementType::i32, ElementType::i32, ElementType::i32 }
};

std::vector<std::vector<EltwiseTypes>> eltwiseOps = {
        { EltwiseTypes::ADD, EltwiseTypes::MULTIPLY, EltwiseTypes::SUBTRACT },
        { EltwiseTypes::DIVIDE, EltwiseTypes::SQUARED_DIFF, EltwiseTypes::ADD },
};

INSTANTIATE_TEST_SUITE_P(smoke_EltwiseChain, EltwiseChainTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(static_shapes_to_test_representation(inputShapes)),
                                ::testing::Values(ngraph::helpers::InputLayerType::CONSTANT),
                                ::testing::ValuesIn(inputPrecisions),
                                ::testing::ValuesIn(eltwiseOps),
                                ::testing::Values(false),
                                ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        EltwiseChainTest::getTestCaseName);

std::vector<std::vector<ngraph::Shape>> inputShapesFQ = {
    {{1, 2, 2, 3}, {1, 2, 2, 3}, {1, 2, 2, 3}, {1, 2, 2, 3}},
    {{2, 33, 5, 5}, {2, 33, 5, 5}, {2, 33, 1, 5}, {2, 33, 5, 5}},
    {{2, 33, 5, 17}, {2, 33, 5, 17}, {2, 33, 5, 17}, {2, 33, 5, 17}},
    {{2, 33, 5, 256}, {2, 33, 5, 256}, {2, 33, 5, 256}, {2, 33, 5, 256}},
    {{2, 5, 7, 5}, {2, 5, 1, 5}, {2, 5, 7, 5}, {2, 5, 7, 5}},
    {{2, 17, 7, 5}, {2, 17, 7, 5}, {2, 17, 7, 5}, {2, 17, 7, 5}},
    {{2, 256, 7, 5}, {2, 256, 7, 5}, {2, 256, 1, 5}, {2, 256, 7, 5}},
    {{1, 36, 34, 34}, {1, 36, 34, 34}, {1, 36, 34, 34}, {1, 36, 34, 34}},
    {{1, 12, 1, 1, 6}, {1, 12, 5, 1, 6}, {3, 12, 1, 5, 1}, {3, 12, 5, 1, 1}},
    {{1, 12, 1, 1, 6}, {1, 12, 5, 5, 6}, {3, 12, 1, 5, 1}, {3, 12, 5, 5, 1}},
    {{1, 12, 1, 1, 1}, {1, 12, 5, 1, 7}, {3, 12, 1, 5, 7}, {3, 12, 5, 1, 7}},
    {{1, 7, 1, 1, 12}, {1, 7, 5, 1, 12}, {3, 7, 1, 5, 12}, {3, 7, 5, 1, 12}},
    {{1, 7, 1, 1, 12, 3, 7}, {1, 7, 5, 1, 12, 3, 7}, {3, 7, 1, 5, 12, 3, 7}, {3, 7, 5, 1, 12, 3, 7}},
    {{1, 7, 1, 1, 12, 3, 1}, {1, 7, 5, 1, 12, 3, 7}, {3, 7, 1, 5, 12, 1, 7}, {3, 7, 5, 1, 12, 3, 1}}
};

std::vector<std::vector<ElementType>> inputPrecisionsFQ {
        { ElementType::f32, ElementType::f32, ElementType::f32, ElementType::f32 }
};

INSTANTIATE_TEST_SUITE_P(smoke_EltwiseChainWithFQ, EltwiseChainTest,
                    ::testing::Combine(
                            ::testing::ValuesIn(static_shapes_to_test_representation(inputShapesFQ)),
                            ::testing::Values(ngraph::helpers::InputLayerType::CONSTANT),
                            ::testing::ValuesIn(inputPrecisionsFQ),
                            ::testing::ValuesIn(eltwiseOps),
                            ::testing::Values(true),
                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        EltwiseChainTest::getTestCaseName);

// =============================================== dynamic ==============================================
std::vector<std::vector<InputShape>> inputShapes_dyn = {
    {
        // inp1
        {
            // dynamic
            {-1, -1, -1},
            // target
            {
                {1, 2, 3},
                {5, 2, 7},
                {3, 1, 10},
            }
        },
        // inp2
        {
            // dynamic
            {-1},
            // target
            {
                {3}, {7}, {1},
            }
        },
        // inp3
        {
            // dynamic
            {-1},
            // target
            {
                {3}, {1}, {1}
            }
        },
        // inp4
        {
            // dynamic
            {-1},
            // target
            {
                {3}, {1}, {1}
            }
        }
    },
    {
        // inp1
        {
            // dynamic
            {-1, -1, -1, -1},
            // target
            {
                {1, 12, 5, 5},
                {5, 16, 1, 5},
                {2, 1, 1, 5},
            }
        },
        // inp2
        {
            // dynamic
            {-1, -1},
            // target
            {
                {5, 5}, {1, 5}, {5, 1},
            }
        },
        // inp3
        {
            // dynamic
            {-1, -1, -1},
            // target
            {
                {12, 5, 5},
                {1, 5, 1},
                {16, 5, 5},
            }
        },
        // inp4
        {
            // dynamic
            {-1},
            // target
            {
                {1}, {1}, {5}
            }
        }
    },
    {
        // inp1
        {
            // dynamic
            {-1, -1, -1, -1},
            // target
            {
                {1, 2, 2, 3},
                {2, 33, 5, 5},
                {2, 33, 5, 17},
                {2, 33, 5, 256},
                {2, 5, 7, 5},
                {2, 17, 7, 5},
                {2, 256, 7, 5},
                {1, 36, 34, 34},
            }
        },
        // inp2
        {
            // dynamic
            {-1, -1, -1, -1},
            // target
            {
                {1, 2, 2, 3},
                {2, 33, 5, 5},
                {2, 33, 5, 17},
                {2, 33, 5, 256},
                {2, 5, 1, 5},
                {2, 17, 7, 5},
                {2, 256, 7, 5},
                {1, 36, 34, 34},
            }
        },
        // inp3
        {
            // dynamic
            {-1, -1, -1, -1},
            // target
            {
                {1, 2, 2, 3},
                {2, 33, 1, 5},
                {2, 33, 5, 17},
                {2, 33, 5, 256},
                {2, 5, 7, 5},
                {2, 17, 7, 5},
                {2, 256, 1, 5},
                {1, 36, 34, 34}
            }
        },
        // inp4
        {
            // dynamic
            {-1, -1, -1, -1},
            // target
            {
                {1, 2, 2, 3},
                {2, 33, 5, 5},
                {2, 33, 5, 17},
                {2, 33, 5, 256},
                {2, 5, 7, 5},
                {2, 17, 7, 5},
                {2, 256, 7, 5},
                {1, 36, 34, 34}
            }
        }
    },
    {
        // inp1
        {
            // dynamic
            {-1, -1, -1, -1, -1},
            // target
            {
                {1, 12, 1, 1, 6},
                {1, 12, 1, 1, 6},
                {1, 12, 1, 1, 1},
                {1, 7, 1, 1, 12},
            }
        },
        // inp2
        {
            // dynamic
            {-1, -1, -1, -1, -1},
            // target
            {
                {1, 12, 5, 1, 6},
                {1, 12, 5, 5, 6},
                {1, 12, 5, 1, 7},
                {1, 7, 5, 1, 12},
            }
        },
        // inp3
        {
            // dynamic
            {-1, -1, -1, -1, -1},
            // target
            {
                {3, 12, 1, 5, 1},
                {3, 12, 1, 5, 1},
                {3, 12, 1, 5, 7},
                {3, 7, 1, 5, 12}
            }
        },
        // inp4
        {
            // dynamic
            {-1, -1, -1, -1, -1},
            // target
            {
                {3, 12, 5, 1, 1},
                {3, 12, 5, 5, 1},
                {3, 12, 5, 1, 7},
                {3, 7, 5, 1, 12}
            }
        }
    },
    {
        // inp1
        {
            // dynamic
            {-1, -1, -1, -1, -1, -1, -1},
            // target
            {
                {1, 7, 1, 1, 12, 3, 7},
                {1, 7, 1, 1, 12, 3, 1},
                {5, 7, 1, 2, 12, 1, 8},
            }
        },
        // inp2
        {
            // dynamic
            {-1, -1, -1, -1, -1, -1, -1},
            // target
            {
                {1, 7, 5, 1, 12, 3, 7},
                {1, 7, 5, 1, 12, 3, 7},
                {1, 7, 5, 1, 12, 3, 8},
            }
        },
        // inp3
        {
            // dynamic
            {-1, -1, -1, -1, -1, -1, -1},
            // target
            {
                {3, 7, 1, 5, 12, 3, 7},
                {3, 7, 1, 5, 12, 1, 7},
                {5, 1, 1, 2, 12, 1, 8},
            }
        },
        // inp4
        {
            // dynamic
            {-1, -1, -1, -1, -1, -1, -1},
            // target
            {
                {3, 7, 5, 1, 12, 3, 7},
                {3, 7, 5, 1, 12, 3, 1},
                {1, 7, 5, 1, 12, 3, 1}
            }
        }
    }
};

INSTANTIATE_TEST_SUITE_P(smoke_EltwiseChain_dyn, EltwiseChainTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inputShapes_dyn),
                                ::testing::Values(ngraph::helpers::InputLayerType::PARAMETER),
                                ::testing::ValuesIn(inputPrecisions),
                                ::testing::ValuesIn(eltwiseOps),
                                ::testing::Values(false),
                                ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        EltwiseChainTest::getTestCaseName);

} // namespace
} // namespace CPUSubgraphTestsDefinitions
