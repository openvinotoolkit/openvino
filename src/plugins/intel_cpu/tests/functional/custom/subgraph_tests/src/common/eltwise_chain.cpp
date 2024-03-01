// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <string>
#include <vector>
#include <memory>

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "common_test_utils/node_builders/constant.hpp"
#include "common_test_utils/node_builders/fake_quantize.hpp"
#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "utils/cpu_test_utils.hpp"
#include "common_test_utils/node_builders/eltwise.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {
using namespace ov::test::utils;

typedef std::tuple<std::vector<InputShape>,    // Input shapes
                   InputLayerType,             // Secondary input type
                   std::vector<ElementType>,   // Input precisions
                   std::vector<EltwiseTypes>,  // Eltwise operations
                   bool,                       // With quantization
                   ov::element::Type,          // Conversion type
                   std::string                 // Device name
                   >
    EltwiseChainTuple;

class EltwiseChainTest : public testing::WithParamInterface<EltwiseChainTuple>,
                         virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<EltwiseChainTuple> &obj) {
        std::vector<InputShape> inputShapes;
        InputLayerType secondaryInputType;
        std::vector<ElementType> inputPrecisions;
        std::vector<EltwiseTypes> eltwiseOpTypes;
        bool withQuantization;
        ov::element::Type conversion;
        std::string targetName;
        std::tie(inputShapes, secondaryInputType, inputPrecisions, eltwiseOpTypes, withQuantization, conversion, targetName) = obj.param;
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
        if (conversion == ov::element::undefined) {
            results << "Conversion" << conversion << "_";
        }
        results << "targetDevice=" << targetName;

        return results.str();
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        for (size_t i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            ov::Tensor tensor;
            ov::test::utils::InputGenerateData in_data;
            in_data.start_from = 1;
            in_data.range = 10;
            tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i], in_data);
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }

protected:
    void SetUp() override {
        abs_threshold = 0.1f;

        std::vector<InputShape> inputShapes;
        InputLayerType secondaryInputType;
        std::vector<ElementType> inputPrecisions;
        std::vector<EltwiseTypes> eltwiseOpTypes;
        bool withQuantization;
        ov::element::Type conversion;
        std::tie(inputShapes, secondaryInputType, inputPrecisions, eltwiseOpTypes, withQuantization, conversion, targetDevice) = this->GetParam();

        init_input_shapes(inputShapes);

        ov::ParameterVector paramVec;
        std::vector<std::shared_ptr<ov::Node>> inputNodes1;
        std::vector<std::shared_ptr<ov::Node>> inputNodes2;
        if (secondaryInputType == utils::InputLayerType::PARAMETER) {
            for (size_t i = 0; i < inputDynamicShapes.size(); i++) {
                const auto param = std::make_shared<ov::op::v0::Parameter>(inputPrecisions[i], inputDynamicShapes[i]);
                paramVec.push_back(param);

                const auto inputNode = (conversion == ov::element::undefined) ?
                    param :
                    std::dynamic_pointer_cast<ov::Node>(std::make_shared<ov::op::v0::Convert>(param, conversion));
                if (inputNodes1.empty()) {
                    inputNodes1.push_back(inputNode);
                }
                inputNodes2.push_back(inputNode);
            }
        } else {
            paramVec = ov::ParameterVector {std::make_shared<ov::op::v0::Parameter>(inputPrecisions[0], inputDynamicShapes.front())};
            inputNodes1.push_back(
                    conversion == ov::element::undefined ?
                    paramVec.front() :
                    std::dynamic_pointer_cast<ov::Node>(std::make_shared<ov::op::v0::Convert>(paramVec.front(), conversion)));

            for (size_t i = 1; i < inputPrecisions.size(); i++) {
                std::vector<float> input1Data(ov::shape_size(targetStaticShapes[0][i]));
                inputNodes2.push_back(ov::test::utils::deprecated::make_constant(
                    conversion == ov::element::undefined ? static_cast<ov::element::Type>(inputPrecisions[i]) : conversion,
                    targetStaticShapes[0][i],
                    input1Data,
                    true));
            }
        }

        if (withQuantization) {
            std::vector<std::shared_ptr<ov::Node>> eltwiseOps;
            eltwiseOps.push_back(make_eltwise(inputNodes1[0], inputNodes2[0], eltwiseOpTypes[0]));
            for (size_t i = 1; i < eltwiseOpTypes.size() - 1; i++) {
                eltwiseOps.push_back(make_eltwise(eltwiseOps[eltwiseOps.size() - 1], inputNodes2[i], eltwiseOpTypes[i]));
            }

            std::vector<size_t> constShape(targetStaticShapes[0][0].size(), 1);
            constShape[1] = targetStaticShapes[0][0][1];
            auto fq = ov::test::utils::make_fake_quantize(eltwiseOps[eltwiseOps.size() - 1],
                                                        ov::element::Type(ov::element::f32),
                                                        256,
                                                        constShape);

            eltwiseOps.push_back(make_eltwise(fq, inputNodes2[eltwiseOpTypes.size() - 1], eltwiseOpTypes[eltwiseOpTypes.size() - 1]));

            ov::ResultVector results{std::make_shared<ov::op::v0::Result>(eltwiseOps[eltwiseOps.size() - 1])};
            function = std::make_shared<ov::Model>(results, paramVec, "eltwise_chain_fq");
        } else {
            std::vector<std::shared_ptr<ov::Node>> eltwiseOps;
            eltwiseOps.push_back(make_eltwise(inputNodes1[0], inputNodes2[0], eltwiseOpTypes[0]));
            for (size_t i = 1; i < eltwiseOpTypes.size(); i++) {
                eltwiseOps.push_back(make_eltwise(eltwiseOps[eltwiseOps.size() - 1], inputNodes2[i], eltwiseOpTypes[i]));
            }

            ov::ResultVector results{std::make_shared<ov::op::v0::Result>(eltwiseOps[eltwiseOps.size() - 1])};
            function = std::make_shared<ov::Model>(results, paramVec, "eltwise_chain");
        }
    }
};

TEST_P(EltwiseChainTest, CompareWithRefs) {
    run();
}

namespace {

std::vector<std::vector<ov::Shape>> inputShapes = {
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
        { EltwiseTypes::DIVIDE, EltwiseTypes::SQUARED_DIFF, EltwiseTypes::ADD }
};

INSTANTIATE_TEST_SUITE_P(smoke_EltwiseChain, EltwiseChainTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(static_shapes_to_test_representation(inputShapes)),
                                ::testing::Values(InputLayerType::CONSTANT),
                                ::testing::ValuesIn(inputPrecisions),
                                ::testing::ValuesIn(eltwiseOps),
                                ::testing::Values(false),
                                ::testing::Values(ov::element::undefined),
                                ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        EltwiseChainTest::getTestCaseName);


    std::vector<std::vector<ov::Shape>> inputShapesConvert = {
            {{1, 1, 2, 3}, {1, 1, 2, 3}, {1, 1, 2, 3}}
    };

    std::vector<std::vector<ElementType>> inputPrecisionsConvert = {
            { ElementType::i8, ElementType::f32, ElementType::f32 },
            { ElementType::u8, ElementType::f32, ElementType::f32 },
            { ElementType::i16, ElementType::f32, ElementType::f32 },
            { ElementType::u16, ElementType::f32, ElementType::f32 },
            { ElementType::i32, ElementType::f32, ElementType::f32 },
            // { ElementType::u32, ElementType::f32, ElementType::f32 }, // plugin doesn't support
            { ElementType::f16, ElementType::f32, ElementType::f32 },
            { ElementType::f32, ElementType::f32, ElementType::f32 },
    };

    std::vector<std::vector<EltwiseTypes>> eltwiseOpsConvert = {
            { EltwiseTypes::MULTIPLY },
            { EltwiseTypes::ADD },
            { EltwiseTypes::DIVIDE },
            { EltwiseTypes::SUBTRACT },
            { EltwiseTypes::POWER },
    };

    INSTANTIATE_TEST_SUITE_P(smoke_EltwiseChain_MergeConvert, EltwiseChainTest,
                             ::testing::Combine(
                                     ::testing::ValuesIn(static_shapes_to_test_representation(inputShapesConvert)),
                                     ::testing::Values(InputLayerType::CONSTANT),
                                     ::testing::ValuesIn(inputPrecisionsConvert),
                                     ::testing::ValuesIn(eltwiseOpsConvert),
                                     ::testing::Values(false),
                                     ::testing::Values(ov::element::f32),
                                     ::testing::Values(ov::test::utils::DEVICE_CPU)),
                             EltwiseChainTest::getTestCaseName);

std::vector<std::vector<ov::Shape>> inputShapesFQ = {
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
                            ::testing::Values(InputLayerType::CONSTANT),
                            ::testing::ValuesIn(inputPrecisionsFQ),
                            ::testing::ValuesIn(eltwiseOps),
                            ::testing::Values(true),
                            ::testing::Values(ov::element::undefined),
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
                                ::testing::Values(InputLayerType::PARAMETER),
                                ::testing::ValuesIn(inputPrecisions),
                                ::testing::ValuesIn(eltwiseOps),
                                ::testing::Values(false),
                                ::testing::Values(ov::element::undefined),
                                ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        EltwiseChainTest::getTestCaseName);

}  // namespace
}  // namespace test
}  // namespace ov
