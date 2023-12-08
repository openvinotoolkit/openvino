// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/mat_mul.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "ie_precision.hpp"
#include "ov_models/builders.hpp"
#include <string>
#include "common_test_utils/ov_tensor_utils.hpp"

using namespace ngraph;
using namespace InferenceEngine;
using namespace ov::test;

namespace GPULayerTestsDefinitions {

struct ShapeRelatedParams {
    std::vector<InputShape> inputShapes;
    std::pair<bool, bool> transpose;
};

typedef std::tuple<
        ShapeRelatedParams,
        ElementType,                       // Network precision
        ElementType,                       // Input precision
        ElementType,                       // Output precision
        ngraph::helpers::InputLayerType,   // Secondary input type
        TargetDevice,                      // Device name
        std::map<std::string, std::string> // Additional network configuration
> MatMulLayerTestParamsSet;

class MatMulLayerGPUTest : public testing::WithParamInterface<MatMulLayerTestParamsSet>,
                           virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MatMulLayerTestParamsSet>& obj) {
        MatMulLayerTestParamsSet basicParamsSet = obj.param;

        ElementType netType;
        ElementType inType, outType;
        ShapeRelatedParams shapeRelatedParams;
        ngraph::helpers::InputLayerType secondaryInputType;
        TargetDevice targetDevice;
        std::map<std::string, std::string> additionalConfig;
        std::tie(shapeRelatedParams, netType, inType, outType, secondaryInputType, targetDevice, additionalConfig) =
                basicParamsSet;

        std::ostringstream result;
        result << "IS=";
        for (const auto& shape : shapeRelatedParams.inputShapes) {
            result << ov::test::utils::partialShape2str({shape.first}) << "_";
        }
        result << "TS=";
        for (const auto& shape : shapeRelatedParams.inputShapes) {
            result << "(";
            if (!shape.second.empty()) {
                auto itr = shape.second.begin();
                do {
                    result << ov::test::utils::vec2str(*itr);
                } while (++itr != shape.second.end() && result << "_");
            }
            result << ")_";
        }
        result << "transpose_a=" << shapeRelatedParams.transpose.first << "_";
        result << "transpose_b=" << shapeRelatedParams.transpose.second << "_";
        result << "secondaryInputType=" << secondaryInputType << "_";
        result << "netPRC=" << netType << "_";
        result << "inPRC=" << inType << "_";
        result << "outPRC=" << outType << "_";
        result << "trgDev=" << targetDevice;
        result << "config=(";
        for (const auto& configEntry : additionalConfig) {
            result << configEntry.first << ", " << configEntry.second << ":";
        }
        result << ")";

        return result.str();
    }

protected:
    template<typename T>
    void transpose(T& shape) {
        IE_ASSERT(shape.size() > 1);
        std::swap(*(shape.end() - 1), *(shape.end() - 2));
    }

    void SetUp() override {
        MatMulLayerTestParamsSet basicParamsSet = this->GetParam();

        ShapeRelatedParams shapeRelatedParams;
        ElementType netType;
        helpers::InputLayerType secondaryInputType;
        std::map<std::string, std::string> additionalConfig;

        std::tie(shapeRelatedParams, netType, inType, outType, secondaryInputType, targetDevice, additionalConfig) = basicParamsSet;

        init_input_shapes(shapeRelatedParams.inputShapes);

        bool transpA = shapeRelatedParams.transpose.first;
        bool transpB = shapeRelatedParams.transpose.second;

        if (transpA) {
            transpose(inputDynamicShapes[0]);
            for (auto& shapes : targetStaticShapes) {
                transpose(shapes[0]);
            }
        }
        if (transpB) {
            transpose(inputDynamicShapes[1]);
            for (auto& shapes : targetStaticShapes) {
                transpose(shapes[1]);
            }
        }

        const auto& inShapeA = inputDynamicShapes[0];
        const auto& inShapeB = inputDynamicShapes[1];

        configuration.insert(additionalConfig.begin(), additionalConfig.end());

        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(netType, inShapeA)};

        std::shared_ptr<ov::Node> matrixB;
        if (secondaryInputType == helpers::InputLayerType::PARAMETER) {
            auto param = std::make_shared<ov::op::v0::Parameter>(netType, inShapeB);
            matrixB = param;
            params.push_back(param);
        } else {
            ASSERT_TRUE(inShapeB.is_static());
            auto tensor = ov::test::utils::create_and_fill_tensor(netType, inShapeB.to_shape());
            matrixB = std::make_shared<ov::op::v0::Constant>(tensor);
        }

        auto matMul = std::make_shared<ov::op::v0::MatMul>(params[0], matrixB, transpA, transpB);
        auto makeFunction = [](const ngraph::element::Type &ngPrc, ngraph::ParameterVector &params, const std::shared_ptr<ngraph::Node> &lastNode) {
            ngraph::ResultVector results;

            for (size_t i = 0; i < lastNode->get_output_size(); i++)
                results.push_back(std::make_shared<ngraph::opset1::Result>(lastNode->output(i)));

            return std::make_shared<ngraph::Function>(results, params, "MatMul");
        };
        function = makeFunction(netType, params, matMul);
    }
};

TEST_P(MatMulLayerGPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    run();
}

namespace {

/* ============= Common params ============= */
std::map<std::string, std::string> emptyAdditionalConfig;

std::vector<std::map<std::string, std::string>> additionalConfig {
    std::map<std::string, std::string>{/* empty config */},
};

const std::vector<ElementType> netPRCs {
    ElementType::f32,
};


/* ============= FullyConnected ============= */
namespace fullyConnected {

const std::vector<ShapeRelatedParams> IS2D_smoke = {
    {static_shapes_to_test_representation({{59, 1}, {1, 120}}), {false, true}},
    {static_shapes_to_test_representation({{59, 1}, {1, 120}}), {true, true}},

    {static_shapes_to_test_representation({{59, 120}, {120, 1}}), {false, false}},
    {static_shapes_to_test_representation({{59, 120}, {120, 1}}), {true, true}},

    {static_shapes_to_test_representation({{1, 120}, {120, 59}}), {false, false}},
    {static_shapes_to_test_representation({{1, 120}, {120, 59}}), {true, false}},

    {static_shapes_to_test_representation({{71, 128}, {128, 20}}), {true, false}},
    {static_shapes_to_test_representation({{71, 128}, {128, 20}}), {false, true}},

    {
        {
            {{-1, -1}, {{20, 60}, {20, 60}}},
            {{60, 120}, {{60, 120}, {60, 120}}}
        },
        {false, false}
    },
    {
        {
            {{{0, 100}, {0, 12}}, {{20, 1}, {14, 1}, {20, 1}, {14, 1}}},
            {{1, 120}, {{1, 120}, {1, 120}, {1, 120}, {1, 120}}}
        },
        {true, true}
    }
};

const std::vector<ShapeRelatedParams> IS2D_nightly = {
    {static_shapes_to_test_representation({{59, 1}, {1, 120}}), {false, false}},
    {static_shapes_to_test_representation({{59, 1}, {1, 120}}), {true, false}},

    {static_shapes_to_test_representation({{59, 120}, {120, 1}}), {true, false}},
    {static_shapes_to_test_representation({{59, 120}, {120, 1}}), {false, true}},

    {static_shapes_to_test_representation({{1, 120}, {120, 59}}), {true, true}},
    {static_shapes_to_test_representation({{1, 120}, {120, 59}}), {false, true}},

    {static_shapes_to_test_representation({{71, 128}, {128, 20}}), {true, true}},
    {static_shapes_to_test_representation({{71, 128}, {128, 20}}), {false, false}},

    {
        {
            {{-1, -1}, {{71, 128}, {50, 128}}},
            {{128, 20}, {{128, 20}, {128, 20}}}
        },
        {false, false}
    },
    {
        {
            {{-1, 59}, {{10, 59}, {15, 59}, {15, 59}}},
            {{59, 1}, {{59, 1}, {59, 1}, {59, 1}}}
        },
        {true, false}
    },
    {
        {
            {{{0, 120}, 59}, {{5, 59}, {11, 59}, {5, 59}, {10, 59}}},
            {{59, 120}, {{59, 120}, {59, 120}, {59, 120}, {59, 120}}}
        },
        {false, true}
    }
};

const auto testParams2D_smoke = ::testing::Combine(::testing::ValuesIn(IS2D_smoke),
                                                                ::testing::Values(ElementType::f32),
                                                                ::testing::Values(ElementType::undefined),
                                                                ::testing::Values(ElementType::undefined),
                                                                ::testing::Values(helpers::InputLayerType::CONSTANT),
                                                                ::testing::Values(ov::test::utils::DEVICE_GPU),
                                                                ::testing::Values(emptyAdditionalConfig));

INSTANTIATE_TEST_SUITE_P(smoke_FC_2D, MatMulLayerGPUTest, testParams2D_smoke, MatMulLayerGPUTest::getTestCaseName);

const auto testParams2D_nightly = ::testing::Combine(::testing::ValuesIn(IS2D_nightly),
                                                                ::testing::Values(ElementType::f32),
                                                                ::testing::Values(ElementType::undefined),
                                                                ::testing::Values(ElementType::undefined),
                                                                ::testing::Values(helpers::InputLayerType::CONSTANT),
                                                                ::testing::Values(ov::test::utils::DEVICE_GPU),
                                                                ::testing::Values(emptyAdditionalConfig));

INSTANTIATE_TEST_SUITE_P(nightly_FC_2D, MatMulLayerGPUTest, testParams2D_nightly, MatMulLayerGPUTest::getTestCaseName);

const std::vector<ShapeRelatedParams> IS3D_smoke = {
    {static_shapes_to_test_representation({{1, 32, 120}, {120, 5}}), {false, false}},
    {static_shapes_to_test_representation({{1, 32, 120}, {120, 5}}), {false, true}},

    {static_shapes_to_test_representation({{1, 32, 120}, {120, 50}}), {true, false}},
    {static_shapes_to_test_representation({{1, 32, 120}, {120, 50}}), {false, true}},

    {
        {
            {{1, 5, 32}, {{1, 5, 32}, {1, 5, 32}}},
            {{32, 3}, {{32, 3}, {32, 3}}}
        },
        {false, true}
    },

    {static_shapes_to_test_representation({{1, 429}, {1, 429, 1}}), {true, true}},
    {
        {
            {{-1, -1}, {{1, 129}, {2, 129}, {1, 129}, {2, 129}}},
            {{1, 129, 1}, {{1, 129, 1}, {1, 129, 1}, {1, 129, 1}, {1, 129, 1}}}
        },
        {true, true}
    },

    {
        {
            {{{0, 60}, {0, 60}, {0, 60}}, {{1, 3, 14}, {1, 7, 14}}},
            {{14, 10}, {{14, 10}, {14, 10}}}
        },
        {true, true}
    }
};

const std::vector<ShapeRelatedParams> IS3D_nightly = {
    {static_shapes_to_test_representation({{1, 32, 120}, {120, 5}}), {true, false}},
    {static_shapes_to_test_representation({{1, 32, 120}, {120, 5}}), {true, true}},

    {static_shapes_to_test_representation({{1, 32, 120}, {120, 50}}), {false, false}},
    {static_shapes_to_test_representation({{1, 32, 120}, {120, 50}}), {true, true}},

    {
        {
            {{-1, -1, -1}, {{1, 32, 120}, {1, 12, 120}}},
            {{120, 3}, {{120, 3}, {120, 3}}}
        },
        {false, false}
    },
    {
        {
            {{-1, -1, 50}, {{1, 2, 50}, {1, 10, 50}, {1, 2, 50}, {2, 2, 50}}},
            {{50, 7}, {{50, 7}, {50, 7}, {50, 7}, {50, 7}}}
        },
        {true, false}
    },
    {
        {
            {{-1, -1, 32}, {{1, 5, 32}, {1, 5, 32}}},
            {{32, 3}, {{32, 3}, {32, 3}}}
        },
        {false, true}
    }
};

const auto fullyConnectedParams3D_smoke = ::testing::Combine(::testing::ValuesIn(IS3D_smoke),
                                                       ::testing::Values(ElementType::f32),
                                                       ::testing::Values(ElementType::undefined),
                                                       ::testing::Values(ElementType::undefined),
                                                       ::testing::Values(helpers::InputLayerType::CONSTANT),
                                                       ::testing::Values(ov::test::utils::DEVICE_GPU),
                                                       ::testing::Values(emptyAdditionalConfig));

INSTANTIATE_TEST_SUITE_P(smoke_FC_3D, MatMulLayerGPUTest, fullyConnectedParams3D_smoke, MatMulLayerGPUTest::getTestCaseName);

const auto fullyConnectedParams3D_nightly = ::testing::Combine(::testing::ValuesIn(IS3D_nightly),
                                                       ::testing::Values(ElementType::f32),
                                                       ::testing::Values(ElementType::undefined),
                                                       ::testing::Values(ElementType::undefined),
                                                       ::testing::Values(helpers::InputLayerType::CONSTANT),
                                                       ::testing::Values(ov::test::utils::DEVICE_GPU),
                                                       ::testing::Values(emptyAdditionalConfig));

INSTANTIATE_TEST_SUITE_P(nightly_FC_3D, MatMulLayerGPUTest, fullyConnectedParams3D_nightly, MatMulLayerGPUTest::getTestCaseName);

const std::vector<ShapeRelatedParams> IS4D_smoke = {
    {
        {
            {{-1, -1, -1, -1}, {{1, 32, 20, 120}, {1, 12, 20, 120}}},
            {{120, 3}, {{120, 3}, {120, 3}}}
        },
        {false, false}
    },
    {
        {
            {{-1, -1, -1, 50}, {{1, 1, 4, 50}, {1, 5, 10, 50}, {1, 2, 5, 50}, {2, 2, 2, 50}}},
            {{50, 7}, {{50, 7}, {50, 7}, {50, 7}, {50, 7}}}
        },
        {true, false}
    },
    {
        {
            {{-1, -1, -1, 32}, {{1, 1, 5, 32}, {1, 2, 5, 32}}},
            {{32, 3}, {{32, 3}, {32, 3}}}
        },
        {false, true}
    },
    {
        {
            {{{0, 60}, {0, 60}, {0, 60}, {0, 60}}, {{1, 3, 6, 14}, {1, 7, 10, 14}}},
            {{14, 10}, {{14, 10}, {14, 10}}}
        },
        {true, true}
    }
};

const auto fullyConnectedParams4D_smoke = ::testing::Combine(::testing::ValuesIn(IS4D_smoke),
                                                       ::testing::Values(ElementType::f32),
                                                       ::testing::Values(ElementType::undefined),
                                                       ::testing::Values(ElementType::undefined),
                                                       ::testing::Values(helpers::InputLayerType::CONSTANT),
                                                       ::testing::Values(ov::test::utils::DEVICE_GPU),
                                                       ::testing::Values(emptyAdditionalConfig));

INSTANTIATE_TEST_SUITE_P(smoke_FC_4D, MatMulLayerGPUTest, fullyConnectedParams4D_smoke, MatMulLayerGPUTest::getTestCaseName);

} // namespace fullyConnected

/* ============= MatMul ============= */
namespace matmul {

const std::vector<ShapeRelatedParams> IS = {
    {static_shapes_to_test_representation({{1, 2, 32, 120}, {120, 5}}), {false, false}},
    {static_shapes_to_test_representation({{1, 2, 32, 120}, {120, 5}}), {true, false}},
    {static_shapes_to_test_representation({{1, 2, 32, 120}, {120, 5}}), {false, true}},
    {static_shapes_to_test_representation({{1, 2, 32, 120}, {120, 5}}), {true, true}},

    {static_shapes_to_test_representation({{1, 2, 100010, 120}, {120, 5}}), {true, true}},
    {static_shapes_to_test_representation({{1, 2, 200010, 120}, {120, 5}}), {false, true}},
    {static_shapes_to_test_representation({{1, 2, 30, 120}, {120, 100010}}), {true, true}},
    {static_shapes_to_test_representation({{1, 2, 30, 120}, {120, 100010}}), {true, false}},

    {static_shapes_to_test_representation({{7, 32, 120}, {3, 7, 120, 50}}), {false, false}},
    {static_shapes_to_test_representation({{7, 32, 120}, {3, 7, 120, 50}}), {true, false}},
    {static_shapes_to_test_representation({{7, 32, 120}, {3, 7, 120, 50}}), {false, true}},
    {static_shapes_to_test_representation({{7, 32, 120}, {3, 7, 120, 50}}), {true, true}},

    {static_shapes_to_test_representation({{10, 10, 10}, {10, 10, 10}}), {false, false}},
    {static_shapes_to_test_representation({{10, 10, 10}, {10, 10, 10}}), {true, false}},
    {static_shapes_to_test_representation({{10, 10, 10}, {10, 10, 10}}), {false, true}},
    {static_shapes_to_test_representation({{10, 10, 10}, {10, 10, 10}}), {true, true}},

    {static_shapes_to_test_representation({{55, 12}, {12, 55}}), {false, false}},
    {static_shapes_to_test_representation({{55, 12}, {12, 55}}), {true, false}},
    {static_shapes_to_test_representation({{55, 12}, {12, 55}}), {false, true}},
    {static_shapes_to_test_representation({{55, 12}, {12, 55}}), {true, true}}
};

const std::vector<ShapeRelatedParams> IS_OneDNN = {
    {static_shapes_to_test_representation({{2, 4, 32, 120}, {2, 4, 120, 5}}), {false, false}},
    {static_shapes_to_test_representation({{2, 4, 32, 120}, {2, 4, 120, 5}}), {true, false}},
    {static_shapes_to_test_representation({{2, 4, 32, 120}, {2, 4, 120, 5}}), {false, true}},
    {static_shapes_to_test_representation({{2, 4, 32, 120}, {2, 4, 120, 5}}), {true, true}},

    {static_shapes_to_test_representation({{2, 2, 32, 120}, {1, 1, 120, 5}}), {false, false}},
    {static_shapes_to_test_representation({{2, 2, 32, 120}, {1, 1, 120, 5}}), {true, false}},
    {static_shapes_to_test_representation({{2, 2, 32, 120}, {1, 1, 120, 5}}), {false, true}},
    {static_shapes_to_test_representation({{2, 2, 32, 120}, {1, 1, 120, 5}}), {true, true}},

    {static_shapes_to_test_representation({{12, 12}, {12, 12}}), {false, false}},
    {static_shapes_to_test_representation({{12, 12}, {12, 12}}), {true, false}},
    {static_shapes_to_test_representation({{12, 12}, {12, 12}}), {false, true}},
    {static_shapes_to_test_representation({{12, 12}, {12, 12}}), {true, true}}
};

const std::vector<ShapeRelatedParams> IS_Dynamic = {
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{-1, -1}, {{55, 12}, {33, 7}}}, // input 0
            {{-1, -1}, {{12, 55}, {7, 33}}}  // input 1
        },
        {false, false}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{-1, -1}, {{55, 12}, {33, 7}}}, // input 0
            {{-1, -1}, {{12, 55}, {7, 33}}} // input 1
        },
        {true, false}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{-1, -1}, {{55, 12}, {33, 7}}}, // input 0
            {{-1, -1}, {{12, 55}, {7, 33}}} // input 1
        },
        {false, true}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{-1, -1}, {{55, 12}, {33, 7}}}, // input 0
            {{-1, -1}, {{12, 55}, {7, 33}}} // input 1
        },
        {true, true}
    },

    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{-1, -1, -1, -1}, {{1, 2, 32, 60}, {1, 2, 32, 30}}}, // input 0
            {{-1, -1}, {{60, 5}, {30, 5}}}  // input 1
        },
        {false, false}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{-1, -1, -1, -1}, {{1, 2, 32, 60}, {1, 2, 32, 30}}}, // input 0
            {{-1, -1}, {{60, 5}, {30, 5}}} // input 1
        },
        {true, false}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{-1, -1, -1, -1}, {{1, 2, 32, 60}, {1, 2, 32, 30}}}, // input 0
            {{-1, -1}, {{60, 5}, {30, 5}}} // input 1
        },
        {false, true}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{-1, -1, -1, -1}, {{1, 2, 32, 60}, {1, 2, 32, 30}}}, // input 0
            {{-1, -1}, {{60, 5}, {30, 5}}} // input 1
        },
        {true, true}
    },

    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{-1, -1, -1}, {{7, 32, 60}, {7, 32, 30}}}, // input 0
            {{-1, -1, -1, -1}, {{3, 7, 60, 25}, {3, 7, 30, 25}}}  // input 1
        },
        {false, false}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{-1, -1, -1}, {{7, 32, 60}, {7, 32, 30}}}, // input 0
            {{-1, -1, -1, -1}, {{3, 7, 60, 25}, {3, 7, 30, 25}}} // input 1
        },
        {true, false}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{-1, -1, -1}, {{7, 32, 60}, {7, 32, 30}}}, // input 0
            {{-1, -1, -1, -1}, {{3, 7, 60, 25}, {3, 7, 30, 25}}} // input 1
        },
        {false, true}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{-1, -1, -1}, {{7, 32, 60}, {7, 32, 30}}}, // input 0
            {{-1, -1, -1, -1}, {{3, 7, 60, 25}, {3, 7, 30, 25}}} // input 1
        },
        {true, true}
    },

    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{-1, -1, -1}, {{10, 10, 10}, {5, 5, 5}}}, // input 0
            {{-1, -1, -1}, {{10, 10, 10}, {5, 5, 5}}}  // input 1
        },
        {false, false}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{-1, -1, -1}, {{10, 10, 10}, {5, 5, 5}}}, // input 0
            {{-1, -1, -1}, {{10, 10, 10}, {5, 5, 5}}} // input 1
        },
        {true, false}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{-1, -1, -1}, {{10, 10, 10}, {5, 5, 5}}}, // input 0
            {{-1, -1, -1}, {{10, 10, 10}, {5, 5, 5}}} // input 1
        },
        {false, true}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{-1, -1, -1}, {{10, 10, 10}, {5, 5, 5}}}, // input 0
            {{-1, -1, -1}, {{10, 10, 10}, {5, 5, 5}}} // input 1
        },
        {true, true}
    },

    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{{1, 15}, {1, 15}, {1, 15}}, {{10, 10, 10}, {5, 5, 5}}}, // input 0
            {{{1, 15}, {1, 15}, {1, 15}}, {{10, 10, 10}, {5, 5, 5}}}  // input 1
        },
        {false, false}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{{1, 15}, {1, 15}, {1, 15}}, {{10, 10, 10}, {5, 5, 5}}}, // input 0
            {{{1, 15}, {1, 15}, {1, 15}}, {{10, 10, 10}, {5, 5, 5}}} // input 1
        },
        {true, false}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{{1, 15}, {1, 15}, {1, 15}}, {{10, 10, 10}, {5, 5, 5}}}, // input 0
            {{{1, 15}, {1, 15}, {1, 15}}, {{10, 10, 10}, {5, 5, 5}}} // input 1
        },
        {false, true}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{ -1, 16 }, {{ 4, 16 }, { 2, 16 }}}, // input 0
            {{ {1, 5}, 12, -1, 4 }, {{ 1, 12, 16, 4 }, { 1, 12, 16, 4 }}}  // input 1
        },
        {true, true}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{ -1, 12, -1, 16 }, {{ 1, 12, 4, 16 }, { 2, 12, 2, 16 }}}, // input 0
            {{ {1, 5}, 12, -1, 4 }, {{ 1, 12, 16, 4 }, { 1, 12, 16, 4 }}}  // input 1
        },
        {false, false}
    }
};

const std::vector<ShapeRelatedParams> IS_Dynamic_nightly = {
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{{5, 15}, {1, 12}, {4, 15}}, {{10, 10, 10}, {5, 5, 5}}}, // input 0
            {{{1, 13}, {3, 15}, {1, 10}}, {{10, 10, 10}, {5, 5, 5}}} // input 1
        },
        {true, true}
    },

    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{ {2, 10}, {3, 15}, -1, 16 }, {{ 2, 12, 4, 16 }, { 3, 12, 2, 16 }}}, // input 0
            {{ 1, 1, -1, 4 }, {{ 1, 1, 16, 4 }, { 1, 1, 16, 4 }}}  // input 1
        },
        {true, true}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{ 1, 1, -1, 16 }, {{ 1, 1, 4, 16 }, { 1, 1, 2, 16 }}}, // input 0
            {{ {2, 5}, {3, 15}, -1, 4 }, {{ 2, 12, 16, 4 }, { 2, 12, 16, 4 }}}  // input 1
        },
        {false, false}
    },

    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{ -1, 16 }, {{ 4, 16 }, { 2, 16 }}}, // input 0
            {{ {1, 5}, 12, -1, 4 }, {{ 1, 12, 16, 4 }, { 1, 12, 16, 4 }}}  // input 1
        },
        {false, false}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{ -1, {2, 15}, -1, 16 }, {{ 1, 12, 4, 16 }, { 2, 12, 2, 16 }}}, // input 0
            {{ -1, 4 }, {{ 16, 4 }, { 16, 4 }}}  // input 1
        },
        {true, true}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{ -1, {1, 15}, -1, 16 }, {{ 1, 12, 4, 16 }, { 2, 12, 2, 16 }}}, // input 0
            {{ -1, 4 }, {{ 16, 4 }, { 16, 4 }}}  // input 1
        },
        {false, false}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{ {1, 3}, {1, 9}, {1, 5}, {1, 10} }, {{ 1, 7, 4, 5 }, { 1, 7, 4, 4 }}}, // input 0
            {{ {1, 5}, {1, 7}, {1, 8}, {1, 5} }, {{ 1, 7, 5, 4 }, { 1, 7, 4, 4 }}}  // input 1
        },
        {true, true}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{ {1, 3}, {1, 9}, {1, 5}, {1, 10} }, {{ 1, 7, 4, 5 }, { 1, 7, 4, 4 }}}, // input 0
            {{ {1, 5}, {1, 7}, {1, 8}, {1, 5} }, {{ 1, 7, 5, 4 }, { 1, 7, 4, 4 }}}  // input 1
        },
        {false, false}
    },

    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{ 1, 7, 4, -1 }, {{ 1, 7, 4, 5 }, { 1, 7, 4, 4 }}}, // input 0
            {{ 1, 7, -1, 4 }, {{ 1, 7, 5, 4 }, { 1, 7, 4, 4 }}}  // input 1
        },
        {true, true}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{ 1, 7, 4, -1 }, {{ 1, 7, 4, 5 }, { 1, 7, 4, 4 }}}, // input 0
            {{ 1, 7, -1, 4 }, {{ 1, 7, 5, 4 }, { 1, 7, 4, 4 }}}  // input 1
        },
        {false, false}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{ -1, 12, -1, 16 }, {{ 1, 12, 4, 16 }, { 2, 12, 2, 16 }}}, // input 0
            {{ {1, 5}, 12, -1, 4 }, {{ 1, 12, 16, 4 }, { 1, 12, 16, 4 }}}  // input 1
        },
        {true, true}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{ -1, 12, -1, 16 }, {{ 1, 12, 4, 16 }, { 2, 12, 2, 16 }}}, // input 0
            {{ {1, 5}, 12, -1, 4 }, {{ 1, 12, 16, 4 }, { 1, 12, 16, 4 }}}  // input 1
        },
        {true, false}
    },

    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{ -1, 12, -1, 16 }, {{ 1, 12, 4, 16 }, { 2, 12, 2, 16 }}}, // input 0
            {{ {1, 5}, 12, -1, 4 }, {{ 1, 12, 16, 4 }, { 1, 12, 16, 4 }}}  // input 1
        },
        {false, true}
    }
};

const auto testParams = ::testing::Combine(::testing::ValuesIn(IS),
                                           ::testing::ValuesIn(netPRCs),
                                           ::testing::Values(ElementType::undefined),
                                           ::testing::Values(ElementType::undefined),
                                           ::testing::Values(helpers::InputLayerType::PARAMETER),
                                           ::testing::Values(ov::test::utils::DEVICE_GPU),
                                           ::testing::ValuesIn(additionalConfig));

INSTANTIATE_TEST_SUITE_P(smoke_MM_Static, MatMulLayerGPUTest, testParams, MatMulLayerGPUTest::getTestCaseName);

const auto testParamsOneDNN = ::testing::Combine(::testing::ValuesIn(IS_OneDNN),
                                                 ::testing::Values(ElementType::f16),
                                                 ::testing::Values(ElementType::undefined),
                                                 ::testing::Values(ElementType::undefined),
                                                 ::testing::Values(helpers::InputLayerType::PARAMETER),
                                                 ::testing::Values(ov::test::utils::DEVICE_GPU),
                                                 ::testing::ValuesIn(additionalConfig));

INSTANTIATE_TEST_SUITE_P(smoke_MM_Static_OneDNN, MatMulLayerGPUTest, testParamsOneDNN, MatMulLayerGPUTest::getTestCaseName);

const auto testParamsDynamic = ::testing::Combine(::testing::ValuesIn(IS_Dynamic),
                                                  ::testing::ValuesIn(netPRCs),
                                                  ::testing::Values(ElementType::undefined),
                                                  ::testing::Values(ElementType::undefined),
                                                  ::testing::Values(helpers::InputLayerType::PARAMETER),
                                                  ::testing::Values(ov::test::utils::DEVICE_GPU),
                                                  ::testing::ValuesIn(additionalConfig));

INSTANTIATE_TEST_SUITE_P(smoke_MM_Dynamic, MatMulLayerGPUTest, testParamsDynamic, MatMulLayerGPUTest::getTestCaseName);

const auto testParamsDynamic_nightly = ::testing::Combine(::testing::ValuesIn(IS_Dynamic_nightly),
                                             ::testing::ValuesIn(netPRCs),
                                             ::testing::Values(ElementType::undefined),
                                             ::testing::Values(ElementType::undefined),
                                             ::testing::Values(helpers::InputLayerType::PARAMETER),
                                             ::testing::Values(ov::test::utils::DEVICE_GPU),
                                             ::testing::ValuesIn(additionalConfig));

INSTANTIATE_TEST_SUITE_P(nightly_MM_Dynamic, MatMulLayerGPUTest, testParamsDynamic_nightly, MatMulLayerGPUTest::getTestCaseName);

} // namespace matmul
} // namespace
} // namespace GPULayerTestsDefinitions
