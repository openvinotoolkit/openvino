// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/test_enums.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/matmul.hpp"

namespace {
using ov::test::InputShape;

struct ShapeRelatedParams {
    std::vector<InputShape> inputShapes;
    std::pair<bool, bool> transpose;
};

typedef std::tuple<
        ShapeRelatedParams,
        ov::element::Type,                       // Network precision
        ov::element::Type,                       // Input precision
        ov::element::Type,                       // Output precision
        ov::test::utils::InputLayerType,         // Secondary input type
        std::string,                             // Device name
        std::map<std::string, std::string> // Additional network configuration
> MatMulLayerTestParamsSet;

class MatMulLayerGPUTest : public testing::WithParamInterface<MatMulLayerTestParamsSet>,
                           virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MatMulLayerTestParamsSet>& obj) {
        MatMulLayerTestParamsSet basicParamsSet = obj.param;

        ov::element::Type model_type;
        ov::element::Type inType, outType;
        ShapeRelatedParams shape_related_params;
        ov::test::utils::InputLayerType secondary_input_type;
        std::string targetDevice;
        std::map<std::string, std::string> additional_config;
        std::tie(shape_related_params, model_type, inType, outType, secondary_input_type, targetDevice, additional_config) =
                basicParamsSet;

        std::ostringstream result;
        result << "IS=";
        for (const auto& shape : shape_related_params.inputShapes) {
            result << ov::test::utils::partialShape2str({shape.first}) << "_";
        }
        result << "TS=";
        for (const auto& shape : shape_related_params.inputShapes) {
            result << "(";
            if (!shape.second.empty()) {
                auto itr = shape.second.begin();
                do {
                    result << ov::test::utils::vec2str(*itr);
                } while (++itr != shape.second.end() && result << "_");
            }
            result << ")_";
        }
        result << "transpose_a=" << shape_related_params.transpose.first << "_";
        result << "transpose_b=" << shape_related_params.transpose.second << "_";
        result << "secondary_input_type=" << secondary_input_type << "_";
        result << "netPRC=" << model_type << "_";
        result << "inPRC=" << inType << "_";
        result << "outPRC=" << outType << "_";
        result << "trgDev=" << targetDevice;
        result << "config=(";
        for (const auto& configEntry : additional_config) {
            result << configEntry.first << ", " << configEntry.second << ":";
        }
        result << ")";

        return result.str();
    }

protected:
    template<typename T>
    void transpose(T& shape) {
        OPENVINO_ASSERT(shape.size() > 1);
        std::swap(*(shape.end() - 1), *(shape.end() - 2));
    }

    void SetUp() override {
        MatMulLayerTestParamsSet basicParamsSet = this->GetParam();

        ShapeRelatedParams shape_related_params;
        ov::element::Type model_type;
        ov::test::utils::InputLayerType secondary_input_type;
        std::map<std::string, std::string> additional_config;

        std::tie(shape_related_params, model_type, inType, outType, secondary_input_type, targetDevice, additional_config) = basicParamsSet;

        init_input_shapes(shape_related_params.inputShapes);

        bool transpA = shape_related_params.transpose.first;
        bool transpB = shape_related_params.transpose.second;

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

        configuration.insert(additional_config.begin(), additional_config.end());

        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(model_type, inShapeA)};

        std::shared_ptr<ov::Node> matrixB;
        if (secondary_input_type == ov::test::utils::InputLayerType::PARAMETER) {
            auto param = std::make_shared<ov::op::v0::Parameter>(model_type, inShapeB);
            matrixB = param;
            params.push_back(param);
        } else {
            ASSERT_TRUE(inShapeB.is_static());
            auto tensor = ov::test::utils::create_and_fill_tensor(model_type, inShapeB.to_shape());
            matrixB = std::make_shared<ov::op::v0::Constant>(tensor);
        }

        auto matMul = std::make_shared<ov::op::v0::MatMul>(params[0], matrixB, transpA, transpB);
        auto makeFunction = [](const ov::element::Type &ngPrc, ov::ParameterVector &params, const std::shared_ptr<ov::Node> &lastNode) {
            ov::ResultVector results;

            for (size_t i = 0; i < lastNode->get_output_size(); i++)
                results.push_back(std::make_shared<ov::op::v0::Result>(lastNode->output(i)));

            return std::make_shared<ov::Model>(results, params, "MatMul");
        };
        function = makeFunction(model_type, params, matMul);
    }
};

TEST_P(MatMulLayerGPUTest, Inference) {
    run();
}

/* ============= Common params ============= */
std::map<std::string, std::string> emptyAdditionalConfig;

std::vector<std::map<std::string, std::string>> additional_config {
    std::map<std::string, std::string>{/* empty config */},
};

const std::vector<ov::element::Type> netPRCs {
    ov::element::f32,
};

const std::vector<ov::element::Type> netPRCs_f32_i32 {
    ov::element::f32,
    ov::element::i32
};


/* ============= FullyConnected ============= */

const std::vector<ShapeRelatedParams> IS2D_smoke = {
    {ov::test::static_shapes_to_test_representation({{59, 1}, {1, 120}}), {false, true}},
    {ov::test::static_shapes_to_test_representation({{59, 1}, {1, 120}}), {true, true}},

    {ov::test::static_shapes_to_test_representation({{59, 120}, {120, 1}}), {false, false}},
    {ov::test::static_shapes_to_test_representation({{59, 120}, {120, 1}}), {true, true}},

    {ov::test::static_shapes_to_test_representation({{1, 120}, {120, 59}}), {false, false}},
    {ov::test::static_shapes_to_test_representation({{1, 120}, {120, 59}}), {true, false}},

    {ov::test::static_shapes_to_test_representation({{71, 128}, {128, 20}}), {true, false}},
    {ov::test::static_shapes_to_test_representation({{71, 128}, {128, 20}}), {false, true}},

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
    {ov::test::static_shapes_to_test_representation({{59, 1}, {1, 120}}), {false, false}},
    {ov::test::static_shapes_to_test_representation({{59, 1}, {1, 120}}), {true, false}},

    {ov::test::static_shapes_to_test_representation({{59, 120}, {120, 1}}), {true, false}},
    {ov::test::static_shapes_to_test_representation({{59, 120}, {120, 1}}), {false, true}},

    {ov::test::static_shapes_to_test_representation({{1, 120}, {120, 59}}), {true, true}},
    {ov::test::static_shapes_to_test_representation({{1, 120}, {120, 59}}), {false, true}},

    {ov::test::static_shapes_to_test_representation({{71, 128}, {128, 20}}), {true, true}},
    {ov::test::static_shapes_to_test_representation({{71, 128}, {128, 20}}), {false, false}},

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
                                                   ::testing::Values(ov::element::f32),
                                                   ::testing::Values(ov::element::dynamic),
                                                   ::testing::Values(ov::element::dynamic),
                                                   ::testing::Values(ov::test::utils::InputLayerType::CONSTANT),
                                                   ::testing::Values(ov::test::utils::DEVICE_GPU),
                                                   ::testing::Values(emptyAdditionalConfig));

INSTANTIATE_TEST_SUITE_P(smoke_FC_2D, MatMulLayerGPUTest, testParams2D_smoke, MatMulLayerGPUTest::getTestCaseName);

const auto testParams2D_nightly = ::testing::Combine(::testing::ValuesIn(IS2D_nightly),
                                                     ::testing::Values(ov::element::f32),
                                                     ::testing::Values(ov::element::dynamic),
                                                     ::testing::Values(ov::element::dynamic),
                                                     ::testing::Values(ov::test::utils::InputLayerType::CONSTANT),
                                                     ::testing::Values(ov::test::utils::DEVICE_GPU),
                                                     ::testing::Values(emptyAdditionalConfig));

INSTANTIATE_TEST_SUITE_P(nightly_FC_2D, MatMulLayerGPUTest, testParams2D_nightly, MatMulLayerGPUTest::getTestCaseName);

const std::vector<ShapeRelatedParams> IS3D_smoke = {
    {ov::test::static_shapes_to_test_representation({{1, 32, 120}, {120, 5}}), {false, false}},
    {ov::test::static_shapes_to_test_representation({{1, 32, 120}, {120, 5}}), {false, true}},

    {ov::test::static_shapes_to_test_representation({{1, 32, 120}, {120, 50}}), {true, false}},
    {ov::test::static_shapes_to_test_representation({{1, 32, 120}, {120, 50}}), {false, true}},

    {
        {
            {{1, 5, 32}, {{1, 5, 32}, {1, 5, 32}}},
            {{32, 3}, {{32, 3}, {32, 3}}}
        },
        {false, true}
    },

    {ov::test::static_shapes_to_test_representation({{1, 429}, {1, 429, 1}}), {true, true}},

    {
        {
            {{-1, -1, -1}, {{1, 1, 129}, {1, 2, 129}, {1, 1, 129}, {1, 2, 129}}},
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
    {ov::test::static_shapes_to_test_representation({{1, 32, 120}, {120, 5}}), {true, false}},
    {ov::test::static_shapes_to_test_representation({{1, 32, 120}, {120, 5}}), {true, true}},

    {ov::test::static_shapes_to_test_representation({{1, 32, 120}, {120, 50}}), {false, false}},
    {ov::test::static_shapes_to_test_representation({{1, 32, 120}, {120, 50}}), {true, true}},

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

const auto fullyConnectedParams3D_smoke =
    ::testing::Combine(::testing::ValuesIn(IS3D_smoke),
                       ::testing::ValuesIn(netPRCs_f32_i32),
                       ::testing::Values(ov::element::dynamic),
                       ::testing::Values(ov::element::dynamic),
                       ::testing::Values(ov::test::utils::InputLayerType::CONSTANT),
                       ::testing::Values(ov::test::utils::DEVICE_GPU),
                       ::testing::Values(emptyAdditionalConfig));

INSTANTIATE_TEST_SUITE_P(smoke_FC_3D, MatMulLayerGPUTest, fullyConnectedParams3D_smoke, MatMulLayerGPUTest::getTestCaseName);

const auto fullyConnectedParams3D_nightly =
    ::testing::Combine(::testing::ValuesIn(IS3D_nightly),
                       ::testing::Values(ov::element::f32),
                       ::testing::Values(ov::element::dynamic),
                       ::testing::Values(ov::element::dynamic),
                       ::testing::Values(ov::test::utils::InputLayerType::CONSTANT),
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

const auto fullyConnectedParams4D_smoke =
    ::testing::Combine(::testing::ValuesIn(IS4D_smoke),
                       ::testing::Values(ov::element::f32),
                       ::testing::Values(ov::element::dynamic),
                       ::testing::Values(ov::element::dynamic),
                       ::testing::Values(ov::test::utils::InputLayerType::CONSTANT),
                       ::testing::Values(ov::test::utils::DEVICE_GPU),
                       ::testing::Values(emptyAdditionalConfig));

INSTANTIATE_TEST_SUITE_P(smoke_FC_4D, MatMulLayerGPUTest, fullyConnectedParams4D_smoke, MatMulLayerGPUTest::getTestCaseName);

/* ============= MatMul ============= */

const std::vector<ShapeRelatedParams> IS = {
    {ov::test::static_shapes_to_test_representation({{1, 2, 32, 120}, {120, 5}}), {false, false}},
    {ov::test::static_shapes_to_test_representation({{1, 2, 32, 120}, {120, 5}}), {true, false}},
    {ov::test::static_shapes_to_test_representation({{1, 2, 32, 120}, {120, 5}}), {false, true}},
    {ov::test::static_shapes_to_test_representation({{1, 2, 32, 120}, {120, 5}}), {true, true}},

    {ov::test::static_shapes_to_test_representation({{1, 2, 100010, 120}, {120, 5}}), {true, true}},
    {ov::test::static_shapes_to_test_representation({{1, 2, 200010, 120}, {120, 5}}), {false, true}},
    {ov::test::static_shapes_to_test_representation({{1, 2, 30, 120}, {120, 100010}}), {true, true}},
    {ov::test::static_shapes_to_test_representation({{1, 2, 30, 120}, {120, 100010}}), {true, false}},

    {ov::test::static_shapes_to_test_representation({{7, 32, 120}, {3, 7, 120, 50}}), {false, false}},
    {ov::test::static_shapes_to_test_representation({{7, 32, 120}, {3, 7, 120, 50}}), {true, false}},
    {ov::test::static_shapes_to_test_representation({{7, 32, 120}, {3, 7, 120, 50}}), {false, true}},
    {ov::test::static_shapes_to_test_representation({{7, 32, 120}, {3, 7, 120, 50}}), {true, true}},

    {ov::test::static_shapes_to_test_representation({{10, 10, 10}, {10, 10, 10}}), {false, false}},
    {ov::test::static_shapes_to_test_representation({{10, 10, 10}, {10, 10, 10}}), {true, false}},
    {ov::test::static_shapes_to_test_representation({{10, 10, 10}, {10, 10, 10}}), {false, true}},
    {ov::test::static_shapes_to_test_representation({{10, 10, 10}, {10, 10, 10}}), {true, true}},

    {ov::test::static_shapes_to_test_representation({{55, 12}, {12, 55}}), {false, false}},
    {ov::test::static_shapes_to_test_representation({{55, 12}, {12, 55}}), {true, false}},
    {ov::test::static_shapes_to_test_representation({{55, 12}, {12, 55}}), {false, true}},
    {ov::test::static_shapes_to_test_representation({{55, 12}, {12, 55}}), {true, true}}
};

const std::vector<ShapeRelatedParams> IS_OneDNN = {
    {ov::test::static_shapes_to_test_representation({{2, 4, 32, 120}, {2, 4, 120, 5}}), {false, false}},
    {ov::test::static_shapes_to_test_representation({{2, 4, 32, 120}, {2, 4, 120, 5}}), {true, false}},
    {ov::test::static_shapes_to_test_representation({{2, 4, 32, 120}, {2, 4, 120, 5}}), {false, true}},
    {ov::test::static_shapes_to_test_representation({{2, 4, 32, 120}, {2, 4, 120, 5}}), {true, true}},

    {ov::test::static_shapes_to_test_representation({{2, 2, 32, 120}, {1, 1, 120, 5}}), {false, false}},
    {ov::test::static_shapes_to_test_representation({{2, 2, 32, 120}, {1, 1, 120, 5}}), {true, false}},
    {ov::test::static_shapes_to_test_representation({{2, 2, 32, 120}, {1, 1, 120, 5}}), {false, true}},
    {ov::test::static_shapes_to_test_representation({{2, 2, 32, 120}, {1, 1, 120, 5}}), {true, true}},

    {ov::test::static_shapes_to_test_representation({{12, 12}, {12, 12}}), {false, false}},
    {ov::test::static_shapes_to_test_representation({{12, 12}, {12, 12}}), {true, false}},
    {ov::test::static_shapes_to_test_representation({{12, 12}, {12, 12}}), {false, true}},
    {ov::test::static_shapes_to_test_representation({{12, 12}, {12, 12}}), {true, true}}
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
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{}, {{64, 64}}}, // input 0
            {{-1, 128, 33, 64, 1}, {{1, 128, 33, 64, 1}}}  // input 1
        },
        {false, false}
    },
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
                                           ::testing::Values(ov::element::dynamic),
                                           ::testing::Values(ov::element::dynamic),
                                           ::testing::Values(ov::test::utils::InputLayerType::PARAMETER),
                                           ::testing::Values(ov::test::utils::DEVICE_GPU),
                                           ::testing::ValuesIn(additional_config));

INSTANTIATE_TEST_SUITE_P(smoke_MM_Static, MatMulLayerGPUTest, testParams, MatMulLayerGPUTest::getTestCaseName);

const auto testParamsOneDNN = ::testing::Combine(::testing::ValuesIn(IS_OneDNN),
                                                 ::testing::Values(ov::element::f16),
                                                 ::testing::Values(ov::element::dynamic),
                                                 ::testing::Values(ov::element::dynamic),
                                                 ::testing::Values(ov::test::utils::InputLayerType::PARAMETER),
                                                 ::testing::Values(ov::test::utils::DEVICE_GPU),
                                                 ::testing::ValuesIn(additional_config));

INSTANTIATE_TEST_SUITE_P(smoke_MM_Static_OneDNN, MatMulLayerGPUTest, testParamsOneDNN, MatMulLayerGPUTest::getTestCaseName);

const auto testParamsDynamic = ::testing::Combine(::testing::ValuesIn(IS_Dynamic),
                                                  ::testing::ValuesIn(netPRCs),
                                                  ::testing::Values(ov::element::dynamic),
                                                  ::testing::Values(ov::element::dynamic),
                                                  ::testing::Values(ov::test::utils::InputLayerType::PARAMETER),
                                                  ::testing::Values(ov::test::utils::DEVICE_GPU),
                                                  ::testing::ValuesIn(additional_config));

INSTANTIATE_TEST_SUITE_P(smoke_MM_Dynamic, MatMulLayerGPUTest, testParamsDynamic, MatMulLayerGPUTest::getTestCaseName);

const auto testParamsDynamic_nightly = ::testing::Combine(::testing::ValuesIn(IS_Dynamic_nightly),
                                                          ::testing::ValuesIn(netPRCs),
                                                          ::testing::Values(ov::element::dynamic),
                                                          ::testing::Values(ov::element::dynamic),
                                                          ::testing::Values(ov::test::utils::InputLayerType::PARAMETER),
                                                          ::testing::Values(ov::test::utils::DEVICE_GPU),
                                                          ::testing::ValuesIn(additional_config));

INSTANTIATE_TEST_SUITE_P(nightly_MM_Dynamic, MatMulLayerGPUTest, testParamsDynamic_nightly, MatMulLayerGPUTest::getTestCaseName);
} // namespace
