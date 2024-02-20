// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/constant.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/fusing_test_utils.hpp"
#include "transformations/rt_info/decompression.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

/*
 * WP - weights precision
 * DP - decompression precision
 * IP - input precision
 * OP - output precision
 *
 *    Weights(WP)     Subtract_const(WP)
 *       |               /
 *    Convert(DP)   Convert(DP)
 *            \        /
 *            Subtract(DP)
 *                  \      Multiply_const(DP)
 *                   \       /
 *                    Multiply
 *                      /
 *       Data(IP)   Convert(OP)
 *            \      /
 *             Gather(OP) Weights(OP)
 *                 \      /
 *                  MatMul(OP)  (Add MatMul in order to test OP==bf16 in SPR)
 */

struct InputAndWeigthsShapeParams {
    InputAndWeigthsShapeParams() = default;
    InputAndWeigthsShapeParams(InputShape _data_shape, ov::Shape _weights_shape)
        : data_shape(std::move(_data_shape)),
          weights_shape(std::move(_weights_shape)) {}

    InputShape data_shape;
    ov::Shape weights_shape;
};

using GatherWeightsDecompressParams = std::tuple<InputAndWeigthsShapeParams,
                                                 ov::AnyMap,  // additional config
                                                 fusingSpecificParams,
                                                 bool>;  // should use decompression implementation

class GatherWeightsDecompression : public testing::WithParamInterface<GatherWeightsDecompressParams>,
                                virtual public SubgraphBaseTest,
                                public CpuTestWithFusing {
public:
    static std::string getTestCaseName(testing::TestParamInfo<GatherWeightsDecompressParams> obj) {
        InputAndWeigthsShapeParams shape_params;
        ov::AnyMap additional_config;
        fusingSpecificParams fusing_params;
        bool should_fuse;

        std::tie(shape_params, additional_config, fusing_params, should_fuse) = obj.param;

        std::ostringstream result;
        result << "data_shape=" << shape_params.data_shape << "_";
        result << "weights_shape=" << shape_params.weights_shape << "_";

        result << "config=(";
        for (const auto& configEntry : additional_config) {
            result << configEntry.first << ", " << configEntry.second.as<std::string>() << ":";
        }
        result << ")";
        result << CpuTestWithFusing::getTestCaseName(fusing_params);

        return result.str();
    }

protected:
    std::shared_ptr<ov::Node> initDecompressionWeights(const ov::Shape& weights_shape,
                                                       const ov::element::Type weights_precision) {
        auto weights = ov::test::utils::make_constant(weights_precision,
                                                      weights_shape,
                                                      ov::test::utils::InputGenerateData{0, 255});
        weights->set_friendly_name("Compressed_weights");
        auto weights_convert = std::make_shared<ov::op::v0::Convert>(weights, ov::element::f16);

        std::shared_ptr<ov::Node> zp_const = ov::test::utils::make_constant(ov::element::u8,
                                                                            ov::Shape{weights_shape[0], 1},
                                                                            ov::test::utils::InputGenerateData{});
        auto zp_convert = std::make_shared<ov::op::v0::Convert>(zp_const, ov::element::f16);

        std::shared_ptr<ov::Node> scale_const =
            ov::test::utils::make_constant(ov::element::f16,
                                           ov::Shape{weights_shape[0], 1},
                                           ov::test::utils::InputGenerateData{});
        auto subtract = std::make_shared<ov::op::v1::Subtract>(weights_convert, zp_convert);
        auto multiply = std::make_shared<ov::op::v1::Multiply>(subtract, scale_const);
        auto last_node = std::make_shared<ov::op::v0::Convert>(multiply, ov::element::f32);
        return last_node;
    }

    std::shared_ptr<ov::Model> initSubgraph(const ov::PartialShape& data_shape,
                                            const ov::Shape& weights_shape,
                                            const ov::element::Type data_precision) {
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::i64, data_shape)};
        auto params_convert = std::make_shared<ov::op::v0::Convert>(params[0], ov::element::i32);
        auto axis = ov::op::v0::Constant::create(element::i32, Shape{1}, {0});

        const auto weights_subgraph = initDecompressionWeights(weights_shape,
                                                               ov::element::u8);

        auto gather = std::make_shared<ov::op::v8::Gather>(weights_subgraph, params_convert, axis);
        gather->set_friendly_name("GatherCompression");

        auto matB = ov::op::v0::Constant::create(element::f32, Shape{weights_shape[1], 1}, {1});
        auto matMul = std::make_shared<ov::op::v0::MatMul>(gather, matB, false, false);
        return makeNgraphFunction(data_precision, params, matMul, "GatherWeightsDecompression");
    }

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;

        InputAndWeigthsShapeParams shape_params;
        ov::AnyMap additional_config;
        fusingSpecificParams fusing_params;
        bool should_fuse;

        std::tie(shape_params, additional_config, fusing_params, should_fuse) = GetParam();

        configuration.insert(additional_config.begin(), additional_config.end());
        std::tie(postOpMgrPtr, fusedOps) = fusing_params;
        init_input_shapes({shape_params.data_shape, {{}, {{shape_params.weights_shape}}}});

        ElementType netType = ov::element::f32;
        inType = outType = netType;

        function = initSubgraph(inputDynamicShapes[0], shape_params.weights_shape, netType);
    }

    void check_results() {
        bool weights_found = false;
        for (const auto& n : compiledModel.get_runtime_model()->get_ordered_ops()) {
            if (n->get_friendly_name() == "Compressed_weights") {
                ASSERT_EQ(n->get_output_element_type(0), ov::element::u8);
                weights_found = true;
            }
        }
        ASSERT_TRUE(weights_found);

        bool gather_found = false;
        for (const auto& n : compiledModel.get_runtime_model()->get_ordered_ops()) {
            if (n->get_friendly_name() == "GatherCompression") {
                ASSERT_EQ(n->get_input_element_type(0), ov::element::u8);
                ASSERT_EQ(n->get_output_element_type(0), ov::element::f32);
                gather_found = true;
            }
        }
        ASSERT_TRUE(gather_found);

        CheckNumberOfNodesWithType(compiledModel, "Convert", 1);
        CheckNumberOfNodesWithType(compiledModel, "Subtract", 0);
        CheckNumberOfNodesWithType(compiledModel, "Multiply", 0);
        CheckNumberOfNodesWithType(compiledModel, "Subgraph", 0);
    }
};

TEST_P(GatherWeightsDecompression, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
    check_results();
}

namespace {

std::vector<ov::AnyMap> filter_additional_config() {
    std::vector<ov::AnyMap> additional_config = {};
    additional_config.push_back({{ov::hint::inference_precision(ov::element::f32)}});
    if (ov::with_cpu_x86_bfloat16()) {
        additional_config.push_back({{ov::hint::inference_precision(ov::element::bf16)}});
    }

    return additional_config;
}

const std::vector<InputAndWeigthsShapeParams> input_weights_shapes = {
    {{{-1, -1}, {{1, 1}}}, {16, 32}},
    {{{-1, -1}, {{1, 8}}}, {16, 64}},
    {{{}, {{2, 1}}}, {16, 33}}
};

const std::vector<fusingSpecificParams> fs_params{emptyFusingSpec, fusingBias};

INSTANTIATE_TEST_SUITE_P(smoke_GatherCompressedWeights_basic,
                         GatherWeightsDecompression,
                         ::testing::Combine(::testing::ValuesIn(input_weights_shapes),
                                            ::testing::ValuesIn(filter_additional_config()),
                                            ::testing::ValuesIn(fs_params),
                                            ::testing::Values(true)),
                         GatherWeightsDecompression::getTestCaseName);
}  // namespace
}  // namespace test
}  // namespace ov
