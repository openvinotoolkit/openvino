// Copyright (C) 2018-2023 Intel Corporation
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
    InputAndWeigthsShapeParams(InputShape _data_shape, ov::Shape _weights_shape, ov::Shape _zp_shape, ov::Shape _scale_shape)
        : data_shape(std::move(_data_shape)),
          weights_shape(std::move(_weights_shape)),
          zp_shape(std::move(_zp_shape)),
          scale_shape(std::move(_scale_shape)) {}

    InputShape data_shape;
    ov::Shape weights_shape;
    ov::Shape zp_shape;
    ov::Shape scale_shape;
};

using GatherWeightsDecompressParams = std::tuple<InputAndWeigthsShapeParams,
                                                 ov::element::Type,  // Weights precision
                                                 ov::element::Type,  // Inference precision
                                                 bool,               // Scalar axis or not
                                                 fusingSpecificParams,
                                                 bool>;  // should use decompression implementation

class GatherWeightsDecompression : public testing::WithParamInterface<GatherWeightsDecompressParams>,
                                virtual public SubgraphBaseTest,
                                public CpuTestWithFusing {
public:
    static std::string getTestCaseName(testing::TestParamInfo<GatherWeightsDecompressParams> obj) {
        InputAndWeigthsShapeParams shape_params;
        ov::element::Type weights_precision;
        ov::element::Type infer_precision;
        bool scalar_axis;
        fusingSpecificParams fusing_params;
        bool should_fuse;

        std::tie(shape_params, weights_precision, infer_precision, scalar_axis, fusing_params, should_fuse) = obj.param;

        std::ostringstream result;
        result << "data_shape=" << shape_params.data_shape << "_";
        result << "weights_shape=" << shape_params.weights_shape << "_";
        result << "zp_shape=" << shape_params.zp_shape << "_";
        result << "scale_shape=" << shape_params.scale_shape << "_";
        result << "weights_precision=" << weights_precision << "_";
        result << "infer_precision=" << infer_precision << "_";
        result << "scalar_axis=" << scalar_axis << "_";
        result << CpuTestWithFusing::getTestCaseName(fusing_params);
        return result.str();
    }

protected:
    std::shared_ptr<ov::Node> initDecompressionWeights(const InputAndWeigthsShapeParams& shape_params,
                                                       const ov::element::Type weights_precision,
                                                       const ov::element::Type infer_precision) {
        auto weights = ov::test::utils::make_constant(weights_precision,
                                                      shape_params.weights_shape,
                                                      ov::test::utils::InputGenerateData{0, 255});
        weights->set_friendly_name("Compressed_weights");
        auto weights_convert = std::make_shared<ov::op::v0::Convert>(weights, infer_precision);

        std::shared_ptr<ov::Node> zp_const =
            ov::test::utils::make_constant(weights_precision, shape_params.zp_shape, ov::test::utils::InputGenerateData{});
        auto zp_convert = std::make_shared<ov::op::v0::Convert>(zp_const, infer_precision);
        zp_convert->set_friendly_name("zp_convert");

        std::shared_ptr<ov::Node> scale_const =
            ov::test::utils::make_constant(infer_precision, shape_params.scale_shape, ov::test::utils::InputGenerateData{});
        auto subtract = std::make_shared<ov::op::v1::Subtract>(weights_convert, zp_convert);
        auto multiply = std::make_shared<ov::op::v1::Multiply>(subtract, scale_const);
        multiply->set_friendly_name("multiply:subtract_by_scale");

        std::shared_ptr<ov::op::v1::Reshape> multiply_reshape = nullptr;
        if (shape_params.weights_shape.size() != 2u) {
            // Group quantization
            auto constReshape = ov::op::v0::Constant::create(
                ov::element::i32,
                {2u},
                {shape_params.weights_shape[0], shape_params.weights_shape[1] * shape_params.weights_shape[2]});
            multiply_reshape = std::make_shared<ov::op::v1::Reshape>(multiply, constReshape, true);
            if (multiply_reshape->get_element_type() != ov::element::f32) {
                return std::make_shared<ov::op::v0::Convert>(multiply_reshape, ov::element::f32);
            }
            return multiply_reshape;
        }

        if (multiply->get_element_type() != ov::element::f32) {
            return std::make_shared<ov::op::v0::Convert>(multiply, ov::element::f32);
        }
        return multiply;
    }

    std::shared_ptr<ov::Model> initSubgraph(const ov::PartialShape& data_shape,
                                            const InputAndWeigthsShapeParams& shape_params,
                                            const ov::element::Type& weights_precision,
                                            const ov::element::Type& infer_precision,
                                            const bool& scalar_axis) {
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::i64, data_shape)};
        auto params_convert = std::make_shared<ov::op::v0::Convert>(params[0], ov::element::i32);
        auto axis = ov::op::v0::Constant::create(element::i32, scalar_axis ? ov::Shape() : ov::Shape{1}, {0});

        const auto weights_subgraph = initDecompressionWeights(shape_params, weights_precision, infer_precision);

        auto gather = std::make_shared<ov::op::v8::Gather>(weights_subgraph, params_convert, axis);
        gather->set_friendly_name("GatherCompression");

        auto fea_dim = shape_params.weights_shape.size() == 2u
                           ? shape_params.weights_shape[1]
                           : shape_params.weights_shape[1] * shape_params.weights_shape[2];
        auto matB = ov::op::v0::Constant::create(element::f32, Shape{fea_dim, 1}, {1});
        matB->set_friendly_name("matB");
        auto matMul = std::make_shared<ov::op::v0::MatMul>(gather, matB, false, false);
        matMul->set_friendly_name("lastMatMul");
        return makeNgraphFunction(ov::element::f32, params, matMul, "GatherWeightsDecompression");
    }

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;

        InputAndWeigthsShapeParams shape_params;
        ov::element::Type weights_precision;
        ov::element::Type infer_precision;
        bool scalar_axis;
        fusingSpecificParams fusing_params;
        bool should_fuse;

        std::tie(shape_params, weights_precision, infer_precision, scalar_axis, fusing_params, should_fuse) = GetParam();

        configuration.insert({ov::hint::inference_precision.name(), infer_precision});

        std::tie(postOpMgrPtr, fusedOps) = fusing_params;
        init_input_shapes({shape_params.data_shape, {{}, {{shape_params.weights_shape}}}});

        inType = outType = infer_precision;

        function = initSubgraph(inputDynamicShapes[0], shape_params, weights_precision, infer_precision, scalar_axis);
    }

    void check_results() {
        bool weights_found = false;
        for (const auto& n : compiledModel.get_runtime_model()->get_ordered_ops()) {
            if (n->get_friendly_name() == "Compressed_weights") {
                ASSERT_TRUE(n->get_output_element_type(0) == ov::element::u8 ||
                            n->get_output_element_type(0) == ov::element::u4);
                weights_found = true;
            }
        }
        ASSERT_TRUE(weights_found);

        CheckNumberOfNodesWithType(compiledModel, "Convert", 1);
        CheckNumberOfNodesWithType(compiledModel, "Subtract", 0);
        CheckNumberOfNodesWithType(compiledModel, "Multiply", 0);
        CheckNumberOfNodesWithType(compiledModel, "Subgraph", 0);
        CheckNumberOfNodesWithType(compiledModel, "Gather", 0);
        CheckNumberOfNodesWithType(compiledModel, "GatherCompression", 1);
    }
};

TEST_P(GatherWeightsDecompression, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
    check_results();
}

namespace {

std::vector<ov::element::Type> filter_infer_precision() {
    std::vector<ov::element::Type> additional_infer_precision = {};
    additional_infer_precision.push_back(ov::element::f32);
    if (ov::with_cpu_x86_bfloat16()) {
        additional_infer_precision.push_back(ov::element::bf16);
    }
    return additional_infer_precision;
}

const std::vector<InputAndWeigthsShapeParams> input_weights_shapes = {
    {{{-1, -1}, {{1, 1}}}, {64, 4, 32}, {64, 4, 1}, {64, 4, 1}},
    {{{-1, -1}, {{1, 1}}}, {8, 4, 32}, {1}, {8, 4, 1}},
    {{{-1, -1}, {{1, 1}}}, {16, 32}, {16, 1}, {16, 1}},
    {{{-1, -1}, {{1, 8}}}, {16, 64}, {16, 1}, {16, 1}},
    {{{}, {{2, 1}}}, {16, 33}, {16, 1}, {16, 1}}
};

const std::vector<ov::element::Type> input_weights_precision = {{ov::element::u8, ov::element::u4}};

const std::vector<fusingSpecificParams> fs_params{emptyFusingSpec, fusingBias};

const std::vector<bool> vec_scalar_axis{true, false};

INSTANTIATE_TEST_SUITE_P(smoke_GatherCompressedWeights_basic,
                         GatherWeightsDecompression,
                         ::testing::Combine(::testing::ValuesIn(input_weights_shapes),
                                            ::testing::ValuesIn(input_weights_precision),
                                            ::testing::ValuesIn(filter_infer_precision()),
                                            ::testing::ValuesIn(vec_scalar_axis),
                                            ::testing::ValuesIn(fs_params),
                                            ::testing::Values(true)),
                         GatherWeightsDecompression::getTestCaseName);
}  // namespace
}  // namespace test
}  // namespace ov
