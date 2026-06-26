// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <optional>

#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/subgraph_builders/weights_decompression_builders.hpp"
#include "intel_gpu/runtime/internal_properties.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "transformations/rt_info/decompression.hpp"
#include "transformations/utils/utils.hpp"

namespace {
using ov::test::InputShape;

/*
 *                        Subtract_const(U8/NF4/U4/I4)
 *                             /
 *    Weights(U8/NF4/U4/I4)  Convert(F32)
 *       |                 /
 *    Convert(F32)   Reshape(optional)
 *            \        /       Multiply_const(F32)
 *            Subtract(optional)     /
 *                  \       Reshape(optional)
 *                   \       /
 *                   Multiply
 *                      |
 *      Data(F32)   Transpose(optional) or Multiply(optional)
 *            \     /
 *             Matmul
 *               |
 *              Bias
 */

struct ShapeParams {
    ShapeParams() = default;
    ShapeParams(InputShape data_shape, ov::Shape weights_shape, int weights_group_size = -1)
        : data_shape(std::move(data_shape)),
          weights_shape(std::move(weights_shape)),
          weights_group_size(weights_group_size) {}

    InputShape data_shape;
    ov::Shape weights_shape;
    // Decompression group size. If the value is equal to -1, ordinary decompression is used
    int weights_group_size;
};

using MatmulWeightsDecompressionParams = std::tuple<ShapeParams,                         // input shapes
                                                    ov::element::Type,                   // weights type
                                                    ov::element::Type,                   // activations type
                                                    ov::element::Type,                   // scale type
                                                    bool,                                // transpose on weights
                                                    ov::test::utils::DecompressionType,  // decompression subtract type
                                                    bool,                                // reshape on decompression constants
                                                    bool,                                // extra multiply
                                                    bool,                                // parameter weights
                                                    uint64_t,                            // dynamic_quantization_group_size
                                                    float                                // abs_threshold_f16
                                                    >;

class MatmulWeightsDecompression : public testing::WithParamInterface<MatmulWeightsDecompressionParams>,
                                   virtual public ov::test::SubgraphBaseTest {
public:
    static std::string get_test_case_name(testing::TestParamInfo<MatmulWeightsDecompressionParams> obj) {
        const auto& [shape_params,
                     weights_precision,
                     activations_precision,
                     scale_precision,
                     transpose,
                     decompression_sub,
                     reshape_on_decompression,
                     extra_multiply,
                     param_weights,
                     dyn_quan_group_size,
                     abs_threshold_f16] = obj.param;

        std::ostringstream result;
        result << "data_shape=";
        result << ov::test::utils::partialShape2str({shape_params.data_shape.first}) << "_";
        for (const auto& actual_shape : shape_params.data_shape.second) {
            result << ov::test::utils::partialShape2str({actual_shape}) << "_";
        }
        result << "_" << "weights_shape=" << shape_params.weights_shape << "_";
        result << "group_size=" << (shape_params.weights_group_size == -1 ? 1111 : shape_params.weights_group_size) << "_";
        result << "weights_precision=" << weights_precision << "_";
        result << "activations_precision=" << activations_precision << "_";
        result << "scale_precision=" << scale_precision << "_";
        result << "transpose_weights=" << transpose << "_";
        result << "decompression_subtract=" << decompression_sub << "_";
        result << "reshape_on_decompression=" << reshape_on_decompression << "_";
        result << "extra_multiply=" << extra_multiply << "_";
        result << "param_weights=" << param_weights << "_";
        result << "dyn_quan_group_size=" << dyn_quan_group_size << "_";

        return result.str();
    }

    void configure_model() override {
        ov::preprocess::PrePostProcessor p(function);
        {
            if (inType != ov::element::Type_t::dynamic) {
                p.input(0).tensor().set_element_type(inType);
            }
        }

        {
            auto results = function->get_results();
            for (size_t i = 0; i < results.size(); i++) {
                if (outType != ov::element::Type_t::dynamic) {
                    p.output(i).tensor().set_element_type(outType);
                }
            }
        }
        function = p.build();
    }

protected:
    std::shared_ptr<ov::Model> init_subgraph(const ov::PartialShape& data_shape,
                                              const ov::Shape& weights_shape,
                                              const int group_size,
                                              const ov::element::Type data_precision,
                                              const ov::element::Type weights_precision,
                                              const ov::element::Type scale_precision,
                                              const bool transpose_weights,
                                              const ov::test::utils::DecompressionType decompression_subtract_type,
                                              const bool reshape_on_decompression,
                                              const bool extra_multiply,
                                              const bool param_weight) {
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(data_precision, data_shape)};
        const auto decompression_multiply_type = ov::test::utils::DecompressionType::full;
        const std::optional<bool> insert_transpose_node = std::nullopt;
        const int seed = 1;
        const auto weights_subgraph = ov::test::utils::initMatMulDecompressionSubgraph(weights_shape,
                                                                                       group_size,
                                                                                       data_precision,
                                                                                       weights_precision,
                                                                                       data_precision,
                                                                                       scale_precision,
                                                                                       transpose_weights,
                                                                                       decompression_multiply_type,
                                                                                       decompression_subtract_type,
                                                                                       reshape_on_decompression,
                                                                                       insert_transpose_node,
                                                                                       seed,
                                                                                       extra_multiply,
                                                                                       param_weight);

        std::unordered_set<ov::Node*> visited_nodes;
        auto add_params = [&params](ov::Node* node) {
            if (auto param = std::dynamic_pointer_cast<ov::op::v0::Parameter>(node->shared_from_this())) {
                params.push_back(param);
            }
        };
        ov::op::util::visit_path(weights_subgraph.get(), visited_nodes, add_params, [](ov::Node*) {
            return false;
        });
        auto mat_mul = std::make_shared<ov::op::v0::MatMul>(params[0], weights_subgraph);
        return std::make_shared<ov::Model>(ov::OutputVector{mat_mul}, params, "MatmulWeightsDecompression");
    }

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;

        const auto& [shape_params,
                     weights_precision,
                     activations_precision,
                     scale_precision,
                     transpose_weights,
                     decompression_sub,
                     reshape_on_decompression,
                     extra_multiply,
                     param_weights,
                     dyn_quan_group_size,
                     abs_threshold_f16] = GetParam();

        init_input_shapes({shape_params.data_shape, {{}, {{shape_params.weights_shape}}}});

        inType = outType = activations_precision;

        const ov::element::Type scale_precision_to_use = scale_precision == ov::element::dynamic ? activations_precision : scale_precision;
        function = init_subgraph(inputDynamicShapes[0],
                                 shape_params.weights_shape,
                                 shape_params.weights_group_size,
                                 activations_precision,
                                 weights_precision,
                                 scale_precision_to_use,
                                 transpose_weights,
                                 decompression_sub,
                                 reshape_on_decompression,
                                 extra_multiply,
                                 param_weights);

        if (activations_precision == ov::element::f16) {
            abs_threshold = abs_threshold_f16;
        } else {
            abs_threshold = 1e-4f;
        }

        this->configuration.insert({ov::hint::dynamic_quantization_group_size(dyn_quan_group_size)});
    }

    void generate_inputs(const std::vector<ov::Shape>& target_input_static_shapes) override {
          inputs.clear();
          const auto& model_inputs = function->inputs();
          for (size_t i = 0; i < model_inputs.size(); ++i) {
                const auto& model_input = model_inputs[i];
                ov::test::utils::InputGenerateData in_data;
                in_data.start_from = -1;
                in_data.range = 3;
                in_data.resolution = 10000;
                ov::Tensor tensor = ov::test::utils::create_and_fill_tensor(model_input.get_element_type(),
                    model_input.get_partial_shape().is_static() ? model_input.get_shape() : target_input_static_shapes[i],
                                                                            in_data);
                inputs.insert({model_input.get_node_shared_ptr(), tensor});
          }
    }

    void check_results() {
        const auto& test_param = GetParam();
        ov::element::Type weights_precision = std::get<1>(test_param);
        for (const auto& n : compiledModel.get_runtime_model()->get_ordered_ops()) {
            if (n->get_friendly_name() == "Compressed_weights") {
                ASSERT_EQ(n->get_output_element_type(0), weights_precision);
            }
        }
    }
};

TEST_P(MatmulWeightsDecompression, Inference) {
    const auto& [shape_params,
                 weights_precision,
                 activations_precision,
                 scale_precision,
                 transpose_weights,
                 decompression_sub,
                 reshape_on_decompression,
                 extra_multiply,
                 param_weights,
                 dyn_quan_group_size,
                 abs_threshold_f16] = GetParam();
    // Skip tests for 4-bit parameter weights because 4-bit transpose is not supported
    if (param_weights && weights_precision != ov::element::u8) {
        GTEST_SKIP();
    }
    SKIP_IF_CURRENT_TEST_IS_DISABLED(); // This is necessary because of check_results
    run();
    check_results();
}

const std::vector<ov::element::Type> activations_precisions = {ov::element::f32, ov::element::f16};
const std::vector<ov::element::Type> weights_precisions = {ov::element::u8, ov::element::u4, ov::element::i4};
const std::vector<bool> transpose_weights = {true, false};
const std::vector<bool> param_weights = {true, false};
const std::vector<ShapeParams> input_shapes_basic = {
    {{{-1, -1, -1}, {{1, 4, 16}, {10, 16, 16}}}, {16, 32}},
    {{{}, {{1, 4, 16}}}, {16, 32}, 2ul},
    {{{}, {{1, 4, 16}}}, {1, 16, 32}},
    {{{}, {{1, 4, 48}}}, {48, 256}},
    {{{}, {{11, 339, 377}}}, {377, 335}}
};

const std::vector<ShapeParams> input_shapes_extra_multiply = {
    {{{}, {{1, 4, 2}}}, {2, 32}, 2ul},
    {{{}, {{1, 4, 16}}}, {1, 16, 32}},
};

const std::vector<ShapeParams> input_shapes_extra_multiply_non_trivial_batch_broadcast = {
    {{{}, {{1, 4, 16}}}, {16, 32}, 2ul},
};

INSTANTIATE_TEST_SUITE_P(smoke_MatMulCompressedWeights_basic,
                         MatmulWeightsDecompression,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_basic),
                                            ::testing::ValuesIn(weights_precisions),
                                            ::testing::ValuesIn(activations_precisions),
                                            ::testing::Values(ov::element::dynamic),
                                            ::testing::ValuesIn(transpose_weights),
                                            ::testing::Values(ov::test::utils::DecompressionType::full),
                                            ::testing::Values(true),
                                            ::testing::Values(false),
                                            ::testing::ValuesIn(param_weights),
                                            ::testing::Values(0),
                                            ::testing::Values(1.0f)),
                         MatmulWeightsDecompression::get_test_case_name);

INSTANTIATE_TEST_SUITE_P(smoke_MatMulCompressedWeights_extra_multiply,
                         MatmulWeightsDecompression,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_extra_multiply),
                                            ::testing::ValuesIn(weights_precisions),
                                            ::testing::ValuesIn(activations_precisions),
                                            ::testing::Values(ov::element::dynamic),
                                            ::testing::Values(false),
                                            ::testing::Values(ov::test::utils::DecompressionType::empty),
                                            ::testing::Values(false),
                                            ::testing::Values(true),
                                            ::testing::ValuesIn(param_weights),
                                            ::testing::Values(0),
                                            ::testing::Values(1.0f)),
                         MatmulWeightsDecompression::get_test_case_name);

INSTANTIATE_TEST_SUITE_P(smoke_MatMulCompressedWeights_extra_multiply_non_trivial_batch_broadcast_no_convert,
                         MatmulWeightsDecompression,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_extra_multiply_non_trivial_batch_broadcast),
                                            ::testing::ValuesIn(weights_precisions),
                                            ::testing::ValuesIn(activations_precisions),
                                            ::testing::Values(ov::element::dynamic),
                                            ::testing::Values(false),
                                            ::testing::Values(ov::test::utils::DecompressionType::empty),
                                            ::testing::Values(false),
                                            ::testing::Values(true),
                                            ::testing::ValuesIn(param_weights),
                                            ::testing::Values(0),
                                            ::testing::Values(1.0f)),
                         MatmulWeightsDecompression::get_test_case_name);

const std::vector<ShapeParams> input_shapes_corner_cases_basic = {
    {{{-1, -1, -1}, {{1, 4, 16}}}, {1, 16, 32}},
    {{{-1, -1, -1}, {{1, 4, 16}}}, {16, 32}},
    {{{-1, -1, 16}, {{1, 4, 16}}}, {16, 32}, 4},
    {{{-1, 16}, {{4, 16}}}, {16, 32}, 4},
};
const std::vector<ShapeParams> input_shapes_corner_cases_big = {
    {{{-1, -1, -1}, {{10, 40, 480}, {11, 40, 480}}}, {1, 480, 256}},
    {{{-1, -1, -1}, {{1, 1, 4096}}}, {4096, 4096}, 128},
    {{{-1, -1, -1}, {{1, 1, 4096}}}, {4096, 4096}},
    {{{-1, 4096}, {{1, 4096}}}, {4096, 4096}, 128},
};

const std::vector<ov::test::utils::DecompressionType> decompression_sub = {ov::test::utils::DecompressionType::full,
                                                                           ov::test::utils::DecompressionType::scalar,
                                                                           ov::test::utils::DecompressionType::empty};
const std::vector<ov::test::utils::DecompressionType> decompression_sub_no_full = {ov::test::utils::DecompressionType::scalar,
                                                                                   ov::test::utils::DecompressionType::empty};
const std::vector<ov::test::utils::DecompressionType> decompression_sub_no_scalar = {ov::test::utils::DecompressionType::full,
                                                                                     ov::test::utils::DecompressionType::empty};
const std::vector<bool> reshape_on_decompression = {true, false};

INSTANTIATE_TEST_SUITE_P(smoke_MatMulCompressedWeights_corner_cases_basic,
                         MatmulWeightsDecompression,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_corner_cases_basic),
                                            ::testing::ValuesIn(weights_precisions),
                                            ::testing::ValuesIn(activations_precisions),
                                            ::testing::Values(ov::element::dynamic),
                                            ::testing::ValuesIn(transpose_weights),
                                            ::testing::ValuesIn(decompression_sub),
                                            ::testing::ValuesIn(reshape_on_decompression),
                                            ::testing::Values(false),
                                            ::testing::ValuesIn(param_weights),
                                            ::testing::Values(0),
                                            ::testing::Values(1.0f)),
                         MatmulWeightsDecompression::get_test_case_name);

INSTANTIATE_TEST_SUITE_P(MatMulCompressedWeights_corner_cases_big,
                         MatmulWeightsDecompression,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_corner_cases_big),
                                            ::testing::ValuesIn(weights_precisions),
                                            ::testing::ValuesIn(activations_precisions),
                                            ::testing::Values(ov::element::dynamic),
                                            ::testing::ValuesIn(transpose_weights),
                                            ::testing::ValuesIn(decompression_sub),
                                            ::testing::ValuesIn(reshape_on_decompression),
                                            ::testing::Values(false),
                                            ::testing::ValuesIn(param_weights),
                                            ::testing::Values(0),
                                            ::testing::Values(1.0f)),
                         MatmulWeightsDecompression::get_test_case_name);


// per_tensor_zp=0 is not supported
// transpose_weights is not supported
// weight precision u4 is only supported
const std::vector<uint64_t> group_size = {128, 256, std::numeric_limits<int64_t>::max()};
INSTANTIATE_TEST_SUITE_P(smoke_MatMulCompressedWeights_dyn_quan,
                         MatmulWeightsDecompression,
                         ::testing::Combine(::testing::Values(ShapeParams{{{-1, -1, 1024}, {{1024, 1, 1024}, {1, 1, 1024}, {1024, 1, 1024}}},
                                                                            {1024, 1024}, 128}),  // shape
                                            ::testing::Values(ov::element::u4),
                                            ::testing::Values(ov::element::f16),
                                            ::testing::Values(ov::element::f16),
                                            ::testing::Values(false),
                                            ::testing::ValuesIn(decompression_sub_no_full),
                                            ::testing::Values(true),
                                            ::testing::Values(false),
                                            ::testing::Values(false),
                                            ::testing::ValuesIn(group_size),
                                            ::testing::Values(2.0f)),   // Note: this is because of potential cldnn accuracy issue
                         MatmulWeightsDecompression::get_test_case_name);

INSTANTIATE_TEST_SUITE_P(smoke_MatMulCompressedWeights_dyn_quan_precomputed_reduction,
                         MatmulWeightsDecompression,
                         ::testing::Combine(::testing::Values(ShapeParams{{{-1, -1, 1024}, {{1024, 1, 1024}}},
                                                                            {1024, 1024}, 128}),  // shape
                                            ::testing::Values(ov::element::u4),
                                            ::testing::Values(ov::element::f16),
                                            ::testing::Values(ov::element::f16),
                                            ::testing::Values(false),
                                            ::testing::ValuesIn(decompression_sub_no_scalar),
                                            ::testing::Values(true),
                                            ::testing::Values(false),
                                            ::testing::Values(false),
                                            ::testing::Values(128),
                                            ::testing::Values(2.0f)),   // Note: this is because of potential cldnn accuracy issue
                         MatmulWeightsDecompression::get_test_case_name);

INSTANTIATE_TEST_SUITE_P(smoke_MatMulCompressedWeights_dyn_quan_unaligned,     // dyn_quan is turned off because of innermost-shape
                         MatmulWeightsDecompression,
                         ::testing::Combine(::testing::ValuesIn({ShapeParams{{{-1, -1, 1008}, {{1, 1, 1008}}},
                                                                            {1008, 1024}, 1008}
                                                                            }),  // shape
                                            ::testing::Values(ov::element::u4),
                                            ::testing::Values(ov::element::f16),
                                            ::testing::Values(ov::element::f16),
                                            ::testing::Values(false),
                                            ::testing::Values(ov::test::utils::DecompressionType::empty),
                                            ::testing::Values(true),
                                            ::testing::Values(false),
                                            ::testing::Values(false),
                                            ::testing::Values(std::numeric_limits<int64_t>::max()),
                                            ::testing::Values(2.0f)),   // Note: this is because of potential cldnn accuracy issue
                         MatmulWeightsDecompression::get_test_case_name);

INSTANTIATE_TEST_SUITE_P(smoke_MatMulCompressedWeights_dyn_quan_no_slm,
                         MatmulWeightsDecompression,
                         ::testing::Combine(::testing::Values(ShapeParams{{{-1, -1, 1024}, {{2, 1, 1024}, {1, 1, 1024}, {2, 1, 1024}}},
                                                                            {1024, 1}, 128}),  // shape
                                            ::testing::Values(ov::element::u8),
                                            ::testing::Values(ov::element::f16),
                                            ::testing::Values(ov::element::f16),
                                            ::testing::Values(false),
                                            ::testing::ValuesIn(decompression_sub_no_full),
                                            ::testing::Values(true),
                                            ::testing::Values(false),
                                            ::testing::Values(false),
                                            ::testing::ValuesIn(group_size),
                                            ::testing::Values(2.0f)),   // Note: this is because of potential cldnn accuracy issue
                         MatmulWeightsDecompression::get_test_case_name);

INSTANTIATE_TEST_SUITE_P(smoke_MatMulCompressedWeights_3D_weight,
                         MatmulWeightsDecompression,
                         ::testing::Combine(::testing::Values(ShapeParams{{{-1, -1, 32}, {{32, 128, 32}}},
                                                                            {32, 32, 1024}, 32}),
                                            ::testing::Values(ov::element::u4),
                                            ::testing::Values(ov::element::f16),
                                            ::testing::Values(ov::element::f16),
                                            ::testing::Values(false),
                                            ::testing::Values(ov::test::utils::DecompressionType::empty),
                                            ::testing::Values(false),
                                            ::testing::Values(false),
                                            ::testing::Values(false),
                                            ::testing::Values(0),
                                            ::testing::Values(2.0f)),
                         MatmulWeightsDecompression::get_test_case_name);

INSTANTIATE_TEST_SUITE_P(
   smoke_MatMulCompressedWeights_dyn_quan_mxfp8,
   MatmulWeightsDecompression,
   ::testing::Combine(::testing::Values(ShapeParams{{{-1, -1, 4096}, {{1, 1, 4096}, {8, 1, 4096}}}, {4096, 1024}, 32}),  // shape
                      ::testing::ValuesIn({ov::element::f8e4m3, ov::element::f8e5m2}),
                      ::testing::Values(ov::element::f16),
                      ::testing::Values(ov::element::f8e8m0),
                      ::testing::Values(true),
                      ::testing::Values(ov::test::utils::DecompressionType::empty),
                      ::testing::Values(false),
                      ::testing::Values(false),
                      ::testing::Values(false),
                      ::testing::Values(32),
                      ::testing::Values(2.5f)),
   MatmulWeightsDecompression::get_test_case_name);

INSTANTIATE_TEST_SUITE_P(
   smoke_MatMulCompressedWeights_dyn_quan_fp8,
   MatmulWeightsDecompression,
   ::testing::Combine(::testing::Values(ShapeParams{{{-1, -1, 128}, {{2, 1, 128}, {1, 1, 128}, {2, 1, 128}}}, {128, 16}, 128}),  // shape
                      ::testing::ValuesIn({ov::element::f8e4m3, ov::element::f8e5m2}),
                      ::testing::Values(ov::element::f16),
                      ::testing::Values(ov::element::f16),
                      ::testing::Values(true),
                      ::testing::Values(ov::test::utils::DecompressionType::empty),
                      ::testing::Values(false),
                      ::testing::Values(false),
                      ::testing::Values(false),
                      ::testing::ValuesIn(std::vector<uint64_t>{32, 128, std::numeric_limits<uint64_t>::max()}),
                      ::testing::Values(1.0f)),
   MatmulWeightsDecompression::get_test_case_name);

INSTANTIATE_TEST_SUITE_P(smoke_MatMulCompressedWeights_dyn_quan_scalar_wzp,
                         MatmulWeightsDecompression,
                         ::testing::Combine(::testing::Values(ShapeParams{{{-1, -1, 1024}, {{1024, 1, 1024}}},
                                                                          {1024, 1024}, 128}),
                                            ::testing::Values(ov::element::u4),
                                            ::testing::Values(ov::element::f16),
                                            ::testing::Values(ov::element::f16),
                                            ::testing::Values(false),
                                            ::testing::Values(ov::test::utils::DecompressionType::scalar),
                                            ::testing::Values(false),
                                            ::testing::Values(false),
                                            ::testing::Values(false),
                                            ::testing::Values(128),
                                            ::testing::Values(2.0f)),
                         MatmulWeightsDecompression::get_test_case_name);

INSTANTIATE_TEST_SUITE_P(smoke_MatMulCompressedWeights_dyn_quan_precomputed_reduction_with_gs16,
                         MatmulWeightsDecompression,
                         ::testing::Combine(::testing::Values(ShapeParams{{{-1, -1, 128}, {{128, 1, 128}}},
                                                                            {128, 128}, 16}),  // shape
                                            ::testing::Values(ov::element::u8),
                                            ::testing::Values(ov::element::f16),
                                            ::testing::Values(ov::element::f16),
                                            ::testing::Values(false),
                                            ::testing::ValuesIn(decompression_sub_no_scalar),
                                            ::testing::Values(true),
                                            ::testing::Values(false),
                                            ::testing::Values(false),
                                            ::testing::Values(128),
                                            ::testing::Values(2.0f)),
                         MatmulWeightsDecompression::get_test_case_name);

INSTANTIATE_TEST_SUITE_P(smoke_MatMulCompressedWeights_input_4d,
                         MatmulWeightsDecompression,
                         ::testing::Combine(::testing::Values(ShapeParams{{{1, 1, -1, 2048}, {{1, 1, 2, 2048}}},
                                                                            {2048, 512}, 32}),  // shape
                                            ::testing::Values(ov::element::u4),
                                            ::testing::Values(ov::element::f16),
                                            ::testing::Values(ov::element::f16),
                                            ::testing::Values(false),
                                            ::testing::Values(ov::test::utils::DecompressionType::empty),
                                            ::testing::Values(false),
                                            ::testing::Values(false),
                                            ::testing::Values(false),
                                            ::testing::Values(0),
                                            ::testing::Values(1.0f)),
                         MatmulWeightsDecompression::get_test_case_name);

/*
 * Regression for the intel_gpu DynamicQuantize fanout assert (dynamic_quantize.cpp).
 *
 * A single activation feeding several FullyConnected nodes (e.g. VAE mid-block
 * self-attention to_q / to_k / to_v in FLUX.2-klein) yields, after
 * DynamicQuantizeFullyConnected + SharedOpOptimization, one shared DynamicQuantize
 * with multiple FC users (get_users().size() > get_outputs_count()). For a
 * dynamically-shaped activation this used to trip
 * OPENVINO_ASSERT(get_users().size() == get_outputs_count()) at the first infer.
 * This verifies compile + inference succeeds with dynamic quantization enabled.
 *
 *      Data (dynamic [-1,-1,512])
 *        /        |        \
 *    MatMul_q  MatMul_k  MatMul_v   (each with decompressed int8 weights)
 *
 * The three projections use distinct output channel counts so GPU horizontal FC
 * fusion keeps them separate, guaranteeing the shared DynamicQuantize retains
 * multiple FC users (which is what triggered the assert).
 */
class MatmulSharedDynQuantMultipleFC : public testing::WithParamInterface<uint64_t>,
                                       virtual public ov::test::SubgraphBaseTest {
public:
    static std::string get_test_case_name(testing::TestParamInfo<uint64_t> obj) {
        std::ostringstream result;
        result << "dyn_quan_group_size=" << obj.param;
        return result.str();
    }

protected:
    std::shared_ptr<ov::Node> make_compressed_weights(size_t input_channels,
                                                      size_t output_channels,
                                                      const ov::element::Type& data_precision) {
        auto weights_tensor = ov::test::utils::create_and_fill_tensor(ov::element::u8, ov::Shape{input_channels, output_channels});
        auto weights = std::make_shared<ov::op::v0::Constant>(weights_tensor);
        weights->set_friendly_name("Compressed_weights");
        auto weights_convert = std::make_shared<ov::op::v0::Convert>(weights, data_precision);

        ov::test::utils::InputGenerateData in_data;
        in_data.start_from = -0.5;
        in_data.range = 1;
        in_data.resolution = 30000;
        auto scale_tensor = ov::test::utils::create_and_fill_tensor(data_precision, ov::Shape{1, output_channels}, in_data);
        for (size_t i = 0; i < scale_tensor.get_size(); i++) {
            if (data_precision == ov::element::f16)
                scale_tensor.data<ov::float16>()[i] /= ov::float16(16.f);
            else if (data_precision == ov::element::f32)
                scale_tensor.data<float>()[i] /= 16.f;
        }
        auto scale_const = std::make_shared<ov::op::v0::Constant>(scale_tensor);
        return std::make_shared<ov::op::v1::Multiply>(weights_convert, scale_const);
    }

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;
        const uint64_t dyn_quan_group_size = GetParam();
        const size_t hidden = 512;

        // Dynamic 3D activation, matching the VAE mid-block attention input [-1, -1, 512].
        InputShape data_shape = {{-1, -1, static_cast<ov::Dimension::value_type>(hidden)},
                                 {{1, 64, hidden}, {1, 256, hidden}}};
        init_input_shapes({data_shape});
        inType = outType = ov::element::f16;

        auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, inputDynamicShapes[0]);
        // Distinct output sizes for to_q / to_k / to_v so horizontal FC fusion keeps them
        // separate, while they still share one DynamicQuantize on the common activation.
        const std::vector<size_t> output_channels = {hidden, hidden - 128, hidden - 256};
        ov::OutputVector matmuls;
        for (auto out_ch : output_channels) {
            auto weights = make_compressed_weights(hidden, out_ch, ov::element::f16);
            matmuls.push_back(std::make_shared<ov::op::v0::MatMul>(param, weights));
        }
        function = std::make_shared<ov::Model>(matmuls, ov::ParameterVector{param}, "SharedDynQuantMultipleFC");

        abs_threshold = 2.0f;  // dynamic quantization accuracy tolerance (see other dyn_quan cases)
        configuration.insert({ov::hint::dynamic_quantization_group_size(dyn_quan_group_size)});
    }
};

TEST_P(MatmulSharedDynQuantMultipleFC, Inference) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
}

INSTANTIATE_TEST_SUITE_P(smoke_MatMulSharedDynQuantMultipleFC,
                         MatmulSharedDynQuantMultipleFC,
                         ::testing::Values<uint64_t>(32, 128, std::numeric_limits<uint64_t>::max()),
                         MatmulSharedDynQuantMultipleFC::get_test_case_name);
} // namespace
