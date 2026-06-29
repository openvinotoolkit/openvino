// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/grouped_matmul.hpp"

#include <algorithm>
#include <numeric>

#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/subgraph_builders/weights_decompression_builders.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/grouped_matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"

namespace ov {
namespace test {

void GroupedMatMulTestBase::SetUp() {
    const auto& [a_input_shape, b_shape, tokens_per_expert] = shape_params_;

    OPENVINO_ASSERT(a_input_shape.first.rank().is_static(),
                    "GroupedMatMul test: mat_a PartialShape must have static rank");
    const auto a_rank = a_input_shape.first.rank().get_length();
    OPENVINO_ASSERT(a_rank == 2 || a_rank == 3,
                    "GroupedMatMul test: mat_a rank must be 2 or 3, got ", a_rank);
    OPENVINO_ASSERT(b_shape.size() == 3,
                    "GroupedMatMul test: b_shape must be 3D [G,N,K], got rank ", b_shape.size());
    const bool is_2d_3d = (a_rank == 2);
    const size_t G = b_shape[0];
    const size_t num_iters = a_input_shape.second.size();

    if (is_2d_3d) {
        init_input_shapes({
            a_input_shape,
            {ov::PartialShape{static_cast<ov::Dimension::value_type>(G)},
             std::vector<ov::Shape>(num_iters, ov::Shape{G})}
        });
    } else {
        init_input_shapes({a_input_shape});
    }

    auto param_a = std::make_shared<ov::op::v0::Parameter>(act_type_, inputDynamicShapes[0]);
    auto const_b = build_weights();

    std::shared_ptr<ov::Node> gm_op;
    ov::ParameterVector params = {param_a};

    if (is_2d_3d) {
        auto param_offsets = std::make_shared<ov::op::v0::Parameter>(
            ov::element::i32, inputDynamicShapes[1]);
        params.push_back(param_offsets);
        gm_op = std::make_shared<ov::op::v17::GroupedMatMul>(param_a, const_b, param_offsets);
    } else {
        gm_op = std::make_shared<ov::op::v17::GroupedMatMul>(param_a, const_b);
    }

    function = std::make_shared<ov::Model>(
        ov::ResultVector{std::make_shared<ov::op::v0::Result>(gm_op)},
        params,
        model_name_);

    abs_threshold = 0.05f;
    rel_threshold = 0.01f;
}

void GroupedMatMulTestBase::validate() {
    SubgraphBaseTest::validate();
    if (!expected_primitive_.empty()) {
        CheckNumberOfNodesWithType(compiledModel, expected_primitive_, 1);
    }
}

void GroupedMatMulTestBase::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    inputs.clear();
    const auto& [a_input_shape, b_shape, tokens_per_expert] = shape_params_;

    OPENVINO_ASSERT(a_input_shape.first.rank().is_static());
    const bool is_2d_3d = (a_input_shape.first.rank().get_length() == 2);
    const auto& a_static_shapes = a_input_shape.second;
    const auto& target_shape = targetInputStaticShapes[0];
    const auto it = std::find(a_static_shapes.begin(), a_static_shapes.end(), target_shape);
    OPENVINO_ASSERT(it != a_static_shapes.end(),
                    "GroupedMatMul test: target shape not found in a_static_shapes");
    const size_t iter = static_cast<size_t>(std::distance(a_static_shapes.begin(), it));

    ov::test::utils::InputGenerateData gen_data;
    gen_data.range = 2;
    gen_data.resolution = 128;
    gen_data.start_from = -1;

    inputs.insert({function->get_parameters()[0],
                   ov::test::utils::create_and_fill_tensor(act_type_, targetInputStaticShapes[0], gen_data)});

    if (is_2d_3d) {
        const size_t G = b_shape[0];
        const size_t T = targetInputStaticShapes[0][0];

        std::vector<int32_t> offsets(G);
        if (!tokens_per_expert.empty()) {
            OPENVINO_ASSERT(iter < tokens_per_expert.size(),
                            "GroupedMatMul test: iter ", iter,
                            " out of range for tokens_per_expert of size ", tokens_per_expert.size());
            const auto& tpe = tokens_per_expert[iter];
            OPENVINO_ASSERT(tpe.size() == G,
                            "tokens_per_expert[", iter, "] must have G=", G,
                            " entries, got ", tpe.size());
            // offsets[g] = tpe[0] + tpe[1] + ... + tpe[g]  (exclusive row-end for expert g)
            std::partial_sum(tpe.begin(), tpe.end(), offsets.begin(), [](int32_t a, size_t b) {
                return a + static_cast<int32_t>(b);
            });
            OPENVINO_ASSERT(offsets[G - 1] == static_cast<int32_t>(T),
                            "tokens_per_expert[", iter, "] sums to ", offsets[G - 1],
                            " but T=", T);
        } else {
            // Even distribution: offsets[g] = ceil(T*(g+1)/G), last clamped to T.
            for (size_t g = 0; g < G; ++g) {
                offsets[g] = static_cast<int32_t>((T * (g + 1) + G - 1) / G);
            }
            offsets[G - 1] = static_cast<int32_t>(T);
        }

        ov::Tensor offsets_tensor(ov::element::i32, ov::Shape{G});
        auto* dst = offsets_tensor.data<int32_t>();
        for (size_t g = 0; g < G; ++g)
            dst[g] = offsets[g];

        inputs.insert({function->get_parameters()[1], offsets_tensor});
    }
}

std::string GroupedMatMulLayerTest::getTestCaseName(const testing::TestParamInfo<GroupedMatMulParams>& obj) {
    const auto& [shape_params, elem_type, target_device, expected_primitive] = obj.param;
    const auto& [a_input_shape, b_shape, tokens_per_expert] = shape_params;

    OPENVINO_ASSERT(a_input_shape.first.rank().is_static(),
                    "GroupedMatMul test: mat_a PartialShape must have static rank");

    std::ostringstream result;
    result << "A_shape=" << a_input_shape << "_";
    result << "B_shape=" << ov::test::utils::vec2str(b_shape) << "_";
    result << "ET=" << elem_type << "_";
    result << "targetDevice=" << target_device;
    return result.str();
}

void GroupedMatMulLayerTest::SetUp() {
    const auto& [shape_params, elem_type, _targetDevice, expected_primitive] = GetParam();
    shape_params_ = shape_params;
    act_type_ = elem_type;
    model_name_ = "GroupedMatMul";
    expected_primitive_ = expected_primitive;
    targetDevice = _targetDevice;
    GroupedMatMulTestBase::SetUp();
}

std::shared_ptr<ov::Node> GroupedMatMulLayerTest::build_weights() {
    ov::test::utils::InputGenerateData b_data;
    b_data.range = 2;
    b_data.resolution = 128;
    b_data.start_from = -1;
    auto b_tensor = ov::test::utils::create_and_fill_tensor(act_type_, shape_params_.b_shape, b_data);
    return std::make_shared<ov::op::v0::Constant>(b_tensor);
}

std::string GroupedMatMulCompressedLayerTest::getTestCaseName(
    const testing::TestParamInfo<GroupedMatMulCompressedParams>& obj) {
    const auto& [shape_params, act_type, weights_prec, decomp_prec, scale_prec,
                 multiply_type, subtract_type, reshape_on_decomp, group_size,
                 target_device, expected_primitive] = obj.param;
    const auto& [a_input_shape, b_shape, tokens_per_expert] = shape_params;

    OPENVINO_ASSERT(a_input_shape.first.rank().is_static());

    std::ostringstream result;
    result << "A_shape=" << a_input_shape << "_";
    result << "B_shape=" << ov::test::utils::vec2str(b_shape) << "_";
    result << "ActET=" << act_type << "_";
    result << "WET=" << weights_prec << "_";
    result << "DecompPrec=" << decomp_prec << "_";
    result << "ScalePrec=" << scale_prec << "_";
    result << "Mul=" << multiply_type << "_Sub=" << subtract_type << "_";
    result << "Reshape=" << reshape_on_decomp << "_";
    result << "GrpSz=" << group_size << "_";
    result << "targetDevice=" << target_device;
    return result.str();
}

void GroupedMatMulCompressedLayerTest::SetUp() {
    const auto& [shape_params, act_type, weights_prec, decomp_prec, scale_prec,
                 multiply_type, subtract_type, reshape_on_decomp, group_size,
                 _targetDevice, expected_primitive] = GetParam();
    shape_params_ = shape_params;
    act_type_ = act_type;
    model_name_ = "GroupedMatMulCompressed";
    expected_primitive_ = expected_primitive;
    weights_prec_ = weights_prec;
    decomp_prec_ = decomp_prec;
    scale_prec_ = scale_prec;
    multiply_type_ = multiply_type;
    subtract_type_ = subtract_type;
    reshape_on_decomp_ = reshape_on_decomp;
    group_size_ = group_size;
    targetDevice = _targetDevice;
    GroupedMatMulTestBase::SetUp();
}

std::shared_ptr<ov::Node> GroupedMatMulCompressedLayerTest::build_weights() {
    // b_shape is the pre-transposed weight shape [G, N, K].
    // We pass the PLANAR shape [G, K, N] to initMatMulDecompressionSubgraphQuantization
    // with transpose_weights=true so the internal constant is laid out as [G, N, K]
    // (per-N scales ≡ per-OC). Passing insert_transpose_node=false suppresses the
    // trailing Transpose node, leaving the decompressed output in [G, N, K] form
    // ready for GroupedMatMul.
    ov::Shape b_planar = shape_params_.b_shape;
    std::swap(b_planar[b_planar.size() - 2], b_planar[b_planar.size() - 1]);  // [G, K, N]

    return ov::test::utils::initMatMulDecompressionSubgraphQuantization(
        b_planar,
        group_size_,
        act_type_,
        weights_prec_,
        decomp_prec_,
        scale_prec_,
        true,              // transpose_weights: constant stored as [G, N, K]
        multiply_type_,
        subtract_type_,
        reshape_on_decomp_,
        false,             // insert_transpose_node: omit trailing Transpose → output [G, N, K]
        1);                // seed
}

}  // namespace test
}  // namespace ov
