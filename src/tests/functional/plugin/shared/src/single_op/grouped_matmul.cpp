// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/grouped_matmul.hpp"

#include <algorithm>

#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/subgraph_builders/weights_decompression_builders.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/grouped_matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"

namespace ov {
namespace test {

std::string GroupedMatMulLayerTest::getTestCaseName(const testing::TestParamInfo<GroupedMatMulParams>& obj) {
    const auto& [shape_params, elem_type, target_device] = obj.param;
    const auto& [a_input_shape, b_shape, tokens_per_expert] = shape_params;

    OPENVINO_ASSERT(a_input_shape.first.rank().is_static(),
                    "GroupedMatMul test: mat_a PartialShape must have static rank");
    const bool is_2d_3d = (a_input_shape.first.rank().get_length() == 2);

    std::ostringstream result;
    result << "Case=" << (is_2d_3d ? "2Dx3D" : "3Dx3D") << "_";
    result << "A_dyn=" << a_input_shape.first << "_";
    result << "A_static=";
    for (size_t i = 0; i < a_input_shape.second.size(); ++i) {
        result << ov::test::utils::vec2str(a_input_shape.second[i]);
        if (i + 1 < a_input_shape.second.size())
            result << ",";
    }
    result << "_B=" << ov::test::utils::vec2str(b_shape) << "_";
    result << "ET=" << elem_type.to_string() << "_";
    result << "Dev=" << target_device;
    return result.str();
}

void GroupedMatMulLayerTest::SetUp() {
    const auto& [shape_params, elem_type, _targetDevice] = GetParam();
    const auto& [a_input_shape, b_shape, tokens_per_expert] = shape_params;
    targetDevice = _targetDevice;

    OPENVINO_ASSERT(a_input_shape.first.rank().is_static(),
                    "GroupedMatMul test: mat_a PartialShape must have static rank");
    const bool is_2d_3d = (a_input_shape.first.rank().get_length() == 2);
    const size_t G = b_shape[0];
    const size_t num_iters = a_input_shape.second.size();

    // For 2D×3D, offsets is a runtime parameter of static shape [G].
    // Include it in init_input_shapes so the base class tracks static shapes for it.
    if (is_2d_3d) {
        init_input_shapes({
            a_input_shape,
            {ov::PartialShape{static_cast<ov::Dimension::value_type>(G)},
             std::vector<ov::Shape>(num_iters, ov::Shape{G})}
        });
    } else {
        init_input_shapes({a_input_shape});
    }

    auto param_a = std::make_shared<ov::op::v0::Parameter>(elem_type, inputDynamicShapes[0]);

    // B: constant weight matrix, pre-transposed [G, N, K].
    ov::test::utils::InputGenerateData b_data;
    b_data.range = 2;
    b_data.resolution = 128;
    b_data.start_from = -1;
    auto b_tensor = ov::test::utils::create_and_fill_tensor(elem_type, b_shape, b_data);
    auto const_b = std::make_shared<ov::op::v0::Constant>(b_tensor);

    std::shared_ptr<ov::Node> gm_op;
    ov::ParameterVector params = {param_a};

    if (is_2d_3d) {
        // Offsets [G] as a runtime parameter — values provided in generate_inputs.
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
        "GroupedMatMul");

    if (elem_type == ov::element::f16) {
        abs_threshold = 0.05f;
        rel_threshold = 0.01f;
    }
}

void GroupedMatMulLayerTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    inputs.clear();
    const auto& [shape_params, elem_type, _target_device] = GetParam();
    const auto& [a_input_shape, b_shape, tokens_per_expert] = shape_params;

    OPENVINO_ASSERT(a_input_shape.first.rank().is_static());
    const bool is_2d_3d = (a_input_shape.first.rank().get_length() == 2);
    const auto& a_static_shapes = a_input_shape.second;
    const size_t iter = [&]() -> size_t {
        const auto it = std::find(a_static_shapes.begin(), a_static_shapes.end(),
                                  targetInputStaticShapes[0]);
        return (it != a_static_shapes.end())
                   ? static_cast<size_t>(std::distance(a_static_shapes.begin(), it))
                   : 0;
    }();

    // Fill A with small random floats.
    ov::test::utils::InputGenerateData gen_data;
    gen_data.range = 2;
    gen_data.resolution = 128;
    gen_data.start_from = -1;

    inputs.insert({function->get_parameters()[0],
                   ov::test::utils::create_and_fill_tensor(elem_type, targetInputStaticShapes[0], gen_data)});

    if (is_2d_3d) {
        const size_t G = b_shape[0];
        const size_t T = targetInputStaticShapes[0][0];

        // Build cumulative end-offsets: offsets[g] = exclusive row end for group g.
        std::vector<int32_t> offsets(G);
        if (!tokens_per_expert.empty() && iter < tokens_per_expert.size()) {
            // Use provided per-iteration distribution.
            const auto& tpe = tokens_per_expert[iter];
            OPENVINO_ASSERT(tpe.size() == G,
                            "tokens_per_expert[", iter, "] must have G=", G,
                            " entries, got ", tpe.size());
            int32_t cum = 0;
            for (size_t g = 0; g < G; ++g) {
                cum += static_cast<int32_t>(tpe[g]);
                offsets[g] = cum;
            }
            OPENVINO_ASSERT(offsets[G - 1] == static_cast<int32_t>(T),
                            "tokens_per_expert[", iter, "] sums to ", offsets[G - 1],
                            " but T=", T);
        } else {
            // Even distribution: ceil_div(T*(g+1), G).
            for (size_t g = 0; g < G; ++g)
                offsets[g] = static_cast<int32_t>((T * (g + 1) + G - 1) / G);
            offsets[G - 1] = static_cast<int32_t>(T);
        }

        ov::Tensor offsets_tensor(ov::element::i32, ov::Shape{G});
        auto* dst = offsets_tensor.data<int32_t>();
        for (size_t g = 0; g < G; ++g)
            dst[g] = offsets[g];

        inputs.insert({function->get_parameters()[1], offsets_tensor});
    }
}

// ---- GroupedMatMulCompressedLayerTest -----------------------------------

std::string GroupedMatMulCompressedLayerTest::getTestCaseName(
    const testing::TestParamInfo<GroupedMatMulCompressedParams>& obj) {
    const auto& [shape_params, act_type, weights_prec, decomp_prec, scale_prec,
                 multiply_type, subtract_type, reshape_on_decomp, group_size,
                 target_device] = obj.param;
    const auto& [a_input_shape, b_shape, tokens_per_expert] = shape_params;

    OPENVINO_ASSERT(a_input_shape.first.rank().is_static());
    const bool is_2d_3d = (a_input_shape.first.rank().get_length() == 2);

    std::ostringstream result;
    result << "Case=" << (is_2d_3d ? "2Dx3D" : "3Dx3D") << "_";
    result << "A_dyn=" << a_input_shape.first << "_";
    result << "A_static=";
    for (size_t i = 0; i < a_input_shape.second.size(); ++i) {
        result << ov::test::utils::vec2str(a_input_shape.second[i]);
        if (i + 1 < a_input_shape.second.size())
            result << ",";
    }
    result << "_B=" << ov::test::utils::vec2str(b_shape) << "_";
    result << "ActET=" << act_type.to_string() << "_";
    result << "WET=" << weights_prec.to_string() << "_";
    result << "DecompPrec=" << decomp_prec.to_string() << "_";
    result << "ScalePrec=" << scale_prec.to_string() << "_";
    result << "Mul=" << multiply_type << "_Sub=" << subtract_type << "_";
    result << "Reshape=" << reshape_on_decomp << "_";
    result << "GrpSz=" << group_size << "_";
    result << "Dev=" << target_device;
    return result.str();
}

void GroupedMatMulCompressedLayerTest::SetUp() {
    const auto& [shape_params, act_type, weights_prec, decomp_prec, scale_prec,
                 multiply_type, subtract_type, reshape_on_decomp, group_size,
                 _targetDevice] = GetParam();
    const auto& [a_input_shape, b_shape, tokens_per_expert] = shape_params;
    targetDevice = _targetDevice;

    OPENVINO_ASSERT(a_input_shape.first.rank().is_static());
    const bool is_2d_3d = (a_input_shape.first.rank().get_length() == 2);
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

    auto param_a = std::make_shared<ov::op::v0::Parameter>(act_type, inputDynamicShapes[0]);

    // Build compressed B via decompression subgraph.
    //
    // b_shape is the pre-transposed weight shape [G, N, K].
    // We pass the PLANAR shape [G, K, N] to initMatMulDecompressionSubgraphQuantization
    // with transpose_weights=true so the internal constant is laid out as [G, N, K]
    // (per-N scales ≡ per-OC). Passing insert_transpose_node=false suppresses the
    // trailing Transpose node, leaving the decompressed output in [G, N, K] form
    // ready for GroupedMatMul.
    ov::Shape b_planar = b_shape;
    std::swap(b_planar[b_planar.size() - 2], b_planar[b_planar.size() - 1]);  // [G, K, N]

    auto const_b = ov::test::utils::initMatMulDecompressionSubgraphQuantization(
        b_planar,
        group_size,
        act_type,
        weights_prec,
        decomp_prec,
        scale_prec,
        true,              // transpose_weights: constant stored as [G, N, K]
        multiply_type,
        subtract_type,
        reshape_on_decomp,
        false,             // insert_transpose_node: omit trailing Transpose → output [G, N, K]
        1);                // seed

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
        "GroupedMatMulCompressed");

    abs_threshold = 0.05f;
    rel_threshold = 0.01f;
}

void GroupedMatMulCompressedLayerTest::generate_inputs(
    const std::vector<ov::Shape>& targetInputStaticShapes) {
    inputs.clear();
    const auto& [shape_params, act_type, weights_prec, decomp_prec, scale_prec,
                 multiply_type, subtract_type, reshape_on_decomp, group_size,
                 _target_device] = GetParam();
    const auto& [a_input_shape, b_shape, tokens_per_expert] = shape_params;

    OPENVINO_ASSERT(a_input_shape.first.rank().is_static());
    const bool is_2d_3d = (a_input_shape.first.rank().get_length() == 2);
    const auto& a_static_shapes = a_input_shape.second;
    const size_t iter = [&]() -> size_t {
        const auto it = std::find(a_static_shapes.begin(), a_static_shapes.end(),
                                  targetInputStaticShapes[0]);
        return (it != a_static_shapes.end())
                   ? static_cast<size_t>(std::distance(a_static_shapes.begin(), it))
                   : 0;
    }();

    ov::test::utils::InputGenerateData gen_data;
    gen_data.range = 2;
    gen_data.resolution = 128;
    gen_data.start_from = -1;

    inputs.insert({function->get_parameters()[0],
                   ov::test::utils::create_and_fill_tensor(act_type, targetInputStaticShapes[0], gen_data)});

    if (is_2d_3d) {
        const size_t G = b_shape[0];
        const size_t T = targetInputStaticShapes[0][0];

        std::vector<int32_t> offsets(G);
        if (!tokens_per_expert.empty() && iter < tokens_per_expert.size()) {
            const auto& tpe = tokens_per_expert[iter];
            OPENVINO_ASSERT(tpe.size() == G);
            int32_t cum = 0;
            for (size_t g = 0; g < G; ++g) {
                cum += static_cast<int32_t>(tpe[g]);
                offsets[g] = cum;
            }
        } else {
            for (size_t g = 0; g < G; ++g)
                offsets[g] = static_cast<int32_t>((T * (g + 1) + G - 1) / G);
            offsets[G - 1] = static_cast<int32_t>(T);
        }

        ov::Tensor offsets_tensor(ov::element::i32, ov::Shape{G});
        auto* dst = offsets_tensor.data<int32_t>();
        for (size_t g = 0; g < G; ++g)
            dst[g] = offsets[g];

        inputs.insert({function->get_parameters()[1], offsets_tensor});
    }
}

}  // namespace test
}  // namespace ov
