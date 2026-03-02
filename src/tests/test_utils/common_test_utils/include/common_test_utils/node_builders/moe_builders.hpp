// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <iostream>
#include <memory>
#include <optional>
#include <utility>
#include <vector>
#include "openvino/core/model.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/gather_elements.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "common_test_utils/subgraph_builders/weights_decompression_builders.hpp"

namespace ov {
namespace test {

// Matches the definition in shared_test_classes/base/ov_subgraph.hpp;
// redeclaring an identical type alias in the same namespace is valid C++.
using InputShape = std::pair<ov::PartialShape, std::vector<ov::Shape>>;

struct MoePatternParams {
    InputShape data_shape;
    size_t topk;
    size_t number_of_experts;
    size_t intermediate_size;
};

enum class MoERoutingType {
    SOFTMAX,       ///< Softmax -> TopK -> normalize routing
    SIGMOID_BIAS,  ///< Sigmoid -> Add(bias) -> TopK routing
};

/// Outputs produced by the routing subgraph that feed into MOECompressed.
struct MoeRoutingResult {
    ov::Output<ov::Node> unsqueeze_moe;  ///< routing weights tensor (4D)
    ov::Output<ov::Node> topk_indices;   ///< expert indices
};

/// Build the routing subgraph starting from a pre-computed routing_weights node
/// (result of MatMul(hidden_states, router_weights)).
///
/// Softmax branch:
///   routing_weights -> Softmax -> TopK -> ReduceSum -> Divide (norm)
///   ShapeOf(topk_idx) -> Gather(0) -> Unsqueeze -> ]
///   Constant(ne)      -> Unsqueeze              -> ] Concat -> Broadcast
///   Broadcast -> ScatterElementsUpdate(SUM) -> Transpose -> Reshape -> Unsqueeze
///
/// Sigmoid+bias branch:
///   routing_weights -> Sigmoid -> Add(bias) -> TopK -> Convert(i32)
///   Sigmoid -> GatherElements -> ReduceSum -> Add(eps) -> Divide -> Slice
///   ShapeOf(indices) -> Gather(0) -> Unsqueeze -> ]
///   Constant(ne)     -> Unsqueeze              -> ] Concat -> Broadcast
///   Broadcast -> ScatterElementsUpdate -> Transpose -> Reshape -> Unsqueeze
inline MoeRoutingResult build_moe_routing_subgraph(const ov::Output<ov::Node>& routing_weights,
                                                    MoERoutingType routing_type,
                                                    ov::element::Type data_precision,
                                                    size_t number_of_experts,
                                                    size_t topk) {
    using namespace ov::op;

    MoeRoutingResult result;

    if (routing_type == MoERoutingType::SOFTMAX) {
        // ---- Softmax branch ----
        auto router_softmax = std::make_shared<v8::Softmax>(routing_weights, 1);

        // TopK: i32 k-constant, no explicit output-type -> default i32 indices
        auto k_const = v0::Constant::create(ov::element::i32, ov::Shape{}, {static_cast<int32_t>(topk)});
        auto router_topk = std::make_shared<v11::TopK>(
            router_softmax, k_const, 1,
            v11::TopK::Mode::MAX,
            v11::TopK::SortType::SORT_VALUES);

        auto reduce_sm = std::make_shared<v1::ReduceSum>(
            router_topk->output(0),
            v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1}),
            true);
        auto scatter_w_sm = ov::Output<ov::Node>{std::make_shared<v1::Divide>(router_topk->output(0), reduce_sm)};
        auto topk_idx_sm  = router_topk->output(1);

        // seq dim: ShapeOf(topk_idx) -> scalar Gather(idx=0,axis=0) -> Unsqueeze([0]) -> [1]{seq}
        auto shapeof_sm = std::make_shared<v3::ShapeOf>(topk_idx_sm);
        auto gather_sm  = std::make_shared<v8::Gather>(
            shapeof_sm,
            v0::Constant::create(ov::element::i64, ov::Shape{}, {0}),
            v0::Constant::create(ov::element::i64, ov::Shape{}, {0}));
        auto unsq_seq_sm = std::make_shared<v0::Unsqueeze>(
            gather_sm,
            v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0}));

        // ne dim: scalar constant -> Unsqueeze([0]) -> [1]{ne}
        auto ne_scalar_sm = v0::Constant::create(
            ov::element::i64, ov::Shape{}, {static_cast<int64_t>(number_of_experts)});
        auto unsq_ne_sm = std::make_shared<v0::Unsqueeze>(
            ne_scalar_sm,
            v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0}));

        // broadcast target [seq, ne]
        ov::OutputVector bcast_vec_sm;
        bcast_vec_sm.push_back(unsq_seq_sm);
        bcast_vec_sm.push_back(unsq_ne_sm);
        auto bcast_shape_sm = std::make_shared<v0::Concat>(bcast_vec_sm, 0);

        // reshape target [ne, seq, 1]
        auto one_sm = v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
        ov::OutputVector reshape_vec_sm;
        reshape_vec_sm.push_back(unsq_ne_sm);
        reshape_vec_sm.push_back(unsq_seq_sm);
        reshape_vec_sm.push_back(one_sm);
        auto reshape_shape_sm = std::make_shared<v0::Concat>(reshape_vec_sm, 0);

        auto zero_sm    = v0::Constant::create(scatter_w_sm.get_element_type(), ov::Shape{1}, {0});
        auto bcast_sm   = std::make_shared<v3::Broadcast>(zero_sm, bcast_shape_sm);
        auto scatter_sm = std::make_shared<v12::ScatterElementsUpdate>(
            bcast_sm,
            topk_idx_sm,
            scatter_w_sm,
            v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1}),
            v12::ScatterElementsUpdate::Reduction::SUM);
        auto transp_sm  = std::make_shared<v1::Transpose>(
            scatter_sm,
            v0::Constant::create(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{1, 0}));
        auto reshape_sm = std::make_shared<v1::Reshape>(transp_sm, reshape_shape_sm, false);
        result.unsqueeze_moe = std::make_shared<v0::Unsqueeze>(
            reshape_sm,
            v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{3}));
        result.topk_indices = topk_idx_sm;
        return result;
    }

    // ---- SIGMOID_BIAS branch ----
    auto sigmoid = std::make_shared<v0::Sigmoid>(routing_weights);
    auto bias    = v0::Constant::create(
        data_precision, ov::Shape{1, number_of_experts},
        std::vector<float>(number_of_experts, 0.1f));
    auto sig_add = std::make_shared<v1::Add>(sigmoid, bias);

    auto router_topk_sig = std::make_shared<v11::TopK>(
        sig_add,
        v0::Constant::create(ov::element::i64, ov::Shape{}, {static_cast<int64_t>(topk)}),
        -1,
        v11::TopK::Mode::MAX,
        v11::TopK::SortType::SORT_VALUES,
        ov::element::i64);

    auto convert_topk  = std::make_shared<v0::Convert>(router_topk_sig->output(1), ov::element::i32);
    auto topk_idx_sig  = ov::Output<ov::Node>{convert_topk};

    auto gather_el  = std::make_shared<v6::GatherElements>(sigmoid, convert_topk, 1);
    auto reduce_sig = std::make_shared<v1::ReduceSum>(
        gather_el,
        v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1}),
        true);
    auto eps     = v0::Constant::create(data_precision, ov::Shape{1, 1}, {1e-6f});
    auto add_eps = std::make_shared<v1::Add>(reduce_sig, eps);
    auto norm    = std::make_shared<v1::Divide>(gather_el, add_eps);

    auto sl_stop       = std::make_shared<v3::ShapeOf>(convert_topk, ov::element::i32);
    auto scatter_w_sig = ov::Output<ov::Node>{std::make_shared<v8::Slice>(
        norm,
        v0::Constant::create(ov::element::i32, ov::Shape{2}, std::vector<int32_t>{0, 0}),
        sl_stop,
        v0::Constant::create(ov::element::i32, ov::Shape{2}, std::vector<int32_t>{1, 1}),
        v0::Constant::create(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{0, 1}))};

    // broadcast target [seq, ne]
    auto shapeof_sig  = std::make_shared<v3::ShapeOf>(convert_topk, ov::element::i64);
    auto gather_sig   = std::make_shared<v8::Gather>(
        shapeof_sig,
        v0::Constant::create(ov::element::i64, ov::Shape{}, {0}),
        v0::Constant::create(ov::element::i64, ov::Shape{}, {0}));
    auto unsq_seq_sig = std::make_shared<v0::Unsqueeze>(
        gather_sig,
        v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0}));

    auto ne_scalar_sig = v0::Constant::create(
        ov::element::i64, ov::Shape{}, {static_cast<int64_t>(number_of_experts)});
    auto unsq_ne_sig   = std::make_shared<v0::Unsqueeze>(
        ne_scalar_sig,
        v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0}));

    ov::OutputVector bcast_vec_sig;
    bcast_vec_sig.push_back(unsq_seq_sig);
    bcast_vec_sig.push_back(unsq_ne_sig);
    auto bcast_shape_sig = std::make_shared<v0::Concat>(bcast_vec_sig, 0);

    auto zero_sig  = v0::Constant::create(scatter_w_sig.get_element_type(), ov::Shape{1}, {0});
    auto bcast_sig = std::make_shared<v3::Broadcast>(zero_sig, bcast_shape_sig);

    auto scatter_sig = std::make_shared<v12::ScatterElementsUpdate>(
        bcast_sig,
        topk_idx_sig,
        scatter_w_sig,
        v0::Constant::create(ov::element::i64, ov::Shape{}, std::vector<int64_t>{1}));

    auto transp_sig = std::make_shared<v1::Transpose>(
        scatter_sig,
        v0::Constant::create(ov::element::i32, ov::Shape{2}, std::vector<int32_t>{1, 0}));

    // reshape [ne, 1, seq]
    auto one_sig = v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
    ov::OutputVector reshape_vec_sig;
    reshape_vec_sig.push_back(unsq_ne_sig);
    reshape_vec_sig.push_back(one_sig);
    reshape_vec_sig.push_back(unsq_seq_sig);
    auto reshape_shape_sig = std::make_shared<v0::Concat>(reshape_vec_sig, 0);
    auto reshape_sig = std::make_shared<v1::Reshape>(transp_sig, reshape_shape_sig, false);

    result.unsqueeze_moe = std::make_shared<v0::Unsqueeze>(
        reshape_sig,
        v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{3}));
    result.topk_indices = topk_idx_sig;
    return result;
}

std::shared_ptr<ov::Model> initMoE2GeMMSubgraph(const MoePatternParams& moe_params,
                                                 const ov::element::Type data_precision,
                                                 const ov::element::Type weights_precision,
                                                 const bool use_weight_decompression = false,
                                                 const std::optional<ov::element::Type> decompression_precision = std::nullopt,
                                                 const std::optional<ov::element::Type> scale_precision = std::nullopt,
                                                 const std::optional<ov::test::utils::DecompressionType> decompression_multiply_type = std::nullopt,
                                                 const std::optional<ov::test::utils::DecompressionType> decompression_subtract_type = std::nullopt,
                                                 const std::optional<bool> reshape_on_decompression = std::nullopt,
                                                 const std::optional<int> decompression_group_size = std::nullopt);

std::shared_ptr<ov::Model> initMoE3GeMMSubgraph(const MoePatternParams& moe_params,
                                                 const ov::element::Type data_precision,
                                                 const ov::element::Type weights_precision,
                                                 const bool use_weight_decompression = false,
                                                 const std::optional<ov::element::Type> decompression_precision = std::nullopt,
                                                 const std::optional<ov::element::Type> scale_precision = std::nullopt,
                                                 const std::optional<ov::test::utils::DecompressionType> decompression_multiply_type = std::nullopt,
                                                 const std::optional<ov::test::utils::DecompressionType> decompression_subtract_type = std::nullopt,
                                                 const std::optional<bool> reshape_on_decompression = std::nullopt,
                                                 const std::optional<int> decompression_group_size = std::nullopt,
                                                 MoERoutingType routing_type = MoERoutingType::SOFTMAX,
                                                 std::optional<ov::element::Type> input_precision = std::nullopt);

}  // namespace test
}  // namespace ov
