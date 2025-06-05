// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/add.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"
#include "transformations_visibility.hpp"

#include "openvino/op/range.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/einsum.hpp"

#include "openvino/cc/pass/itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/tile.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/einsum.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/utils/utils.hpp"

static ov::Output<ov::Node> insert_identity(const ov::Output<ov::Node>& in_node) {
    auto axis_1 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    auto identity_1 = std::make_shared<ov::op::v0::Unsqueeze>(in_node, axis_1);
    return std::make_shared<ov::op::v15::Squeeze>(identity_1, axis_1);
}

using ResultVector = std::vector<std::shared_ptr<ov::op::v0::Result>>;
namespace ov {
namespace pass {

class TRANSFORMATIONS_API PositionIDsReplacer;
class TRANSFORMATIONS_API PositionIDsReplacerQwen;

}  // namespace pass
}  // namespace ov

class ov::pass::PositionIDsReplacer : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("PositionIDsReplacer");
    explicit PositionIDsReplacer(const Output<Node>& position_ids);
};

/**
 * @brief Qwen model expects data processing in order, the "position ids" input is detached and
 * is not explicitly used in the model. The model uses implicitly defined "position ids" based
 * on the past KV cache size.
 *
 * To use this model in Continuous batching mode, we need to apply position_ids and
 * use the corresponding rotary_emb_cos/rotary_emb_sin.
 * For this, we replace
 *      rotary_emb_cos/rotary_emb_sin -> Slice -> Slice
 * With
 *      rotary_emb_cos/rotary_emb_sin -> Gather(by position_ids)
 * Which enables applying RoPE for each token independently of their order in the input tensor.
 */
class ov::pass::PositionIDsReplacerQwen : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("PositionIDsReplacerQwen");
    explicit PositionIDsReplacerQwen(const Output<Node>& position_ids);
};

class ReplaceRoPERangeWithPositionIds : public ov::pass::MatcherPass {

public:
    ReplaceRoPERangeWithPositionIds(const ov::Output<ov::Node>& position_ids, ResultVector& dbg_results, int& layer_index1) {
    using namespace ov::op;
    using namespace ov;
    using namespace ov::pass::pattern;

    MATCHER_SCOPE(ReplaceRoPERangeWithPositionIds);

    auto p_range = wrap_type<ov::op::v4::Range>({any_input(), any_input(), any_input()});
    auto p_inv_freq_source = any_input();
    auto p_einsum = wrap_type<ov::op::v7::Einsum>({p_range, p_inv_freq_source});
    ov::matcher_pass_callback callback = [=](Matcher& m) {
        std::cout << "new transformation start" << std::endl;
        auto pattern_map = m.get_pattern_value_map();
        // 5a) Grab the nodes we matched
        auto range_node = as_type_ptr<ov::op::v4::Range>(pattern_map.at(p_range).get_node_shared_ptr());
        auto inv_freq_node = pattern_map.at(p_inv_freq_source).get_node_shared_ptr();
        auto einsum_node = as_type_ptr<ov::op::v7::Einsum>(pattern_map.at(p_einsum).get_node_shared_ptr());

        if (einsum_node->get_equation() != "i,j->ij") {
            return false;
        }

        std::cout << "position_ids element type: " << position_ids.get_element_type() << std::endl;

        // 5c) Create the “position_ids” processing subgraph
        // Right now we have `position_ids` as an input to this entire pass. We will:
        //   ii)  Tile it to shape [batch, d/2]
        //  iii)  Unsqueeze inv_freq_node to [1, d/2]
        //   iv)  Tile inv_freq to [batch, d/2]
        //    v)  Multiply [batch, d/2] × [batch, d/2] → new_angles [batch, d/2]

        // Extract inv_freq’s shape (should be [d/2])  
        auto inv_freq_shape = inv_freq_node->get_output_partial_shape(0);
        if (inv_freq_shape.rank().is_dynamic() || inv_freq_shape.rank().get_length() != 1) {
            return false;
        }
        int64_t freq_len = inv_freq_shape[0].get_length();  // this is d/2 (e.g. 128)

        // Step (ii): Tile pos_ids to [batch, d/2]
        // We need a 2D tile multipliers tensor = [1, freq_len]
        auto tile_vec_1 = ov::op::v0::Constant::create(element::i64, Shape{2}, {1l, freq_len});
        auto pos_ids_broadcast = std::make_shared<ov::op::v0::Tile>(position_ids, tile_vec_1);
        auto pos_ids_convert = std::make_shared<ov::op::v0::Convert>(pos_ids_broadcast, range_node->output(0).get_element_type());

        // Step (iii): Unsqueeze inv_freq to [1, d/2]
        auto inv_freq_unsqueezed = std::make_shared<v0::Unsqueeze>(inv_freq_node->output(0), v0::Constant::create(element::i64, Shape{}, {0}));

        // Step (iv): Tile inv_freq to [batch, d/2]
        // Build a tile vector = [batch, 1].  But “batch” is unknown at compile time, so:
        // We do: `shape_of(position_ids)[0]` to get “batch”
        auto shape_of_pos_ids = std::make_shared<v3::ShapeOf>(position_ids);
        // shape_of_pos_ids is rank‑1 shape = [batch]; we want shape[0]
        auto batch_dim = std::make_shared<v8::Gather>(
            shape_of_pos_ids->output(0),
            v0::Constant::create(element::i64, Shape{1}, {0}),
            v0::Constant::create(element::i64, Shape{}, {0})
        );  // this is a scalar int64 “batch”

        // Now build tile multipliers: [batch, 1]
        auto tile_vec_2 = std::make_shared<ov::op::v0::Concat>(OutputVector{batch_dim, v0::Constant::create(element::i64, Shape{1}, {1})}, 0);  // shape = [2], but first entry = batch, second = 1

        auto inv_freq_broadcast = std::make_shared<v0::Tile>(inv_freq_unsqueezed, tile_vec_2);  // outputs a [batch, d/2] tensor

        // Step (v): Multiply them → new_angles [batch, d/2]
        auto new_angles = std::make_shared<v1::Multiply>(
            pos_ids_convert, inv_freq_broadcast
        );
        new_angles->set_friendly_name("PositionIDs_new_angles");

        // 5d) Now we need to reconnect the rest of the graph.
        // The original `einsum_node->output(0)` is fed into Cos and Sin. We want to disconnect it,
        // and instead feed `new_angles` into those Cos/Sin nodes. That way Cos(new_angles) = Cos(p_i ⋅ inv_freq).
        //
        // Find all the consumers of `einsum_node->output(0)`.
        auto consumers = einsum_node->output(0).get_target_inputs(); 
        for (auto &input : consumers) {
            input.replace_source_output(new_angles);
        }

        std::cout << "new transformation end" << std::endl;
        return true;
    };

    auto m = std::make_shared<Matcher>(p_einsum, matcher_name);
    register_matcher(m, callback);
    }
};