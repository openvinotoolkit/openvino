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
    ReplaceRoPERangeWithPositionIds(const ov::Output<ov::Node>& unsqueenzed_position_ids, ResultVector& dbg_results, int& layer_index1) {
    using namespace ov::op;
    using namespace ov;
    using namespace ov::pass::pattern;

    MATCHER_SCOPE(ReplaceRoPERangeWithPositionIds);

    auto _const = []() {
        return ov::pass::pattern::wrap_type<v0::Constant>();
    };

    auto reshape = ov::pass::pattern::wrap_type<v1::Reshape>();
    auto unsq = ov::pass::pattern::wrap_type<v0::Unsqueeze>({reshape, any_input()});
    auto p_slice = ov::pass::pattern::wrap_type<v8::Slice>({unsq, any_input(), any_input(), any_input(), any_input()});
    auto unsq_1 = ov::pass::pattern::wrap_type<v0::Unsqueeze>({p_slice, any_input()});


    ov::matcher_pass_callback callback = [=, &layer_index1, &dbg_results](Matcher& m) {
        std::cout << matcher_name << " start" << std::endl;
        auto pvm = m.get_pattern_value_map();
        auto slice = pvm.at(p_slice).get_node_shared_ptr();

        auto gather = std::make_shared<v8::Gather>(slice->input_value(0), unsqueenzed_position_ids, v0::Constant::create(element::i64, {}, {1}));
        auto gather_result = std::make_shared<v0::Result>(gather);
        gather_result->get_output_tensor(0).set_names({"gather_result_" + std::to_string(layer_index1++)});
        dbg_results.push_back({gather_result});

        replace_node(slice, gather);

        std::cout << matcher_name << " end" << std::endl;
        return true;
    };

    auto m = std::make_shared<Matcher>(unsq_1, matcher_name);
    register_matcher(m, callback);
    }
};