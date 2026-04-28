// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/paged_attention/paged_causal_conv1d_fusion.hpp"

#include <algorithm>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/paged_causal_conv1d.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/read_value_base.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/keep_const_precision.hpp"
#include "transformations/utils/utils.hpp"

using ov::pass::pattern::any_input;
using ov::pass::pattern::has_static_rank;
using ov::pass::pattern::has_static_shape;
using ov::pass::pattern::rank_equals;
using ov::pass::pattern::wrap_type;

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v8 = ov::op::v8;

namespace {

std::shared_ptr<ov::Node> make_zero_bias(const ov::element::Type& type, const size_t channels) {
    std::vector<float> values(channels, 0.0f);
    return v0::Constant::create(type, ov::Shape{channels}, values);
}

ov::PartialShape make_conv_state_table_shape(const ov::PartialShape& past_state_shape) {
    if (past_state_shape.rank().is_static() && past_state_shape.rank().get_length() == 3) {
        return ov::PartialShape{ov::Dimension::dynamic(), past_state_shape[1], past_state_shape[2]};
    }
    return ov::PartialShape::dynamic(3);
}

}  // namespace

namespace ov::pass {

PagedCausalConv1DFusion::PagedCausalConv1DFusion(ov::pass::paged_attention::PaParams& pa_params,
                                                 std::unordered_set<std::string>& var_ids_to_remove) {
    // Define parameters for the fused PagedCausalConv1D
    pa_params.add("subsequence_begins", ov::element::i32, ov::PartialShape{-1});
    pa_params.add("la.block_indices", ov::element::i32, ov::PartialShape{-1});
    pa_params.add("la.block_indices_begins", ov::element::i32, ov::PartialShape{-1});
    pa_params.add("la.past_lens", ov::element::i32, ov::PartialShape{-1});
    pa_params.add("la.cache_interval", ov::element::i32, ov::PartialShape{-1});

    MATCHER_SCOPE(PagedCausalConv1DFusion);

    auto p_read_value = wrap_type<ov::op::util::ReadValueBase>(has_static_rank() && rank_equals(3));
    auto p_past_via_gather = ov::pass::pattern::optional<v8::Gather>({p_read_value, any_input(), any_input()});
    auto p_token_input = any_input(rank_equals(3));
    auto p_concat_past_first = wrap_type<v0::Concat>({p_past_via_gather, p_token_input}, {{"axis", -1}});
    auto p_concat_token_first = wrap_type<v0::Concat>({p_token_input, p_past_via_gather}, {{"axis", -1}});
    auto p_state_concat =
        std::make_shared<ov::pass::pattern::op::Or>(ov::OutputVector{p_concat_past_first, p_concat_token_first});

    auto p_weight_input = any_input(has_static_shape() && rank_equals(4));
    auto p_group_conv = wrap_type<v1::GroupConvolution>({p_state_concat, p_weight_input});

    auto p_slice_out = wrap_type<v8::Slice>({p_group_conv, any_input(), any_input(), any_input(), any_input()});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS, &pa_params, &var_ids_to_remove](
                                             ov::pass::pattern::Matcher& m) {
        if (transformation_callback(m.get_match_root())) {
            return false;
        }

        const auto& pm = m.get_pattern_value_map();

        const auto state_concat = pm.at(p_state_concat).get_node_shared_ptr();
        const auto group_conv_node = ov::as_type_ptr<v1::GroupConvolution>(pm.at(p_group_conv).get_node_shared_ptr());
        const auto weight_node = pm.at(p_weight_input).get_node_shared_ptr();
        const auto cache_rv = ov::as_type_ptr<ov::op::util::ReadValueBase>(pm.at(p_read_value).get_node_shared_ptr());
        const auto slice_out = pm.at(p_slice_out).get_node_shared_ptr();

        const auto conv_state_table = pa_params.add("conv_state_table." + std::to_string(m_layer_index++),
                                                    cache_rv->get_output_element_type(0),
                                                    make_conv_state_table_shape(cache_rv->get_output_partial_shape(0)));

        enable_keep_const_precision(conv_state_table);
        var_ids_to_remove.insert(cache_rv->get_variable_id());

        auto token_input = pm.at(p_token_input).get_node_shared_ptr();
        const auto past_state = pm.count(p_past_via_gather) ? pm.at(p_past_via_gather).get_node_shared_ptr() : cache_rv;

        const auto& weight_shape = weight_node->get_output_shape(0);
        const size_t hidden_size = weight_shape[0];
        const size_t kernel_size = weight_shape[3];

        const auto& state_pshape = past_state->get_output_partial_shape(0);
        if (!state_pshape[1].compatible(hidden_size) || !state_pshape[2].compatible(kernel_size)) {
            return false;
        }

        const auto& token_pshape = token_input->get_output_partial_shape(0);
        if (token_pshape[1].compatible(hidden_size)) {
            const auto order = v0::Constant::create(ov::element::i64, ov::Shape{3}, {0, 2, 1});
            token_input = std::make_shared<v1::Transpose>(token_input, order);
        } else if (!token_pshape[2].compatible(hidden_size)) {
            return false;
        }

        const auto input_embeds_shape =
            v0::Constant::create(ov::element::i64,
                                 ov::Shape{2},
                                 std::vector<int64_t>{-1, static_cast<int64_t>(hidden_size)});
        const auto input_embeds_node = std::make_shared<v1::Reshape>(token_input, input_embeds_shape, false);

        const auto pa_weight_shape = v0::Constant::create(ov::element::i64,
                                                          ov::Shape{3},
                                                          std::vector<int64_t>{static_cast<int64_t>(hidden_size),
                                                                               static_cast<int64_t>(1),
                                                                               static_cast<int64_t>(kernel_size)});
        const auto weight_reshaped = std::make_shared<v1::Reshape>(weight_node, pa_weight_shape, false);

        const auto bias_node = make_zero_bias(input_embeds_node->get_output_element_type(0), hidden_size);

        const auto paged_conv =
            std::make_shared<ov::op::internal::PagedCausalConv1D>(input_embeds_node,
                                                                  conv_state_table,
                                                                  weight_reshaped,
                                                                  bias_node,
                                                                  pa_params["subsequence_begins"],
                                                                  pa_params["la.block_indices"],
                                                                  pa_params["la.block_indices_begins"],
                                                                  pa_params["la.past_lens"],
                                                                  pa_params["la.cache_interval"]);

        paged_conv->set_friendly_name(group_conv_node->get_friendly_name() + "/PagedCausalConv1D");

        const auto unsqueeze_axis = v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});
        const auto unsqueeze = std::make_shared<v0::Unsqueeze>(paged_conv, unsqueeze_axis);
        unsqueeze->set_friendly_name(group_conv_node->get_friendly_name());

        ov::NodeVector new_nodes{input_embeds_shape,
                                 input_embeds_node,
                                 pa_weight_shape,
                                 weight_reshaped,
                                 bias_node,
                                 paged_conv,
                                 unsqueeze};

        ov::copy_runtime_info(m.get_matched_nodes(), new_nodes);
        ov::replace_node(slice_out, unsqueeze);
        return true;
    };

    const auto matcher = std::make_shared<ov::pass::pattern::Matcher>(p_slice_out, matcher_name);
    register_matcher(matcher, callback);
}

}  // namespace ov::pass
