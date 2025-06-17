// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rope_fusion.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"
#include "openvino/core/graph_util.hpp"

#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/variadic_split.hpp"
#include "ov_ops/rotary_positional_embeddings.hpp"

namespace ov::intel_gpu {

RoPEFusion::RoPEFusion() {
    add_matcher<RoPEFusionChatGLMHF>();
}

RoPEFusionChatGLMHF::RoPEFusionChatGLMHF() {
    using namespace ov::pass::pattern;

    auto qkv_linear_m = any_input();
    auto qkv_proj_m = wrap_type<ov::op::v1::VariadicSplit>({qkv_linear_m, wrap_type<ov::op::v0::Constant>(), wrap_type<ov::op::v0::Constant>()},
        [](const Output<Node>& node) {
            return node.get_node_shared_ptr()->get_output_size() == 3;
        });
    auto transposed_cur_key_m = wrap_type<ov::op::v1::Reshape>({qkv_proj_m, wrap_type<ov::op::v0::Constant>()});
    auto slice_1_m = wrap_type<ov::op::v1::StridedSlice>({transposed_cur_key_m,
                                                          wrap_type<ov::op::v0::Constant>(),
                                                          wrap_type<ov::op::v0::Constant>(),
                                                          wrap_type<ov::op::v0::Constant>()});
    auto slice_5_m = wrap_type<ov::op::v1::StridedSlice>({transposed_cur_key_m,
                                                          wrap_type<ov::op::v0::Constant>(),
                                                          wrap_type<ov::op::v0::Constant>(),
                                                          wrap_type<ov::op::v0::Constant>()});
    auto slice_2_m = wrap_type<ov::op::v1::StridedSlice>({slice_1_m,
                                                          wrap_type<ov::op::v0::Constant>(),
                                                          wrap_type<ov::op::v0::Constant>(),
                                                          wrap_type<ov::op::v0::Constant>()});
    auto neg_m = wrap_type<ov::op::v1::Multiply>({slice_2_m, wrap_type<ov::op::v0::Constant>()});
    auto unsqueeze_54222_m = wrap_type<ov::op::v1::Reshape>({neg_m, wrap_type<ov::op::v0::Constant>()});
    auto slice_3_m = wrap_type<ov::op::v1::StridedSlice>({slice_1_m,
                                                          wrap_type<ov::op::v0::Constant>(),
                                                          wrap_type<ov::op::v0::Constant>(),
                                                          wrap_type<ov::op::v0::Constant>()});

    auto cos_m = wrap_type<ov::op::v1::StridedSlice>({any_input(),
                                                      wrap_type<ov::op::v0::Constant>(),
                                                      wrap_type<ov::op::v0::Constant>(),
                                                      wrap_type<ov::op::v0::Constant>()});
    auto sin_m = wrap_type<ov::op::v1::StridedSlice>({any_input(),
                                                      wrap_type<ov::op::v0::Constant>(),
                                                      wrap_type<ov::op::v0::Constant>(),
                                                      wrap_type<ov::op::v0::Constant>()});
    auto gather_1_m = wrap_type<ov::op::v8::Gather>({cos_m,
                                                     wrap_type<ov::op::v0::Constant>(),
                                                     wrap_type<ov::op::v0::Constant>()});
    auto gather_3_m = wrap_type<ov::op::v8::Gather>({sin_m,
                                                     wrap_type<ov::op::v0::Constant>(),
                                                     wrap_type<ov::op::v0::Constant>()});

    auto mul_m = wrap_type<ov::op::v1::Multiply>({slice_1_m, gather_1_m});
    auto unsqueeze_54223_m = wrap_type<ov::op::v1::Reshape>({slice_3_m, wrap_type<ov::op::v0::Constant>()});
    auto concat_matches = [](const ov::Output<ov::Node>& output) -> bool {
        if (auto concat = ov::as_type_ptr<ov::op::v0::Concat>(output.get_node_shared_ptr())) {
            return ov::pass::pattern::rank_equals(5)(output) && concat->get_axis() == -1;
        }
        return false;
    };
    auto stack_m = ov::pass::pattern::wrap_type<ov::op::v0::Concat>({unsqueeze_54222_m, unsqueeze_54223_m}, concat_matches);
    auto flatten_m = wrap_type<ov::op::v1::Reshape>({stack_m, wrap_type<ov::op::v0::Constant>()});
    auto mul_1_m = wrap_type<ov::op::v1::Multiply>({flatten_m, gather_3_m});
    auto add_m = wrap_type<ov::op::v1::Add>({mul_m, mul_1_m});
    auto concat_m = ov::pass::pattern::wrap_type<ov::op::v0::Concat>({add_m, slice_5_m});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto root = ov::as_type_ptr<ov::op::v0::Concat>(m.get_match_root());
        if (!root || transformation_callback(root)) {
            return false;
        }

        int64_t total_size_q = 0;
        int64_t total_size_k = 0;
        int64_t total_size_v = 0;
        const auto& qkv_proj = pattern_map.at(qkv_proj_m).get_node_shared_ptr();
        if (auto split_lengths_constant = ov::as_type_ptr<ov::op::v0::Constant>(qkv_proj->get_input_node_shared_ptr(2))) {
            std::vector<int64_t> split_lengths = split_lengths_constant->cast_vector<int64_t>();
            if (split_lengths.size() != 3) {
                return false;
            }
            total_size_q = split_lengths[0];
            total_size_k = split_lengths[1];
            total_size_v = split_lengths[2];
        }

        int32_t head_cnt = 0;
        int32_t head_size = 0;
        const auto& transposed_cur_key = pattern_map.at(transposed_cur_key_m).get_node_shared_ptr();
        if (auto pattern_constant = ov::as_type_ptr<ov::op::v0::Constant>(transposed_cur_key->get_input_node_shared_ptr(1))) {
            std::vector<int32_t> pattern = pattern_constant->cast_vector<int32_t>();
            if (pattern.size() != 4 || pattern[0] != -1 || pattern[2] != 1) {
                return false;
            }
            head_cnt = pattern[1];
            head_size = pattern[3];
            if (head_cnt == 0 || head_size == 0) {
                return false;
            }
        }
        size_t split_output_id = transposed_cur_key->get_input_source_output(0).get_index();

        int32_t ndims = 0;
        const auto& slice_1 = pattern_map.at(slice_1_m).get_node_shared_ptr();
        if (auto end_constant = ov::as_type_ptr<ov::op::v0::Constant>(slice_1->get_input_node_shared_ptr(2))) {
            std::vector<int32_t> end = end_constant->cast_vector<int32_t>();
            if (end.size() != 4 || end[0] != 0 || end[1] != 0 || end[2] != 0) {
                return false;
            }
            ndims = end[3];
            if (ndims == 0) {
                return false;
            }
        }
        const auto& slice_5 = pattern_map.at(slice_5_m).get_node_shared_ptr();
        if (auto begin_constant = ov::as_type_ptr<ov::op::v0::Constant>(slice_5->get_input_node_shared_ptr(1))) {
            std::vector<int32_t> begin = begin_constant->cast_vector<int32_t>();
            if (begin.size() != 4 || begin[0] != 0 || begin[1] != 0 || begin[2] != 0) {
                return false;
            }
            if (ndims != begin[3]) {
                return false;
            }
        }

        // std::cout << "RoPEFusionChatGLMHF::callback | name="
        //           << root->get_friendly_name()
        //           << ", split_output_id = " << split_output_id
        //           << ", total_size_q = " << total_size_q
        //           << ", total_size_k = " << total_size_k
        //           << ", total_size_v = " << total_size_v
        //           << ", head_cnt = " << head_cnt
        //           << ", head_size = " << head_size
        //           << ", ndims = " << ndims
        //           << std::endl;

        ov::op::internal::RoPE::Config config;
        OutputVector new_args;
        config.rotary_ndims = static_cast<size_t>(ndims);
        config.is_chatglm = true;
        config.support_2d_rope = true;
        config.head_cnt = static_cast<size_t>(head_cnt);
        config.head_size = static_cast<size_t>(head_size);

        if (split_output_id == 0) {
            config.slice_start = 0;
            config.slice_stop = static_cast<size_t>(total_size_q);
        } else {
            config.slice_start = static_cast<size_t>(total_size_q);
            config.slice_stop = static_cast<size_t>(config.slice_start + total_size_k);
        }

        new_args.push_back(pattern_map.at(qkv_linear_m));
        new_args.push_back(pattern_map.at(cos_m));
        new_args.push_back(pattern_map.at(sin_m));

        auto old_node = root;

        auto new_node = std::make_shared<ov::op::internal::RoPE>(new_args, config);
        new_node->set_friendly_name(old_node->get_friendly_name());
        ov::copy_runtime_info({root->get_input_node_shared_ptr(0), root}, new_node);
        ov::replace_node(old_node, new_node);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(concat_m, "RoPEFusionChatGLMHF");
    this->register_matcher(m, callback);
}

}  // namespace ov::intel_gpu
