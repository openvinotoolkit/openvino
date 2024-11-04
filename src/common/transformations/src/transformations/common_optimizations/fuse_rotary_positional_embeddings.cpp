// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/fuse_rotary_positional_embeddings.hpp"

#include <cstdint>
#include <limits>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/util/shape_of_base.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset6.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "ov_ops/rotary_positional_embeddings.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "transformations/utils/gen_pattern.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov::gen_pattern;

ov::pass::RoPEFusionGPTNEOX::RoPEFusionGPTNEOX() {
    MATCHER_SCOPE(RoPEFusionGPTNEOX);

    // rope pattern matching triggers a little design flaw:
    //   y1 = mul(x, cos)
    //   y2 = mul(x, sin)
    //   y = add(y1, y2)
    // there is a chance that in 'y1' branch, pattern x is mapped to actual value of cos (mul is commutable)
    // this leads to the matching failure of 'y2' branch, because cos didn't appear in that
    // branch.
    // so here we use a WA, only match the path of rotate_hal(x)*sin and check the x*cos path
    // in the callback
    auto x = makePattern(ov::Rank(4));
    auto x_or_cos1 = makePattern(ov::Rank(4));
    auto x_or_cos2 = makePattern(ov::Rank(4));
    auto t_sin = makePattern(ov::Rank(4));

    auto half_ndims = ov::gen_pattern::Symbol("half_ndims");

    auto varsplit = makePattern<opset1::VariadicSplit>({x, 3, {half_ndims, ov::gen_pattern::Symbol("end")}});
    varsplit->set_output_size(2);

    auto int32_max = std::numeric_limits<std::int32_t>::max();

    // rotate half : [-x2, x1]
    auto x2 = GenSlice(x, half_ndims, int32_max, 1, 3);
    auto x2neg = makePattern<opset1::Multiply>({x2 | varsplit->output(1), -1.0f}, {{"auto_broadcast", "numpy"}});
    auto x1 = GenSlice(x, 0, half_ndims, 1, 3);
    auto x_rotate_half = makePattern<opset1::Concat>({x2neg, x1 | varsplit->output(0)}, {{"axis", -1}});

    auto mul_cos = makePattern<opset1::Multiply>({x_or_cos1, x_or_cos2}, {{"auto_broadcast", "numpy"}});
    auto mul_sin = makePattern<opset1::Multiply>({x_rotate_half, t_sin}, {{"auto_broadcast", "numpy"}});

    // [x1, x2]*cos + [-x2, x1]*sin
    auto result = makePattern<opset1::Add>({mul_cos, mul_sin}, {{"auto_broadcast", "numpy"}});

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        PatternValidator validator(m);
        if (!validator) {
            return false;
        }

        const auto& pattern_map = m.get_pattern_value_map();
        auto root = m.get_match_root();

        // check mul(x, cos) exists
        Output<Node> v_cos;
        if (pattern_map.at(x_or_cos1) == pattern_map.at(x)) {
            v_cos = pattern_map.at(x_or_cos2);
        } else if (pattern_map.at(x_or_cos2) == pattern_map.at(x)) {
            v_cos = pattern_map.at(x_or_cos1);
        } else {
            // not a RoPE
            return false;
        }

        op::internal::RoPE::Config config;
        OutputVector new_args;
        config.rotary_ndims = 2ul * static_cast<size_t>(validator["half_ndims"]);

        new_args.push_back(pattern_map.at(x));
        new_args.push_back(v_cos);
        new_args.push_back(pattern_map.at(t_sin));

        auto old_node = root;
        auto new_node = std::make_shared<op::internal::RoPE>(new_args, config);
        new_node->set_friendly_name(old_node->get_friendly_name());
        ov::copy_runtime_info({pattern_map.at(x2neg).get_node_shared_ptr(),
                               pattern_map.at(x_rotate_half).get_node_shared_ptr(),
                               pattern_map.at(mul_cos).get_node_shared_ptr(),
                               pattern_map.at(mul_sin).get_node_shared_ptr(),
                               pattern_map.at(result).get_node_shared_ptr()},
                              new_node);

        ov::replace_node(old_node, new_node);

        // this new node may match following additional matchers
        register_new_node(new_node);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(result, matcher_name);
    this->register_matcher(m, callback);
}

ov::pass::RoPEFusionCosSinPreprocess::RoPEFusionCosSinPreprocess() {
    MATCHER_SCOPE(RoPEFusionCosSinPreprocess);

    auto cos_const = makePattern<opset1::Constant>({});  // "f32[1,1,2048,24]"
    auto sin_const = makePattern<opset1::Constant>({});  // "f32[1,1,2048,24]"

    auto node_batch_size = makePattern("i32[1]");
    auto tile_batch = makePattern("i32[1]");
    auto gather_positions = makePattern("i32[?,?,?,?]");

    auto prepare_cos_sin_gptneox = [&](std::shared_ptr<Node> const_tab) {
        auto slice = makePattern<ov::opset8::Slice>({const_tab, {0}, node_batch_size, {1}, {0}});
        auto strided_slice = GenStridedSlice(const_tab, {0}, node_batch_size, {1}, 0);
        return makePattern<opset6::GatherElements>({strided_slice | slice, gather_positions}, {{"axis", 2}});
    };

    auto seq_len = makePattern("i32[1]");
    auto gather_positions_2d = makePattern("i32[?,?]");

    auto head_dims = ov::gen_pattern::Symbol("head_dims");
    auto prepare_cos_sin_llama = [&](std::shared_ptr<Node> const_tab) {
        auto ScatterUpdate = makePattern<opset3::ScatterUpdate>({{0, 0, 0}, 2, seq_len, 0});
        auto slice_Slice = makePattern<ov::opset8::Slice>({const_tab, {0}, seq_len, {1}, {2}});
        auto slice_StridedSlice = GenStridedSlice(const_tab, {0, 0, 0}, ScatterUpdate, {1, 1, 1}, 2);
        auto squeeze = makePattern<opset1::Reshape>({slice_StridedSlice | slice_Slice, {-1, head_dims}});
        auto index_Gather = makePattern<opset8::Gather>({squeeze, gather_positions_2d, 0}, {{"batch_dims", 0}});

        // another simplified pattern for gathering at position_ids
        auto slice_Slice2 = makePattern<ov::opset8::Slice>({const_tab, {0}, seq_len, {1}, {0}});
        auto slice_StridedSlice2 = GenStridedSlice(const_tab, {0}, seq_len, {1}, 0);
        auto index_Gather2 = makePattern<opset8::Gather>({slice_Slice2 | slice_StridedSlice2, gather_positions_2d, 0},
                                                         {{"batch_dims", 0}});

        auto unsqueeze = makePattern<opset1::Reshape>({index_Gather | index_Gather2, {1, 1, -1, head_dims}});
        auto unsqueeze2 = makePattern<opset1::Unsqueeze>({index_Gather2, 1});

        return unsqueeze2 | unsqueeze;
    };

    auto cos_tab = prepare_cos_sin_gptneox(cos_const) | prepare_cos_sin_llama(cos_const);
    auto sin_tab = prepare_cos_sin_gptneox(sin_const) | prepare_cos_sin_llama(sin_const);

    auto x = makePattern(ov::Rank(4));
    auto rope = makePattern<op::internal::RoPE>({x, cos_tab, sin_tab});

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        PatternValidator validator(m);
        if (!validator) {
            return false;
        }
        const auto& pattern_map = m.get_pattern_value_map();
        auto root = m.get_match_root();
        auto rope_node = as_type_ptr<op::internal::RoPE>(pattern_map.at(rope).get_node_shared_ptr());
        if (!rope_node)
            return false;

        if (pattern_map.count(cos_const)) {
            rope_node->set_argument(1, pattern_map.at(cos_const));
        }
        if (pattern_map.count(sin_const)) {
            rope_node->set_argument(2, pattern_map.at(sin_const));
        }

        auto config = rope_node->get_config();
        if (pattern_map.count(gather_positions)) {
            auto arg_id = rope_node->get_input_size();
            rope_node->set_argument(arg_id, pattern_map.at(gather_positions));
            config.gather_position_arg_id = static_cast<int>(arg_id);
        } else if (pattern_map.count(gather_positions_2d)) {
            auto arg_id = rope_node->get_input_size();
            rope_node->set_argument(arg_id, pattern_map.at(gather_positions_2d));
            config.gather_position_arg_id = static_cast<int>(arg_id);
        }
        rope_node->set_config(config);
        rope_node->validate_and_infer_types();
        register_new_node(rope_node);
        return true;
    };
    auto m = std::make_shared<ov::pass::pattern::Matcher>(rope, matcher_name);
    this->register_matcher(m, callback);
}

// only a fraction of head_size is rotary-embedded
ov::pass::RoPEFusionIOSlicing::RoPEFusionIOSlicing() {
    MATCHER_SCOPE(RoPEFusionIOSlicing);
    auto int32_max = std::numeric_limits<std::int32_t>::max();
    auto data = makePattern(ov::Rank(4));
    auto ndims = ov::gen_pattern::Symbol("ndims");

    auto varsplit = makePattern<opset1::VariadicSplit>({data, 3, {ndims, ov::gen_pattern::Symbol("end")}});
    varsplit->set_output_size(2);

    auto x = GenSlice(data, 0, ndims, 1, 3);
    auto y = GenSlice(data, ndims, int32_max, 1, 3);
    auto x_emb = makePattern<op::internal::RoPE>({x | varsplit->output(0), {}, {}}) |
                 makePattern<op::internal::RoPE>({x | varsplit->output(0), {}, {}, {}});
    auto result = makePattern<opset1::Concat>({x_emb, y | varsplit->output(1)}, {{"axis", -1}});

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto root = m.get_match_root();

        auto rope_node = as_type_ptr<op::internal::RoPE>(root->input_value(0).get_node_shared_ptr());
        if (!rope_node)
            return false;

        PatternValidator validator(m);
        if (!validator) {
            return false;
        }
        auto ndims = validator["ndims"];

        const auto& config = rope_node->get_config();
        if (config.rotary_ndims != ndims)
            return false;

        // remove slice & concat
        rope_node->set_argument(0, pattern_map.at(data));
        rope_node->set_friendly_name(root->get_friendly_name());
        ov::copy_runtime_info({rope_node, pattern_map.at(result).get_node_shared_ptr()}, rope_node);
        ov::replace_node(root, rope_node);

        rope_node->validate_and_infer_types();
        register_new_node(rope_node);
        return true;
    };
    auto m = std::make_shared<ov::pass::pattern::Matcher>(result, matcher_name);
    this->register_matcher(m, callback);
}

ov::pass::RoPEFusionPreprocess::RoPEFusionPreprocess() {
    MATCHER_SCOPE(RoPEFusionPreprocess);

    // gptneox-preprocess of input data
    auto input_to_slice = makePattern(ov::Rank(4));
    auto input_to_trans = makePattern(ov::Rank(4));  // no need to slice from 3S

    // in some model qkv prejection is combined and
    // needs to be sliced before RoPE
    auto slice_start = ov::gen_pattern::Symbol("slice_start");
    auto slice_stop = ov::gen_pattern::Symbol("slice_stop");
    auto input_slice = GenSlice(input_to_slice, slice_start, slice_stop, 1, 3);

    // some model will transpose from [B,L,H,S] to [B,H,L,S] before RoPE
    auto x = makePattern<opset1::Transpose>({input_slice | input_to_trans, {0, 2, 1, 3}});
    auto result = makePattern<op::internal::RoPE>({x, {}, {}}) | makePattern<op::internal::RoPE>({x, {}, {}, {}});

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        PatternValidator validator(m);
        if (!validator) {
            return false;
        }

        const auto& pattern_map = m.get_pattern_value_map();
        auto root = m.get_match_root();
        auto rope_node = as_type_ptr<op::internal::RoPE>(root);
        if (!rope_node)
            return false;

        auto config = rope_node->get_config();
        if (pattern_map.count(input_to_slice)) {
            config.slice_start = static_cast<size_t>(validator["slice_start"]);
            config.slice_stop = static_cast<size_t>(validator["slice_stop"]);
            config.input_trans0213 = true;
            rope_node->set_argument(0, pattern_map.at(input_to_slice));
        } else if (pattern_map.count(input_to_trans)) {
            config.input_trans0213 = true;
            rope_node->set_argument(0, pattern_map.at(input_to_trans));
        } else {
            return false;
        }
        rope_node->set_config(config);
        rope_node->validate_and_infer_types();
        register_new_node(rope_node);
        return true;
    };
    auto m = std::make_shared<ov::pass::pattern::Matcher>(result, matcher_name);
    this->register_matcher(m, callback);
}

ov::pass::RoPEFusionGPTJ::RoPEFusionGPTJ() {
    MATCHER_SCOPE(RoPEFusionGPTJ);

    auto int32_max = std::numeric_limits<std::int32_t>::max();
    auto ndims = ov::gen_pattern::Symbol("ndims");

    auto view_Reshape = makePattern(ov::Rank(4));

    // view_Reshape : B,L,H,S
    auto slice_Slice_965 = GenSlice(view_Reshape, 0, ndims, 1, 3);

    auto varsplit_view_Reshape =
        makePattern<opset1::VariadicSplit>({view_Reshape, 3, {ndims, ov::gen_pattern::Symbol("end")}});
    varsplit_view_Reshape->set_output_size(2);

    auto gather_sin_cos = makePattern("f32");

    auto varsplit = makePattern<opset1::VariadicSplit>({gather_sin_cos, -1, {ndims / 2, -1}});
    varsplit->set_output_size(2);
    // Reshape or UnSqueeze should both be support
    auto unsqueeze_sin = makePattern<opset1::Reshape>({varsplit->output(0), {1, -1, 1, 32}}) |
                         makePattern<opset1::Unsqueeze>({varsplit->output(0), 2});
    auto unsqueeze_cos = makePattern<opset1::Reshape>({varsplit->output(1), {1, -1, 1, 32}}) |
                         makePattern<opset1::Unsqueeze>({varsplit->output(1), 2});
    // repeate cos/sin table
    auto const_idx = makeConst(ov::element::i32, ov::PartialShape::dynamic(), [](const ov::op::v0::Constant& node) {
        const auto& vec = node.get_vector<int32_t>();
        int32_t v = 0;
        for (size_t i = 0; i < vec.size(); i += 2, v++) {
            if (vec[i] != v || vec[i + 1] != v)
                return false;
        }
        return true;
    });
    auto repeat_interleave_sin = makePattern<opset8::Gather>({unsqueeze_sin, const_idx, 3}, {{"batch_dims", 0}});
    auto repeat_interleave_cos = makePattern<opset8::Gather>({unsqueeze_cos, const_idx, 3}, {{"batch_dims", 0}});

    // x interleave (-x[:,:,:, 1::2], x[:,:,:, 0::2])
    auto slice_Slice_1174 = GenSlice(slice_Slice_965 | varsplit_view_Reshape->output(0), 1, int32_max, 2, 3);

    auto neg_Multiply_1177 = makePattern<opset1::Multiply>({slice_Slice_1174, -1.0f}, {{"auto_broadcast", "numpy"}});
    auto Unsqueeze_65524 = makePattern<opset1::Unsqueeze>({neg_Multiply_1177, -1});

    auto slice_Slice_1168 = GenSlice(slice_Slice_965 | varsplit_view_Reshape->output(0), 0, int32_max, 2, 3);
    auto Unsqueeze_65525 = makePattern<opset1::Unsqueeze>({slice_Slice_1168, -1});
    auto stack_1182 = makePattern<opset1::Concat>({Unsqueeze_65524, Unsqueeze_65525}, {{"axis", -1}});

    auto ShapeOf_169068 = makePattern<opset1::ShapeOf>({stack_1182});
    auto flatten_Slice_1194 = GenSlice(ShapeOf_169068, 0, 3, 1, 0);
    auto flatten_Concat_1197 = makePattern<opset1::Concat>({flatten_Slice_1194, {-1}}, {{"axis", 0}});
    // If with special zero, no need to use shapeof to get full shape
    auto flatten_Reshape_1198 = makePattern<opset1::Reshape>({stack_1182, flatten_Concat_1197});
    auto flatten_Reshape_Zero =
        makePattern<opset1::Reshape>({stack_1182, ov::pass::pattern::any_input()}, {{"special_zero", true}});

    // x*cos [B,L,H,ndims]
    auto mul_cos =
        makePattern<opset1::Multiply>({slice_Slice_965 | varsplit_view_Reshape->output(0), repeat_interleave_cos},
                                      {{"auto_broadcast", "numpy"}});
    auto mul_sin = makePattern<opset1::Multiply>({flatten_Reshape_1198 | flatten_Reshape_Zero, repeat_interleave_sin},
                                                 {{"auto_broadcast", "numpy"}});

    // *cos + *sin
    auto rotary_emb = makePattern<opset1::Add>({mul_cos, mul_sin}, {{"auto_broadcast", "numpy"}});

    auto slice_Slice_971 = GenSlice(view_Reshape, ndims, int32_max, 1, 3);
    auto cat_Concat_1211 =
        makePattern<opset1::Concat>({rotary_emb, slice_Slice_971 | varsplit_view_Reshape->output(1)}, {{"axis", -1}});
    auto permute_Transpose_1213 = makePattern<opset1::Transpose>({cat_Concat_1211, {0, 2, 1, 3}});

    auto result = permute_Transpose_1213;

    matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto root = m.get_match_root();
        PatternValidator validator(m);
        if (!validator) {
            return false;
        }

        op::internal::RoPE::Config config;
        OutputVector new_args;
        config.rotary_ndims = static_cast<size_t>(validator["ndims"]);

        config.is_interleaved = true;

        // input is [B,L,H,S]
        new_args.push_back(pattern_map.at(view_Reshape));
        // sin_cos table (gathered with positions) [1, L, 64]
        new_args.push_back(pattern_map.at(gather_sin_cos));
        new_args.push_back(pattern_map.at(gather_sin_cos));

        auto old_node = root;

        auto new_node = std::make_shared<op::internal::RoPE>(new_args, config);
        new_node->set_friendly_name(old_node->get_friendly_name());
        ov::copy_runtime_info({pattern_map.at(varsplit).get_node_shared_ptr(),
                               pattern_map.at(repeat_interleave_sin).get_node_shared_ptr(),
                               pattern_map.at(repeat_interleave_cos).get_node_shared_ptr(),
                               pattern_map.at(neg_Multiply_1177).get_node_shared_ptr(),
                               pattern_map.at(Unsqueeze_65524).get_node_shared_ptr(),
                               pattern_map.at(Unsqueeze_65525).get_node_shared_ptr(),
                               pattern_map.at(stack_1182).get_node_shared_ptr(),
                               pattern_map.at(mul_cos).get_node_shared_ptr(),
                               pattern_map.at(mul_sin).get_node_shared_ptr(),
                               pattern_map.at(rotary_emb).get_node_shared_ptr(),
                               pattern_map.at(cat_Concat_1211).get_node_shared_ptr(),
                               pattern_map.at(permute_Transpose_1213).get_node_shared_ptr()},
                              new_node);
        ov::replace_node(old_node, new_node);
        // shapeof may be moved up from transpose to add,
        // After RoPE fusion, shapeof must be moved to the data input of RoPE otherwise extra subgraph exists
        std::shared_ptr<ov::Node> rotary_emb_node = pattern_map.at(rotary_emb).get_node_shared_ptr();
        auto rotary_emb_out = rotary_emb_node->output(0);
        if (rotary_emb_out.get_target_inputs().size() == 2) {
            for (auto& input : rotary_emb_out.get_target_inputs()) {
                if (ov::is_type<opset1::ShapeOf>(input.get_node())) {
                    input.replace_source_output(pattern_map.at(view_Reshape));
                }
            }
        }
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(result, matcher_name);
    this->register_matcher(m, callback);
}

ov::pass::RoPEFusionChatGLM::RoPEFusionChatGLM(int split_output_id, const bool support_2d_rope) {
    MATCHER_SCOPE(RoPEFusionChatGLM);

    //  [seq_length, batch_size, input_size(will be cropped to match hidden state size)]
    //  [batch_size, seq_length, input_size] support_2d_rope
    auto qkv_linear = makePattern("[?,?,?]");
    auto seq_length = makePattern("i32[1]");
    // [max_pos_embeddings, batch_size, half_rotary_dims, 2]
    // [batch_size, max_pos_embeddings, half_rotary_dims, 2] support_2d_rope
    auto cos_sin_cache = makePattern("[?,?,?,?]");

    auto ndims = ov::gen_pattern::Symbol("ndims");
    auto head_cnt = ov::gen_pattern::Symbol("head_cnt");
    auto head_size = ov::gen_pattern::Symbol("head_size");
    auto total_size_q = ov::gen_pattern::Symbol("total_size_q");
    auto total_size_k = ov::gen_pattern::Symbol("total_size_k");
    auto total_size_v = ov::gen_pattern::Symbol("total_size_v");
    auto batch = ov::gen_pattern::Symbol("batch");
    auto seq_len = ov::gen_pattern::Symbol("seq_len");

    auto qkv_proj = makePattern<opset1::VariadicSplit>({qkv_linear, -1, {total_size_q, total_size_k, total_size_v}});
    qkv_proj->set_output_size(3);

    auto cur_key = makePattern<opset1::Reshape>({qkv_proj->output(split_output_id), {0, 0, head_cnt, head_size}},
                                                {{"special_zero", true}});

    std::shared_ptr<ov::Node> input_key = nullptr;
    // Extended the RoPE to a two-dimensional form to accommodate the 2D positional encoding in GLM.
    // Calculate positional embedding independent of batch and each head
    if (support_2d_rope) {
        // Get transposed key [batch, head_cnt, seq_length, head_size]
        input_key = makePattern<opset1::Transpose>({cur_key, {0, 2, 1, 3}});
    } else {
        // Get key [seq_length, batch, head_cnt, head_size]
        input_key = std::move(cur_key);
    }

    auto slice_Slice_437 = GenSlice(input_key, 0, ndims, 1, 3);
    auto var_split_1 = makePattern<opset1::VariadicSplit>({input_key, 3, {ndims, ov::gen_pattern::Symbol("end")}});
    var_split_1->set_output_size(2);

    // rotate half
    std::shared_ptr<ov::Node> reshape_Reshape_453 = nullptr;
    if (support_2d_rope) {
        auto const_target_shape_1 = makeConst({0, head_cnt, 0, ndims / 2, 2});
        reshape_Reshape_453 =
            makePattern<opset1::Reshape>({slice_Slice_437 | var_split_1->output(0), const_target_shape_1},
                                         {{"special_zero", true}});
    } else {
        auto ListConstruct_452_Concat =
            makePattern<opset1::Concat>({seq_length, {-1}, {head_cnt}, {ndims / 2}, {2}}, {{"axis", 0}});
        auto const_target_shape_1 = makeConst({seq_len, batch, head_cnt, ndims / 2, 2});
        reshape_Reshape_453 = makePattern<opset1::Reshape>(
            {slice_Slice_437 | var_split_1->output(0), ListConstruct_452_Concat | const_target_shape_1});
    }

    auto x_even = makePattern<opset8::Gather>({reshape_Reshape_453, 0, -1}, {{"batch_dims", 0}});
    auto x_odd = makePattern<opset8::Gather>({reshape_Reshape_453, 1, -1}, {{"batch_dims", 0}});

    auto var_split_2 = makePattern<opset1::VariadicSplit>({cos_sin_cache, 0, {0, ov::gen_pattern::Symbol("end")}});
    var_split_2->set_output_size(2);

    std::shared_ptr<ov::Node> view_Reshape_460 = nullptr;
    if (support_2d_rope) {
        auto ListConstruct_379_Concat =
            makePattern<opset1::Concat>({{-1}, {1}, seq_length, {ndims / 2}, {2}}, {{"axis", 0}});
        auto const_target_shape_2 = makeConst({batch, 1, seq_len, ndims / 2, 2});

        // Slice cos_sin_cache to support 2-dimentional RoPE
        auto ScatterUpdate = makePattern<opset3::ScatterUpdate>({{0, 0}, {1}, seq_length, {0}}, {});
        auto slice_Slice_449_1d = makePattern<ov::opset8::Slice>({cos_sin_cache, {0}, seq_length, {1}, {1}});
        auto slice_Slice_449_2d = makePattern<ov::opset8::Slice>({cos_sin_cache, {0, 0}, ScatterUpdate, {1, 1}, {0}});
        auto slice_StridedSlice_449 = GenStridedSlice(cos_sin_cache, {0, 0}, ScatterUpdate, {1, 1}, 1);

        // [batch, 1, seq_length, half_rotary_dims, 2]
        view_Reshape_460 = makePattern<opset1::Reshape>(
            {slice_StridedSlice_449 | slice_Slice_449_1d | slice_Slice_449_2d | var_split_2->output(0),
             ListConstruct_379_Concat | const_target_shape_2},
            {{"special_zero", false}});
    } else {
        auto ListConstruct_379_Concat =
            makePattern<opset1::Concat>({seq_length, {-1}, {1}, {ndims / 2}, {2}}, {{"axis", 0}});
        auto const_target_shape_2 = makeConst({seq_len, batch, 1, ndims / 2, 2});

        auto slice_Slice_449 = makePattern<ov::opset8::Slice>({cos_sin_cache, {0}, seq_length, {1}, {0}});
        auto slice_StridedSlice_449 = GenStridedSlice(cos_sin_cache, {0}, seq_length, {1}, 0);

        // [seq_length, 1, batch, half_rotary_dims, 2]
        view_Reshape_460 =
            makePattern<opset1::Reshape>({slice_StridedSlice_449 | slice_Slice_449 | var_split_2->output(0),
                                          ListConstruct_379_Concat | const_target_shape_2},
                                         {{"special_zero", false}});
    }

    auto cos_tab = makePattern<opset8::Gather>({view_Reshape_460, 0, -1}, {{"batch_dims", 0}});
    auto x_even_cos = makePattern<opset1::Multiply>({x_even, cos_tab}, {{"auto_broadcast", "numpy"}});

    auto sin_tab = makePattern<opset8::Gather>({view_Reshape_460, 1, -1}, {{"batch_dims", 0}});
    auto x_odd_sin = makePattern<opset1::Multiply>({x_odd, sin_tab}, {{"auto_broadcast", "numpy"}});
    auto neg_x_odd_sin = makePattern<opset1::Multiply>({x_odd_sin, -1.000000f}, {{"auto_broadcast", "numpy"}});
    auto sub_Subtract_469 = makePattern<opset1::Add>({x_even_cos, neg_x_odd_sin}, {{"auto_broadcast", "numpy"}});

    auto y_even = makePattern<opset1::Unsqueeze>({sub_Subtract_469, -1});
    auto x_odd_cos = makePattern<opset1::Multiply>({x_odd, cos_tab}, {{"auto_broadcast", "numpy"}});
    auto x_even_sin = makePattern<opset1::Multiply>({x_even, sin_tab}, {{"auto_broadcast", "numpy"}});
    auto add_Add_476 = makePattern<opset1::Add>({x_odd_cos, x_even_sin}, {{"auto_broadcast", "numpy"}});
    auto y_odd = makePattern<opset1::Unsqueeze>({add_Add_476, -1});

    auto stack_481 = makePattern<opset1::Concat>({y_even, y_odd}, {{"axis", -1}});

    auto ShapeOf_135133 = makePattern<opset1::ShapeOf>({stack_481});
    auto flatten_Slice_497 = GenSlice(ShapeOf_135133, 0, 3, 1, 0);
    auto flatten_Concat_500 = makePattern<opset1::Concat>({flatten_Slice_497, {-1}}, {{"axis", 0}});

    std::shared_ptr<ov::Node> const_target_shape_3 = nullptr;
    std::shared_ptr<ov::Node> flatten_Reshape_501 = nullptr;
    if (support_2d_rope) {
        // [batch, head_cnt, length, half_rotary_dims, 2]
        const_target_shape_3 = makeConst({batch, head_cnt, seq_len, ndims});
        flatten_Reshape_501 = makePattern<opset1::Reshape>({stack_481, flatten_Concat_500 | const_target_shape_3},
                                                           {{"special_zero", true}});
    } else {
        // [length, batch, head_cnt, half_rotary_dims, 2]
        const_target_shape_3 = makeConst({seq_len, batch, head_cnt, ndims});
        flatten_Reshape_501 = makePattern<opset1::Reshape>({stack_481, flatten_Concat_500 | const_target_shape_3},
                                                           {{"special_zero", true}});
    }
    auto slice_Slice_443 = GenSlice(input_key, ndims, INT_MAX, 1, 3);

    auto cat_Concat_505 =
        makePattern<opset1::Concat>({flatten_Reshape_501, slice_Slice_443 | var_split_1->output(1)}, {{"axis", -1}});

    auto result = cat_Concat_505;

    matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto root = m.get_match_root();
        PatternValidator validator(m);
        if (!validator) {
            return false;
        }

        op::internal::RoPE::Config config;
        OutputVector new_args;
        config.rotary_ndims = static_cast<size_t>(validator["ndims"]);
        config.is_chatglm = true;
        config.support_2d_rope = support_2d_rope;
        config.head_cnt = static_cast<size_t>(validator["head_cnt"]);
        config.head_size = static_cast<size_t>(validator["head_size"]);

        if (split_output_id == 0) {
            // query : split_output_id == 0
            config.slice_start = 0;
            config.slice_stop = static_cast<size_t>(validator["total_size_q"]);
        } else {
            // key : split_output_id == 1
            config.slice_start = static_cast<size_t>(validator["total_size_q"]);
            config.slice_stop = static_cast<size_t>(config.slice_start + validator["total_size_k"]);
        }

        new_args.push_back(pattern_map.at(qkv_linear));
        new_args.push_back(pattern_map.at(cos_sin_cache));
        new_args.push_back(pattern_map.at(cos_sin_cache));

        auto old_node = root;

        auto new_node = std::make_shared<op::internal::RoPE>(new_args, config);
        new_node->set_friendly_name(old_node->get_friendly_name());
        ov::copy_runtime_info({pattern_map.at(flatten_Reshape_501).get_node_shared_ptr(),
                               pattern_map.at(cat_Concat_505).get_node_shared_ptr()},
                              new_node);
        ov::replace_node(old_node, new_node);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(result, matcher_name);
    this->register_matcher(m, callback);
}

ov::pass::RoPEFusionQwen::RoPEFusionQwen(int split_output_id) {
    MATCHER_SCOPE(RoPEFusionQwen);

    // rotary_emb_cos & rotary_emb_sin are sliced by present kv-length (past-kv-length + cur_len)
    auto rotary_emb_cos = makePattern("[1,?,1,?]");  // [1,..4096,1,128]
    auto rotary_emb_sin = makePattern("[1,?,1,?]");  // [1,..4096,1,128]
    auto qkv_proj = makePattern("[?,?,?]");          // [?,?,12288]

    auto head_cnt = ov::gen_pattern::Symbol("head_cnt");
    auto head_size = ov::gen_pattern::Symbol("head_size");

    auto ListUnpack_410_VariadicSplit =
        makePattern<opset1::VariadicSplit>({qkv_proj, 2, {head_cnt * head_size, head_cnt * head_size, -1}});
    ListUnpack_410_VariadicSplit->set_output_size(3);
    // B,L,H,S
    auto view_Reshape_424 = makePattern<opset1::Reshape>(
        {ListUnpack_410_VariadicSplit->output(split_output_id), {0, 0, head_cnt, head_size}},
        {{"special_zero", true}});
    auto slice_Slice_543 = GenSlice(view_Reshape_424, 0, head_size, 1, 3);  //  tensor_array<f32[?,?,32,128]>

    auto hidden_states = makePattern();  //
    auto ShapeOf_485735 = makePattern<opset1::ShapeOf>({hidden_states}, {});
    auto Multiply_567524 = makePattern<opset1::Multiply>({ShapeOf_485735, {-1}}, {{"auto_broadcast", "numpy"}});
    auto Gather_377635 = makePattern<opset8::Gather>({Multiply_567524, {1}, 0}, {{"batch_dims", 0}});

    auto input_ids = makePattern();  // [batch, length]
    auto ShapeOf_409241 = makePattern<ov::op::util::ShapeOfBase>({input_ids}, {});
    auto Gather_311651 = makePattern<opset8::Gather>({ShapeOf_409241, {1}, 0}, {{"batch_dims", 0}});
    auto neg_Multiply = makePattern<opset1::Multiply>({Gather_311651, {-1}}, {{"auto_broadcast", "numpy"}});

    auto ScatterUpdate_463814 = makePattern<opset3::ScatterUpdate>({{0, 0}, {1}, Gather_377635 | neg_Multiply, {0}});
    auto slice_Slice_446 =
        makePattern<ov::opset8::Slice>({rotary_emb_cos, Gather_377635 | neg_Multiply, {INT_MAX}, {1}, {1}});
    auto slice_StridedSlice_446 = GenStridedSlice(rotary_emb_cos,
                                                  ScatterUpdate_463814,
                                                  {0, INT_MAX},
                                                  {1, 1},
                                                  1);  //  tensor_array<f32[1,..4096,1,128]>
    auto mul_Multiply_552 =
        makePattern<opset1::Multiply>({slice_Slice_543, slice_StridedSlice_446 | slice_Slice_446},
                                      {{"auto_broadcast", "numpy"}});  //  tensor_array<f32[?,?,32,128]>

    auto reshape_opt1 = [&](std::shared_ptr<Node> input_BLHS) {
        auto ShapeOf_485814 = makePattern<opset1::ShapeOf>({input_BLHS}, {});
        auto Gather_377647 = makePattern<opset8::Gather>({ShapeOf_485814, {1}, 0}, {{"batch_dims", 0}});
        // batch-size, we don't care
        auto Gather_377641 = makePattern("i32[1]");
        auto ListConstruct_581_Concat =
            makePattern<opset1::Concat>({Gather_377641, Gather_377647, {head_cnt}, {2}, {head_size / 2}},
                                        {{"axis", 0}});
        auto Gather_391791 = makePattern<opset8::Gather>({ShapeOf_485814, {0, 1}, 0}, {{"batch_dims", 0}});
        auto ListConstruct_522_Concat = makePattern<opset1::Concat>({Gather_391791, {32}, {2}, {64}}, {{"axis", 0}});

        auto reshape_Reshape_577 =
            makePattern<opset1::Reshape>({input_BLHS, {-1, 2, head_size / 2}}, {{"special_zero", true}});
        return makePattern<opset1::Reshape>({reshape_Reshape_577, ListConstruct_581_Concat | ListConstruct_522_Concat},
                                            {{"special_zero", false}});  //  tensor_array<f32[?,?,32,2,64]>
    };

    // If with sepcial_zero, const_shape should be checked later
    auto const_shape = makePattern<opset1::Constant>({}, {});
    auto reshape_special = makePattern<opset1::Reshape>({slice_Slice_543, const_shape}, {{"special_zero", true}});

    auto ListUnpack_586_Split =
        makePattern<opset1::Split>({reshape_opt1(slice_Slice_543) | reshape_special, -2},
                                   {{"num_splits", 2}});  //  tensor_array<f32[?,?,32,1,64] f32[?,?,32,1,64]>
    ListUnpack_586_Split->set_output_size(2);
    auto Multiply_567527 =
        makePattern<opset1::Multiply>({ListUnpack_586_Split->output(1), -1.000000f},
                                      {{"auto_broadcast", "numpy"}});  //  tensor_array<f32[?,?,32,1,64]>
    auto ListUnpack_586_Squeeze_0 =
        makePattern<opset1::Squeeze>({Multiply_567527, -2});  //  tensor_array<f32[?,?,32,64]>
    auto ListUnpack_586_Squeeze =
        makePattern<opset1::Squeeze>({ListUnpack_586_Split->output(0), -2});  //  tensor_array<f32[?,?,32,64]>
    auto cat_Concat_593 = makePattern<opset1::Concat>({ListUnpack_586_Squeeze_0, ListUnpack_586_Squeeze},
                                                      {{"axis", -1}});  //  tensor_array<f32[?,?,32,128]>
    auto slice_StridedSlice_470 = GenStridedSlice(rotary_emb_sin,
                                                  ScatterUpdate_463814,
                                                  {0, INT_MAX},
                                                  {1, 1},
                                                  1);  //  tensor_array<f32[1,..4096,1,128]>
    auto slice_Slice_470 =
        makePattern<opset8::Slice>({rotary_emb_sin, Gather_377635 | neg_Multiply, {INT_MAX}, {1}, {1}});
    auto mul_Multiply_594 =
        makePattern<opset1::Multiply>({cat_Concat_593, slice_StridedSlice_470 | slice_Slice_470},
                                      {{"auto_broadcast", "numpy"}});  //  tensor_array<f32[?,?,32,128]>
    auto add_Add_597 = makePattern<opset1::Add>({mul_Multiply_552, mul_Multiply_594},
                                                {{"auto_broadcast", "numpy"}});  //  tensor_array<f32[?,?,32,128]>

    auto result = add_Add_597;
    matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto root = m.get_match_root();
        PatternValidator validator(m);
        if (!validator) {
            return false;
        }

        op::internal::RoPE::Config config;
        OutputVector new_args;
        config.is_qwen = true;
        config.head_cnt = static_cast<size_t>(validator["head_cnt"]);
        config.head_size = static_cast<size_t>(validator["head_size"]);
        config.rotary_ndims = config.head_size;

        if (split_output_id == 0) {
            // query : split_output_id == 0
            config.slice_start = 0;
            config.slice_stop = config.head_cnt * config.head_size;
        } else {
            // key : split_output_id == 1
            config.slice_start = config.head_cnt * config.head_size;
            config.slice_stop = config.slice_start + config.head_cnt * config.head_size;
        }

        if (pattern_map.count(reshape_special)) {
            // check reshape_special shape correctness
            auto reshape_special_node = pattern_map.at(reshape_special).get_node_shared_ptr();
            auto data_shape = reshape_special_node->get_input_partial_shape(0);
            auto reshape_shape = pattern_map.at(const_shape);
            auto node = ov::as_type_ptr<opset1::Constant>(reshape_shape.get_node_shared_ptr());
            const auto& target = node->cast_vector<int32_t>();
            // ensure target_shape have correct rank
            if (target.size() < 3) {
                return false;
            }
            int32_t head_size = static_cast<int32_t>(config.head_size);
            int32_t head_cnt = static_cast<int32_t>(config.head_cnt);
            // reshape splits the head_size of input to [2, head_size / 2]
            // head_cnt of target_shape could be 0 or head_cnt
            size_t target_rank = target.size();
            bool is_ok = (target[target_rank - 1] == head_size / 2) && (target[target_rank - 2] == 2) &&
                         ((target[target_rank - 3] == 0 || target[target_rank - 3] == head_cnt));
            if (!is_ok) {
                return false;
            }
        }

        new_args.push_back(pattern_map.at(qkv_proj));
        new_args.push_back(pattern_map.at(rotary_emb_cos));
        new_args.push_back(pattern_map.at(rotary_emb_sin));

        auto old_node = root;
        auto new_node = std::make_shared<op::internal::RoPE>(new_args, config);
        new_node->set_friendly_name(old_node->get_friendly_name());
        ov::copy_runtime_info({pattern_map.at(Multiply_567527).get_node_shared_ptr(),
                               pattern_map.at(ListUnpack_586_Squeeze_0).get_node_shared_ptr(),
                               pattern_map.at(ListUnpack_586_Squeeze).get_node_shared_ptr(),
                               pattern_map.at(cat_Concat_593).get_node_shared_ptr(),
                               pattern_map.at(mul_Multiply_594).get_node_shared_ptr(),
                               pattern_map.at(add_Add_597).get_node_shared_ptr()},
                              new_node);
        ov::replace_node(old_node, new_node);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(result, matcher_name);
    this->register_matcher(m, callback);
}

/*
 in Llama RoPE, cos/sin tables can be shared among all layers but it didn't in the orginal model
 here we try to share the preparation subgraphs results of these tables across layers.
 This is not a generic solution due to difficulty of the algorithm
*/
ov::pass::RoPEShareCosSin::RoPEShareCosSin() {
    MATCHER_SCOPE(RoPEShareCosSin);

    std::vector<std::shared_ptr<Node>> inputs = {makePattern(), makePattern()};
    auto const_inv_freq = makePattern<opset1::Constant>({}, {});

    auto Constant_58774 = makeConst(element::u8, ov::Shape({}), {0});
    auto Broadcast_58775 = makePattern<opset1::Broadcast>({{1.000000f}, inputs[0], Constant_58774},
                                                          {{"mode", "numpy"}});  //  tensor_array<f32[?,?,?]>
    auto expand_Broadcast =
        makePattern<opset1::Multiply>({const_inv_freq, Broadcast_58775},
                                      {{"auto_broadcast", "numpy"}});  //  tensor_array<f32[?,128,?]>
    auto matmul_MatMul =
        makePattern<opset1::MatMul>({expand_Broadcast, inputs[1]}, {{"transpose_a", false}, {"transpose_b", false}});
    auto transpose_Transpose = makePattern<opset1::Transpose>({matmul_MatMul, {0, 2, 1}});
    auto cat_Concat = makePattern<opset1::Concat>({transpose_Transpose, transpose_Transpose}, {{"axis", -1}});
    auto cos_Cos = makePattern<opset1::Cos>({cat_Concat});
    auto sin_Sin = makePattern<opset1::Sin>({cat_Concat});
    auto result = makePattern<opset1::Unsqueeze>({cos_Cos | sin_Sin, 1});

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto root = m.get_match_root();
        PatternValidator validator(m);
        if (!validator) {
            return false;
        }

        auto it = pattern_map.find(const_inv_freq);
        if (it == pattern_map.end()) {
            return false;
        }
        auto cur_inv_freq = ov::as_type_ptr<opset1::Constant>(it->second.get_node_shared_ptr());
        if (!cur_inv_freq) {
            return false;
        }

        // the first match is the one to be shared, collect all inputs
        // and constants into the state capture by lambda
        if (!m_inv_freq) {
            for (size_t i = 0; i < m_shared_inputs.size(); i++) {
                auto it = pattern_map.find(inputs[i]);
                if (it == pattern_map.end())
                    return false;
                auto input_node = it->second.get_node_shared_ptr();
                m_shared_inputs[i] = input_node;
            }
            m_inv_freq = cur_inv_freq;
        }

        // check consts are the same as the one to be shared.
        if (cur_inv_freq->get_element_type() != m_inv_freq->get_element_type())
            return false;
        if (cur_inv_freq->get_shape() != m_inv_freq->get_shape())
            return false;
        auto global_inv_freq = ov::as_type_ptr<opset1::Constant>(m_inv_freq);

        auto cmp_error =
            memcmp(cur_inv_freq->get_data_ptr(), global_inv_freq->get_data_ptr(), global_inv_freq->get_byte_size());
        if (cmp_error != 0)
            return false;
        // check all inputs are the same as the one to be shared.
        for (size_t i = 0; i < inputs.size(); i++) {
            auto it = pattern_map.find(inputs[i]);
            if (it == pattern_map.end())
                return false;
            auto input_node = it->second.get_node_shared_ptr();
            if (m_shared_inputs[i] != input_node)
                return false;
        }

        // now the match share the same topology & inputs(consts) upto the sin/cos node
        // we can intialize the unsqueezed sin/cos to be shared
        bool is_sin_matched = pattern_map.find(sin_Sin) != pattern_map.end();
        if (is_sin_matched && !m_shared_sin0) {
            m_shared_sin0 = root;
            return false;
        }
        if (!is_sin_matched && !m_shared_cos0) {
            m_shared_cos0 = root;
            return false;
        }

        // all inputs & consts are same, we can safely shared the subgraph
        // Just for record, the pattern uses cos | sin as root node. This means that we could match both cases.
        // There we use find to decides whether cons or sin is used
        auto replacement = m_shared_cos0;
        if (pattern_map.find(sin_Sin) != pattern_map.end()) {
            replacement = m_shared_sin0;
        }
        ov::replace_node(root, replacement);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(result, matcher_name);
    this->register_matcher(m, callback);
}
