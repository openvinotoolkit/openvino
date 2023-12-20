// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rope_fusion.hpp"

#include <cstdint>
#include <limits>
#include <openvino/core/rt_info.hpp>
#include "openvino/opsets/opset1.hpp"
#include <openvino/opsets/opset6.hpp>
#include <openvino/opsets/opset8.hpp>
#include <openvino/pass/pattern/op/or.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>
#include <transformations/utils/utils.hpp>

#include "itt.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "transformations/cpu_opset/common/op/rope.hpp"
#include "utils/gen_pattern.hpp"

using namespace ov::gen_pattern;

ov::intel_cpu::RoPEFusionGPTNEOX::RoPEFusionGPTNEOX() {
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

    x->set_friendly_name("x");

    auto half_ndims = Symbol("half_ndims");
    auto int32_max = std::numeric_limits<std::int32_t>::max();

    // rotate half : [-x2, x1]
    auto x2 = GenSlice(x, half_ndims, int32_max, 1, 3);
    auto x2neg = makePattern<opset1::Multiply>({x2, -1.0f}, {{"auto_broadcast", "numpy"}});
    auto x1 = GenSlice(x, 0, half_ndims, 1, 3);
    auto x_rotate_half = makePattern<opset1::Concat>({x2neg, x1}, {{"axis", -1}});

    auto mul_cos = makePattern<opset1::Multiply>({x_or_cos1, x_or_cos2}, {{"auto_broadcast", "numpy"}});
    auto mul_sin = makePattern<opset1::Multiply>({x_rotate_half, t_sin}, {{"auto_broadcast", "numpy"}});

    // [x1, x2]*cos + [-x2, x1]*sin
    auto result = makePattern<opset1::Add>({mul_cos, mul_sin}, {{"auto_broadcast", "numpy"}});

    matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
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

        RoPENode::Config config;
        OutputVector new_args;
        config.rotary_ndims = 2 * validator["half_ndims"];

        new_args.push_back(pattern_map.at(x));
        new_args.push_back(v_cos);
        new_args.push_back(pattern_map.at(t_sin));

        auto old_node = root;
        auto new_node = std::make_shared<RoPENode>(new_args, config);
        new_node->set_friendly_name(old_node->get_friendly_name());
        ov::replace_node(old_node, new_node);

        // this new node may match following additional matchers
        register_new_node(new_node);

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(result, matcher_name);
    this->register_matcher(m, callback);
}

ov::intel_cpu::RoPEFusionCosSinPreprocess::RoPEFusionCosSinPreprocess() {
    MATCHER_SCOPE(RoPEFusionCosSinPreprocess);

    auto cos_const = makePattern<opset1::Constant>({});  // "f32[1,1,2048,24]"
    auto sin_const = makePattern<opset1::Constant>({});  // "f32[1,1,2048,24]"

    auto node_batch_size = makePattern("i32[1]");
    auto tile_batch = makePattern("i32[1]");
    auto gather_positions = makePattern("i32[?,?,?,?]");

    auto prepare_cos_sin_gptneox = [&](std::shared_ptr<Node> const_tab) {
        auto slice1 = makePattern<opset1::StridedSlice>({const_tab, {0}, node_batch_size, {1}},
                                                        {{"begin_mask", {0}},
                                                         {"end_mask", {0}},
                                                         {"new_axis_mask", {}},
                                                         {"shrink_axis_mask", {}},
                                                         {"ellipsis_mask", {}}});
        return makePattern<opset6::GatherElements>({slice1, gather_positions}, {{"axis", 2}});
    };

    auto seq_len = makePattern("i32[1]");
    auto gather_positions_2d = makePattern("i32[?,?]");

    auto head_dims = Symbol("head_dims");
    auto prepare_cos_sin_llama = [&](std::shared_ptr<Node> const_tab) {
        auto ScatterUpdate = makePattern<opset3::ScatterUpdate>({{0, 0, 0}, 2, seq_len, 0});
        auto slice_Slice = makePattern<opset1::StridedSlice>({const_tab, {0, 0, 0}, ScatterUpdate, {1, 1, 1}},
                                                             {{"begin_mask", {1, 1, 0}},
                                                              {"end_mask", {1, 1, 0}},
                                                              {"new_axis_mask", {}},
                                                              {"shrink_axis_mask", {}},
                                                              {"ellipsis_mask", {}}});
        auto squeeze = makePattern<opset1::Reshape>({slice_Slice, {-1, head_dims}});
        auto index_Gather = makePattern<opset8::Gather>({squeeze, gather_positions_2d, 0}, {{"batch_dims", 0}});

        // another simplified pattern for gathering at position_ids
        auto slice_Slice2 = makePattern<opset1::StridedSlice>({const_tab, {0}, seq_len, {1}},
                                                              {{"begin_mask", {0}},
                                                               {"end_mask", {0}},
                                                               {"new_axis_mask", {}},
                                                               {"shrink_axis_mask", {}},
                                                               {"ellipsis_mask", {}}});
        auto index_Gather2 = makePattern<opset8::Gather>({slice_Slice2, gather_positions_2d, 0}, {{"batch_dims", 0}});

        auto unsqueeze = makePattern<opset1::Reshape>({index_Gather | index_Gather2, {1, 1, -1, head_dims}});
        auto unsqueeze2 = makePattern<opset1::Unsqueeze>({index_Gather2, 1});

        return unsqueeze2 | unsqueeze;
    };

    auto cos_tab = prepare_cos_sin_gptneox(cos_const) | prepare_cos_sin_llama(cos_const);
    auto sin_tab = prepare_cos_sin_gptneox(sin_const) | prepare_cos_sin_llama(sin_const);

    auto x = makePattern(ov::Rank(4));
    auto rope = makePattern<RoPENode>({x, cos_tab, sin_tab});

    matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        PatternValidator validator(m);
        if (!validator) {
            return false;
        }
        const auto& pattern_map = m.get_pattern_value_map();
        auto root = m.get_match_root();
        auto rope_node = as_type_ptr<RoPENode>(pattern_map.at(rope).get_node_shared_ptr());
        if (!rope_node)
            return false;

        if (pattern_map.count(cos_const)) {
            rope_node->set_argument(1, pattern_map.at(cos_const));
        }
        if (pattern_map.count(sin_const)) {
            rope_node->set_argument(2, pattern_map.at(sin_const));
        }

        auto& config = rope_node->get_config();
        if (pattern_map.count(gather_positions)) {
            auto arg_id = rope_node->get_input_size();
            rope_node->set_argument(arg_id, pattern_map.at(gather_positions));
            config.gather_position_arg_id = arg_id;
        } else if (pattern_map.count(gather_positions_2d)) {
            auto arg_id = rope_node->get_input_size();
            rope_node->set_argument(arg_id, pattern_map.at(gather_positions_2d));
            config.gather_position_arg_id = arg_id;
        }
        rope_node->validate_and_infer_types();
        register_new_node(rope_node);
        return true;
    };
    auto m = std::make_shared<ngraph::pattern::Matcher>(rope, matcher_name);
    this->register_matcher(m, callback);
}

// only a fraction of head_size is rotary-embedded
ov::intel_cpu::RoPEFusionIOSlicing::RoPEFusionIOSlicing() {
    MATCHER_SCOPE(RoPEFusionIOSlicing);
    auto int32_max = std::numeric_limits<std::int32_t>::max();
    auto data = makePattern(ov::Rank(4));

    auto ndims = Symbol("ndims");
    auto x = GenSlice(data, 0, ndims, 1, 3);
    auto y = GenSlice(data, ndims, int32_max, 1, 3);
    auto x_emb = makePattern<RoPENode>({x, {}, {}}) | makePattern<RoPENode>({x, {}, {}, {}});
    auto result = makePattern<opset1::Concat>({x_emb, y}, {{"axis", -1}});

    matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto root = m.get_match_root();

        auto rope_node = as_type_ptr<RoPENode>(root->input_value(0).get_node_shared_ptr());
        if (!rope_node)
            return false;

        PatternValidator validator(m);
        if (!validator) {
            return false;
        }
        auto ndims = validator["ndims"];

        auto& config = rope_node->get_config();
        if (config.rotary_ndims != ndims)
            return false;

        // remove slice & concat
        rope_node->set_argument(0, pattern_map.at(data));
        rope_node->set_friendly_name(root->get_friendly_name());
        ov::replace_node(root, rope_node);

        rope_node->validate_and_infer_types();
        register_new_node(rope_node);
        return true;
    };
    auto m = std::make_shared<ngraph::pattern::Matcher>(result, matcher_name);
    this->register_matcher(m, callback);
}

ov::intel_cpu::RoPEFusionPreprocess::RoPEFusionPreprocess() {
    MATCHER_SCOPE(RoPEFusionPreprocess);

    // gptneox-preprocess of input data
    auto input_to_slice = makePattern(ov::Rank(4));
    auto input_to_trans = makePattern(ov::Rank(4));  // no need to slice from 3S

    // in some model qkv prejection is combined and
    // needs to be sliced before RoPE
    auto slice_start = Symbol("slice_start");
    auto slice_stop = Symbol("slice_stop");
    auto input_slice = GenSlice(input_to_slice, slice_start, slice_stop, 1, 3);

    // some model will transpose from [B,L,H,S] to [B,H,L,S] before RoPE
    auto x = makePattern<opset1::Transpose>({input_slice | input_to_trans, {0, 2, 1, 3}});
    auto result = makePattern<RoPENode>({x, {}, {}}) | makePattern<RoPENode>({x, {}, {}, {}});

    matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        PatternValidator validator(m);
        if (!validator) {
            return false;
        }

        const auto& pattern_map = m.get_pattern_value_map();
        auto root = m.get_match_root();
        auto rope_node = as_type_ptr<RoPENode>(root);
        if (!rope_node)
            return false;

        auto& config = rope_node->get_config();

        if (pattern_map.count(input_to_slice)) {
            config.slice_start = validator["slice_start"];
            config.slice_stop = validator["slice_stop"];
            config.input_trans0213 = true;
            rope_node->set_argument(0, pattern_map.at(input_to_slice));
        } else if (pattern_map.count(input_to_trans)) {
            config.input_trans0213 = true;
            rope_node->set_argument(0, pattern_map.at(input_to_trans));
        } else {
            return false;
        }
        rope_node->validate_and_infer_types();
        register_new_node(rope_node);
        return true;
    };
    auto m = std::make_shared<ngraph::pattern::Matcher>(result, matcher_name);
    this->register_matcher(m, callback);
}

// remove stridedslice from 0 to int32_max with stride 1
ov::intel_cpu::EliminateStridedSlice::EliminateStridedSlice() {
    MATCHER_SCOPE(EliminateStridedSlice);
    auto data = ov::pass::pattern::any_input(ngraph::pattern::has_static_rank());
    auto begin = ov::pass::pattern::wrap_type<opset1::Constant>(ngraph::pattern::type_matches(ov::element::i32));
    auto end = ov::pass::pattern::wrap_type<opset1::Constant>(ngraph::pattern::type_matches(ov::element::i32));
    auto stride = ov::pass::pattern::wrap_type<opset1::Constant>(ngraph::pattern::type_matches(ov::element::i32));

    auto strided_slice =
        ov::pass::pattern::wrap_type<opset1::StridedSlice>({data, begin, end, stride}, [](const Output<Node>& value) {
            auto s1 = as_type_ptr<opset1::StridedSlice>(value.get_node_shared_ptr());
            if (!s1->get_new_axis_mask().empty() || !s1->get_shrink_axis_mask().empty() ||
                !s1->get_ellipsis_mask().empty()) {
                return false;
            }

            auto inputs = s1->input_values();

            auto begin = as_type_ptr<opset1::Constant>(inputs[1].get_node_shared_ptr());
            auto end = as_type_ptr<opset1::Constant>(inputs[2].get_node_shared_ptr());
            auto stride = as_type_ptr<opset1::Constant>(inputs[3].get_node_shared_ptr());

            if (!begin)
                return false;
            if (!end)
                return false;
            if (!stride)
                return false;

            // stride is all 1
            auto v_stride = stride->cast_vector<int32_t>();
            for (auto& v : v_stride) {
                if (v != 1)
                    return false;
            }

            auto v_begin = begin->cast_vector<int32_t>();
            auto v_end = end->cast_vector<int32_t>();
            if (v_begin.size() != v_end.size()) {
                return false;
            }

            auto& begin_mask = s1->get_begin_mask();
            auto& end_mask = s1->get_end_mask();
            auto mask_size = begin_mask.size();
            if (begin_mask.size() != end_mask.size()) {
                return false;
            }

            auto int32_max = std::numeric_limits<std::int32_t>::max();
            size_t i = 0;
            for (; i < mask_size; i++) {
                if (begin_mask[i] != end_mask[i])
                    return false;
                // all valid [begin, end] are [0, int32_max]
                if (begin_mask[i] == 0 && end_mask[i] == 0) {
                    if (v_begin[i] != 0 || v_end[i] != int32_max)
                        return false;
                }
            }
            // the non-masked part
            for (; i < v_begin.size(); i++) {
                if (v_begin[i] != 0 || v_end[i] != int32_max)
                    return false;
            }
            return true;
        });

    matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        auto root = m.get_match_root();
        return replace_output_update_name(root->output(0), root->input_value(0));
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(strided_slice, matcher_name);
    this->register_matcher(m, callback);
}

ov::intel_cpu::RoPEFusionGPTJ::RoPEFusionGPTJ() {
    MATCHER_SCOPE(RoPEFusionGPTJ);

    auto int32_max = std::numeric_limits<std::int32_t>::max();
    auto ndims = Symbol("ndims");

    auto view_Reshape = makePattern(ov::Rank(4));

    // view_Reshape : B,L,H,S
    auto slice_Slice_965 = GenSlice(view_Reshape, 0, ndims, 1, 3);

    auto gather_sin_cos = makePattern("f32");

    auto varsplit = makePattern<opset1::VariadicSplit>({gather_sin_cos, -1, {ndims / 2, -1}});
    varsplit->set_output_size(2);
    auto unsqueeze_sin = makePattern<opset1::Reshape>({varsplit->output(0), {1, -1, 1, 32}});
    auto unsqueeze_cos = makePattern<opset1::Reshape>({varsplit->output(1), {1, -1, 1, 32}});
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

    auto t_cos = makePattern(ov::Rank(4));
    auto t_sin = makePattern(ov::Rank(4));

    // x interleave (-x[:,:,:, 1::2], x[:,:,:, 0::2])
    auto slice_Slice_1174 = GenSlice(slice_Slice_965, 1, int32_max, 2, 3);

    auto neg_Multiply_1177 = makePattern<opset1::Multiply>({slice_Slice_1174, -1.0f}, {{"auto_broadcast", "numpy"}});
    auto Unsqueeze_65524 = makePattern<opset1::Unsqueeze>({neg_Multiply_1177, -1});

    auto slice_Slice_1168 = GenSlice(slice_Slice_965, 0, int32_max, 2, 3);
    auto Unsqueeze_65525 = makePattern<opset1::Unsqueeze>({slice_Slice_1168, -1});
    auto stack_1182 = makePattern<opset1::Concat>({Unsqueeze_65524, Unsqueeze_65525}, {{"axis", -1}});

    auto ShapeOf_169068 = makePattern<opset1::ShapeOf>({stack_1182});
    auto flatten_Slice_1194 = GenSlice(ShapeOf_169068, 0, 3, 1, 0);
    auto flatten_Concat_1197 = makePattern<opset1::Concat>({flatten_Slice_1194, {-1}}, {{"axis", 0}});
    auto flatten_Reshape_1198 = makePattern<opset1::Reshape>({stack_1182, flatten_Concat_1197});

    // x*cos [B,L,H,ndims]
    auto mul_cos =
        makePattern<opset1::Multiply>({slice_Slice_965, repeat_interleave_cos}, {{"auto_broadcast", "numpy"}});
    auto mul_sin =
        makePattern<opset1::Multiply>({flatten_Reshape_1198, repeat_interleave_sin}, {{"auto_broadcast", "numpy"}});

    // *cos + *sin
    auto rotary_emb = makePattern<opset1::Add>({mul_cos, mul_sin}, {{"auto_broadcast", "numpy"}});

    auto slice_Slice_971 = GenSlice(view_Reshape, ndims, int32_max, 1, 3);
    auto cat_Concat_1211 = makePattern<opset1::Concat>({rotary_emb, slice_Slice_971}, {{"axis", -1}});
    auto permute_Transpose_1213 = makePattern<opset1::Transpose>({cat_Concat_1211, {0, 2, 1, 3}});

    auto result = permute_Transpose_1213;

    matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto root = m.get_match_root();
        PatternValidator validator(m);
        if (!validator) {
            return false;
        }

        RoPENode::Config config;
        OutputVector new_args;
        config.rotary_ndims = validator["ndims"];

        config.is_interleaved = true;

        // input is [B,L,H,S]
        new_args.push_back(pattern_map.at(view_Reshape));
        // sin_cos table (gathered with positions) [1, L, 64]
        new_args.push_back(pattern_map.at(gather_sin_cos));
        new_args.push_back(pattern_map.at(gather_sin_cos));

        auto old_node = root;

        auto new_node = std::make_shared<RoPENode>(new_args, config);
        new_node->set_friendly_name(old_node->get_friendly_name());
        ov::replace_node(old_node, new_node);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(result, matcher_name);
    this->register_matcher(m, callback);
}

ov::intel_cpu::RoPEFusionChatGLM::RoPEFusionChatGLM(int split_output_id) {
    MATCHER_SCOPE(RoPEFusionChatGLM);

    auto qkv_linear = makePattern("f32[?,?,?]");  //  f32[seq_length, batch_size, 4608]
    auto seq_length = makePattern("i32[1]");
    auto cos_sin_cache = makePattern("f32[?,?,?,?]");  // [max_pos_embeddings, batch_size, 32, 2]

    auto ndims = Symbol("ndims");
    auto head_cnt = Symbol("head_cnt");
    auto head_size = Symbol("head_size");
    auto total_size_q = Symbol("total_size_q");
    auto total_size_k = Symbol("total_size_k");
    auto total_size_v = Symbol("total_size_v");

    auto qkv_proj = makePattern<opset1::VariadicSplit>({qkv_linear, -1, {total_size_q, total_size_k, total_size_v}});
    qkv_proj->set_output_size(3);

    // get key [L, B, Hkv, S]
    auto cur_key = makePattern<opset1::Reshape>({qkv_proj->output(split_output_id), {0, 0, head_cnt, head_size}},
                                                {{"special_zero", true}});

    auto slice_Slice_437 = makePattern<opset1::StridedSlice>({cur_key, {0, 0, 0, 0}, {0, 0, 0, ndims}, {1, 1, 1, 1}},
                                                             {{"begin_mask", {1, 1, 1, 0}},
                                                              {"end_mask", {1, 1, 1, 0}},
                                                              {"new_axis_mask", {}},
                                                              {"shrink_axis_mask", {}},
                                                              {"ellipsis_mask", {}}});

    // rotate half
    auto ListConstruct_452_Concat =
        makePattern<opset1::Concat>({seq_length, {-1}, {head_cnt}, {ndims / 2}, {2}}, {{"axis", 0}});
    auto ListConstruct_379_Concat =
        makePattern<opset1::Concat>({seq_length, {-1}, {1}, {ndims / 2}, {2}}, {{"axis", 0}});

    auto reshape_Reshape_453 =
        makePattern<opset1::Reshape>({slice_Slice_437, ListConstruct_452_Concat}, {{"special_zero", false}});
    auto x_even = makePattern<opset8::Gather>({reshape_Reshape_453, 0, -1}, {{"batch_dims", 0}});
    auto slice_Slice_449 = makePattern<opset1::StridedSlice>({cos_sin_cache, {0}, seq_length, {1}},
                                                             {{"begin_mask", {0}},
                                                              {"end_mask", {0}},
                                                              {"new_axis_mask", {}},
                                                              {"shrink_axis_mask", {}},
                                                              {"ellipsis_mask", {}}});
    auto view_Reshape_460 =
        makePattern<opset1::Reshape>({slice_Slice_449, ListConstruct_379_Concat}, {{"special_zero", false}});
    auto cos_tab = makePattern<opset8::Gather>({view_Reshape_460, 0, -1}, {{"batch_dims", 0}});
    auto x_even_cos = makePattern<opset1::Multiply>({x_even, cos_tab}, {{"auto_broadcast", "numpy"}});
    auto x_odd = makePattern<opset8::Gather>({reshape_Reshape_453, 1, -1}, {{"batch_dims", 0}});
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
    auto flatten_Slice_497 = makePattern<opset1::StridedSlice>({ShapeOf_135133, {0}, {3}, {1}},
                                                               {{"begin_mask", {0}},
                                                                {"end_mask", {0}},
                                                                {"new_axis_mask", {}},
                                                                {"shrink_axis_mask", {}},
                                                                {"ellipsis_mask", {}}});
    auto flatten_Concat_500 = makePattern<opset1::Concat>({flatten_Slice_497, {-1}}, {{"axis", 0}});
    auto const_target_shape = makeConst({0, 0, head_cnt, ndims});
    // [length, batch, head_cnt, half_rotary_dims, 2]
    auto flatten_Reshape_501 =
        makePattern<opset1::Reshape>({stack_481, flatten_Concat_500 | const_target_shape}, {{"special_zero", true}});
    auto slice_Slice_443 =
        makePattern<opset1::StridedSlice>({cur_key, {0, 0, 0, ndims}, {0, 0, 0, INT_MAX}, {1, 1, 1, 1}},
                                          {{"begin_mask", {1, 1, 1, 0}},
                                           {"end_mask", {1, 1, 1, 0}},
                                           {"new_axis_mask", {}},
                                           {"shrink_axis_mask", {}},
                                           {"ellipsis_mask", {}}});
    auto cat_Concat_505 = makePattern<opset1::Concat>({flatten_Reshape_501, slice_Slice_443}, {{"axis", -1}});

    auto result = cat_Concat_505;

    matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto root = m.get_match_root();
        PatternValidator validator(m);
        if (!validator) {
            return false;
        }

        RoPENode::Config config;
        OutputVector new_args;
        config.rotary_ndims = validator["ndims"];
        config.is_chatglm = true;
        config.head_cnt = validator["head_cnt"];
        config.head_size = validator["head_size"];

        if (split_output_id == 0) {
            // query : split_output_id == 0
            config.slice_start = 0;
            config.slice_stop = validator["total_size_q"];
        } else {
            // key : split_output_id == 1
            config.slice_start = validator["total_size_q"];
            config.slice_stop = config.slice_start + validator["total_size_k"];
        }

        new_args.push_back(pattern_map.at(qkv_linear));
        new_args.push_back(pattern_map.at(cos_sin_cache));
        new_args.push_back(pattern_map.at(cos_sin_cache));

        auto old_node = root;

        auto new_node = std::make_shared<RoPENode>(new_args, config);
        new_node->set_friendly_name(old_node->get_friendly_name());
        ov::replace_node(old_node, new_node);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(result, matcher_name);
    this->register_matcher(m, callback);
}

ov::intel_cpu::RoPEFusionQwen::RoPEFusionQwen(int split_output_id) {
    MATCHER_SCOPE(RoPEFusionQwen);

    // rotary_emb_cos & rotary_emb_sin are sliced by present kv-length (past-kv-length + cur_len)
    auto rotary_emb_cos = makePattern("f32[1,?,1,?]");  // [1,..4096,1,128]
    auto rotary_emb_sin = makePattern("f32[1,?,1,?]");  // [1,..4096,1,128]
    auto qkv_proj = makePattern("f32[?,?,?]");          // f32[?,?,12288]

    auto head_cnt = Symbol("head_cnt");
    auto head_size = Symbol("head_size");

    auto ListUnpack_410_VariadicSplit =
        makePattern<opset1::VariadicSplit>({qkv_proj, 2, {head_cnt * head_size, head_cnt * head_size, -1}});
    ListUnpack_410_VariadicSplit->set_output_size(3);
    // B,L,H,S
    auto view_Reshape_424 = makePattern<opset1::Reshape>(
        {ListUnpack_410_VariadicSplit->output(split_output_id), {0, 0, head_cnt, head_size}},
        {{"special_zero", true}});
    auto slice_Slice_543 =
        makePattern<opset1::StridedSlice>({view_Reshape_424, {0, 0, 0, 0}, {0, 0, 0, head_size}, {1, 1, 1, 1}},
                                          {{"begin_mask", {1, 1, 1, 0}},
                                           {"end_mask", {1, 1, 1, 0}},
                                           {"new_axis_mask", {}},
                                           {"shrink_axis_mask", {}},
                                           {"ellipsis_mask", {}}});  //  tensor_array<f32[?,?,32,128]>

    auto hidden_states = makePattern("f32[?,?,?]");  //
    auto ShapeOf_485735 = makePattern<opset1::ShapeOf>({hidden_states}, {});
    auto Multiply_567524 = makePattern<opset1::Multiply>({ShapeOf_485735, {-1}}, {{"auto_broadcast", "numpy"}});
    auto Gather_377635 = makePattern<opset8::Gather>({Multiply_567524, {1}, 0}, {{"batch_dims", 0}});

    auto input_ids = makePattern("i32[?,?]");  // [batch, length]
    auto ShapeOf_409241 = makePattern<opset1::ShapeOf>({input_ids}, {});
    auto Gather_311651 = makePattern<opset8::Gather>({ShapeOf_409241, {1}, 0}, {{"batch_dims", 0}});
    auto neg_Multiply = makePattern<opset1::Multiply>({Gather_311651, {-1}}, {{"auto_broadcast", "numpy"}});

    auto ScatterUpdate_463814 = makePattern<opset3::ScatterUpdate>({{0, 0}, {1}, Gather_377635 | neg_Multiply, {0}});

    auto slice_Slice_446 =
        makePattern<opset1::StridedSlice>({rotary_emb_cos, ScatterUpdate_463814, {0, INT_MAX}, {1, 1}},
                                          {{"begin_mask", {1, 0}},
                                           {"end_mask", {1, 0}},
                                           {"new_axis_mask", {}},
                                           {"shrink_axis_mask", {}},
                                           {"ellipsis_mask", {}}});  //  tensor_array<f32[1,..4096,1,128]>
    auto mul_Multiply_552 =
        makePattern<opset1::Multiply>({slice_Slice_543, slice_Slice_446},
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

    auto reshape_opt2 = [&](std::shared_ptr<Node> input_BLHS) {
        return makePattern<opset1::Reshape>({input_BLHS, {0, 0, 0, 2, head_size / 2}},
                                            {{"special_zero", true}});  //  tensor_array<f32[?,?,32,2,64]>
    };

    auto ListUnpack_586_Split =
        makePattern<opset1::Split>({reshape_opt1(slice_Slice_543) | reshape_opt2(slice_Slice_543), -2},
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
    auto slice_Slice_470 =
        makePattern<opset1::StridedSlice>({rotary_emb_sin, ScatterUpdate_463814, {0, INT_MAX}, {1, 1}},
                                          {{"begin_mask", {1, 0}},
                                           {"end_mask", {1, 0}},
                                           {"new_axis_mask", {}},
                                           {"shrink_axis_mask", {}},
                                           {"ellipsis_mask", {}}});  //  tensor_array<f32[1,..4096,1,128]>
    auto mul_Multiply_594 =
        makePattern<opset1::Multiply>({cat_Concat_593, slice_Slice_470},
                                      {{"auto_broadcast", "numpy"}});  //  tensor_array<f32[?,?,32,128]>
    auto add_Add_597 = makePattern<opset1::Add>({mul_Multiply_552, mul_Multiply_594},
                                                {{"auto_broadcast", "numpy"}});  //  tensor_array<f32[?,?,32,128]>

    auto result = add_Add_597;

    matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto root = m.get_match_root();
        PatternValidator validator(m);
        if (!validator) {
            return false;
        }

        RoPENode::Config config;
        OutputVector new_args;
        config.is_qwen = true;
        config.head_cnt = validator["head_cnt"];
        config.head_size = validator["head_size"];
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

        new_args.push_back(pattern_map.at(qkv_proj));
        new_args.push_back(pattern_map.at(rotary_emb_cos));
        new_args.push_back(pattern_map.at(rotary_emb_sin));

        auto old_node = root;
        auto new_node = std::make_shared<RoPENode>(new_args, config);
        new_node->set_friendly_name(old_node->get_friendly_name());
        ov::replace_node(old_node, new_node);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(result, matcher_name);
    this->register_matcher(m, callback);
}