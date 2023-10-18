// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rope_fusion.hpp"

#include <cstdint>
#include <limits>
#include <openvino/core/rt_info.hpp>
#include <openvino/opsets/opset1.hpp>
#include <openvino/opsets/opset6.hpp>
#include <openvino/opsets/opset8.hpp>
#include <openvino/pass/pattern/op/or.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>
#include <transformations/utils/utils.hpp>

#include "itt.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "transformations/cpu_opset/x64/op/rope.hpp"
#include "utils/pattern_node.hpp"

#define CALLBACK_LOG(m) std::cout << matcher_name << " " << m.get_match_root()->get_friendly_name() << std::endl;

static size_t ROPE_cnt = 0;
static size_t ROPE_cnt_limit = []() {
    if (!std::getenv("USE_ROPE"))
        return 99999999;
    return atoi(std::getenv("USE_ROPE"));
}();

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
    auto x = GenPattern(ov::Rank(4));
    auto x_or_cos1 = GenPattern(ov::Rank(4));
    auto x_or_cos2 = GenPattern(ov::Rank(4));
    auto t_sin = GenPattern(ov::Rank(4));

    x->set_friendly_name("x");

    auto half_ndims = Symbol("half_ndims");
    auto int32_max = std::numeric_limits<std::int32_t>::max();

    // rotate half : [-x2, x1]
    auto x2 = GenSlice(x, half_ndims, int32_max, 1, 3, "x2");
    auto x2neg = GenPattern<opset1::Multiply>({x2, {-1}}, nullptr, {{"auto_broadcast", "numpy"}}, "x2neg");
    auto x1 = GenSlice(x, 0, half_ndims, 1, 3, "x1");
    auto x_rotate_half = GenPattern<opset1::Concat>({x2neg, x1}, nullptr, {{"axis", -1}}, "x_rotate_half");

    auto mul_cos = GenPattern<opset1::Multiply>({x_or_cos1, x_or_cos2}, nullptr, {{"auto_broadcast", "numpy"}}, "*cos");
    auto mul_sin = GenPattern<opset1::Multiply>({x_rotate_half, t_sin}, nullptr, {{"auto_broadcast", "numpy"}}, "*sin");

    // [x1, x2]*cos + [-x2, x1]*sin
    auto result = GenPattern<opset1::Add>({mul_cos, mul_sin}, nullptr, {{"auto_broadcast", "numpy"}}, "final");

    matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        ROPE_cnt++;
        if (ROPE_cnt > ROPE_cnt_limit) {
            return false;
        }
        std::cout << "================= ROPE_cnt: " << ROPE_cnt << "/" << ROPE_cnt_limit << std::endl;

        CALLBACK_LOG(m);

        const auto& pattern_map = m.get_pattern_value_map();
        auto root = m.get_match_root();
        std::map<std::string, double> symbol_name2value;
        if (!validate_matched_symbols(m, symbol_name2value)) {
            return false;
        }

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
        config.ndims = 2 * symbol_name2value["half_ndims"];

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

    auto cos_const = GenPattern<opset1::Constant>({});  // "f32[1,1,2048,24]"
    auto sin_const = GenPattern<opset1::Constant>({});  // "f32[1,1,2048,24]"

    auto node_batch_size = GenPattern("i32[1]");
    auto tile_batch = GenPattern("i32[1]");
    auto gather_positions = GenPattern("i32[?,?,?,?]");
    auto gather_positions_2d = GenPattern("i32[?,?]");

    auto tile_repeats = GenPattern<opset1::Concat>({tile_batch, {1}, {1}, {1}}, "i32[4]", {{"axis", 0}});

    auto prepare_cos_sin_gptneox = [&](std::shared_ptr<Node> const_tab) {
        auto slice1 = GenPattern<opset1::StridedSlice>({const_tab, {0}, node_batch_size, {1}},
                                                       nullptr,
                                                       {{"begin_mask", {0}},
                                                        {"end_mask", {0}},
                                                        {"new_axis_mask", {}},
                                                        {"shrink_axis_mask", {}},
                                                        {"ellipsis_mask", {}}});  //"f32[..1,1,2048,24]",
        auto slice2 = GenPattern<opset8::Slice>({const_tab, {0}, node_batch_size, {1}, {0}});
        auto tiled = GenPattern<opset1::Tile>({slice1 | slice2, tile_repeats});  // "f32[?,1,2048,24]"
        return GenPattern<opset6::GatherElements>({tiled, gather_positions}, nullptr, {{"axis", 2}});
    };

    auto seq_len = GenPattern("i32[1]");
    auto prepare_cos_sin_llama = [&](std::shared_ptr<Node> const_tab) {
        auto Slice_1 = GenPattern<opset8::Slice>({const_tab, {0}, seq_len, {1}, {2}});  // "f32[1, 1,..2048,128]"
        auto Squeeze_2 = GenPattern<opset1::Squeeze>({Slice_1, {1}});                   // "f32[1,..2048,128]"
        auto Squeeze_3 = GenPattern<opset1::Squeeze>({Squeeze_2, {0}});                 // "f32[..2048,128]"
        auto Gather_8 = GenPattern<opset8::Gather>({Squeeze_3, gather_positions_2d, 0},
                                                   nullptr,
                                                   {{"batch_dims", 0}});    // "f32[B,L,128]"
        auto Unsqueeze_7 = GenPattern<opset1::Unsqueeze>({Gather_8, {1}});  //  "f32[?,1,?,128]"
        return Unsqueeze_7;                                                 // B,H,L,S (H is broadcasted)
    };

    auto cos_tab = prepare_cos_sin_gptneox(cos_const) | prepare_cos_sin_llama(cos_const);
    auto sin_tab = prepare_cos_sin_gptneox(sin_const) | prepare_cos_sin_llama(sin_const);

    auto x = GenPattern(ov::Rank(4));
    auto rope = GenPattern<RoPENode>({x, cos_tab, sin_tab});

    matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        CALLBACK_LOG(m);
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
    auto data = GenPattern(ov::Rank(4));

    auto ndims = Symbol("ndims");
    auto x = GenSlice(data, 0, ndims, 1, 3);
    auto y = GenSlice(data, ndims, int32_max, 1, 3);
    auto x_emb = GenPattern<RoPENode>({x, {}, {}}) | GenPattern<RoPENode>({x, {}, {}, {}});
    auto result = GenPattern<opset1::Concat>({x_emb, y}, nullptr, {{"axis", -1}});

    matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        CALLBACK_LOG(m);

        const auto& pattern_map = m.get_pattern_value_map();
        auto root = m.get_match_root();

        auto rope_node = as_type_ptr<RoPENode>(root->input_value(0).get_node_shared_ptr());
        if (!rope_node)
            return false;

        std::map<std::string, double> symbol_name2value;
        if (!validate_matched_symbols(m, symbol_name2value)) {
            return false;
        }
        auto ndims = symbol_name2value["ndims"];

        auto& config = rope_node->get_config();
        if (config.ndims != ndims)
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
    auto input_to_slice = GenPattern(ov::Rank(4));
    auto input_to_trans = GenPattern(ov::Rank(4));  // no need to slice from 3S

    // in some model qkv prejection is combined and
    // needs to be sliced before RoPE
    auto slice_start = Symbol("slice_start");
    auto slice_stop = Symbol("slice_stop");
    auto input_slice = GenSlice(input_to_slice, slice_start, slice_stop, 1, 3);

    // some model will transpose from [B,L,H,S] to [B,H,L,S] before RoPE
    auto x = GenPattern<opset1::Transpose>({input_slice | input_to_trans, {0, 2, 1, 3}});
    auto result = GenPattern<RoPENode>({x, {}, {}}) | GenPattern<RoPENode>({x, {}, {}, {}});

    matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        CALLBACK_LOG(m);

        const auto& pattern_map = m.get_pattern_value_map();
        auto root = m.get_match_root();
        auto rope_node = as_type_ptr<RoPENode>(root);
        if (!rope_node)
            return false;
        std::map<std::string, double> symbol_name2value;
        if (!validate_matched_symbols(m, symbol_name2value)) {
            return false;
        }
        auto& config = rope_node->get_config();
        auto arg_size = rope_node->get_input_size();

        if (pattern_map.count(input_to_slice)) {
            config.slice_start = symbol_name2value["slice_start"];
            config.slice_stop = symbol_name2value["slice_stop"];
            config.input_trans0213 = true;
            rope_node->set_argument(0, pattern_map.at(input_to_slice));
        } else if (pattern_map.count(input_to_trans)) {
            config.input_trans0213 = true;
            rope_node->set_argument(0, pattern_map.at(input_to_trans));
        } else {
            return false;
        }
        register_new_node(rope_node);
        return true;
    };
    auto m = std::make_shared<ngraph::pattern::Matcher>(result, matcher_name);
    this->register_matcher(m, callback);
}

ov::intel_cpu::RoPEFusionConcatPast::RoPEFusionConcatPast() {
    MATCHER_SCOPE(RoPEFusionConcatPast);

    auto x = GenPattern(ov::Rank(4));
    auto past = GenPattern(ov::Rank(4));
    auto rope = GenPattern<RoPENode>({x, {}, {}}) | GenPattern<RoPENode>({x, {}, {}, {}});
    auto result = GenPattern<opset1::Concat>({past, rope}, nullptr, {{"axis", -2}}) |
                  GenPattern<opset1::Concat>({past, rope}, nullptr, {{"axis", 2}});

    matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        CALLBACK_LOG(m);

        const auto& pattern_map = m.get_pattern_value_map();
        auto root = m.get_match_root();
        auto rope_node = as_type_ptr<RoPENode>(root->input_value(1).get_node_shared_ptr());
        if (!rope_node)
            return false;
        auto& config = rope_node->get_config();
        if (config.concat_with_past_arg_id)
            return false;

        // append past to input args
        auto arg_id = rope_node->get_input_size();
        rope_node->set_argument(arg_id, pattern_map.at(past));
        config.concat_with_past_arg_id = arg_id;

        rope_node->set_friendly_name(root->get_friendly_name());
        ov::replace_node(root, rope_node);
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
            // stride is all 1
            auto stride = as_type_ptr<opset1::Constant>(inputs[3].get_node_shared_ptr());

            if (!begin)
                return false;
            if (!end)
                return false;
            if (!stride)
                return false;

            auto v_stride = stride->cast_vector<int32_t>();
            for (auto& v : v_stride) {
                if (v != 1)
                    return false;
            }

            auto v_begin = begin->cast_vector<int32_t>();
            auto v_end = end->cast_vector<int32_t>();

            auto& begin_mask = s1->get_begin_mask();
            auto& end_mask = s1->get_end_mask();
            auto mask_size = begin_mask.size();
            if (begin_mask.size() != end_mask.size()) {
                return false;
            }

            auto int32_max = std::numeric_limits<std::int32_t>::max();
            for (size_t i = 0; i < mask_size; i++) {
                if (begin_mask[i] != end_mask[i])
                    return false;
                // all valid [begin, end] are [0, int32_max]
                if (begin_mask[i] == 0 && end_mask[i] == 0) {
                    if (v_begin[i] != 0 || v_end[i] != int32_max)
                        return false;
                }
            }
            return true;
        });

    matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        CALLBACK_LOG(m);
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

    auto view_Reshape = GenPattern(ov::Rank(4));

    // view_Reshape : B,L,H,S
    auto slice_Slice_965 = GenSlice(view_Reshape, 0, ndims, 1, 3);

    auto cos_raw_3d = GenPattern(ov::Rank(3));  // f32[?,?,32]
    auto sin_raw_3d = GenPattern(ov::Rank(3));  // f32[?,?,32]
    auto cos_sin_gather_const = GenPattern<opset1::Constant>({});

    auto double_interleave = [&](std::shared_ptr<Node> node) {                    // f32[?,?,32]
        auto unsqueeze_Unsqueeze_230 = GenPattern<opset1::Unsqueeze>({node, 2});  // f32[?,?,1,32]
        auto repeat_interleave_Gather_251 =
            GenPattern<opset8::Gather>({unsqueeze_Unsqueeze_230, cos_sin_gather_const, 3},
                                       nullptr,
                                       {{"batch_dims", 0}});  // f32[?,?,1,64]
        return repeat_interleave_Gather_251;
    };

    auto t_cos = GenPattern(ov::Rank(4));
    auto t_sin = GenPattern(ov::Rank(4));

    // x interleave (-x[:,:,:, 1::2], x[:,:,:, 0::2])
    auto slice_Slice_1174 = GenSlice(slice_Slice_965, 1, int32_max, 2, 3);

    auto neg_Multiply_1177 = GenPattern<opset1::Multiply>({slice_Slice_1174, {-1}},
                                                          nullptr,
                                                          {{"auto_broadcast", "numpy"}});  // "f32[?,?,16,32]"
    auto Unsqueeze_65524 = GenPattern<opset1::Unsqueeze>({neg_Multiply_1177, -1});         // "f32[?,?,16,32,1]"

    auto slice_Slice_1168 = GenSlice(slice_Slice_965, 0, int32_max, 2, 3);
    auto Unsqueeze_65525 = GenPattern<opset1::Unsqueeze>({slice_Slice_1168, -1});  // "f32[?,?,16,32,1]"
    auto stack_1182 =
        GenPattern<opset1::Concat>({Unsqueeze_65524, Unsqueeze_65525}, nullptr, {{"axis", -1}});  // "f32[?,?,16,32,2]"

    auto ShapeOf_169068 = GenPattern<opset1::ShapeOf>({stack_1182}, "i32[5]", {});
    auto flatten_Slice_1194 = GenSlice(ShapeOf_169068, 0, 3, 1, 0);
    auto flatten_Concat_1197 = GenPattern<opset1::Concat>({flatten_Slice_1194, {-1}}, "i32[4]", {{"axis", 0}});
    auto flatten_Reshape_1198 = GenPattern<opset1::Reshape>({stack_1182, flatten_Concat_1197},
                                                            nullptr,
                                                            {{"special_zero", 1}});  // "f32[?,?,16,64]"

    // x*cos [B,L,H,ndims]
    auto mul_cos = GenPattern<opset1::Multiply>({slice_Slice_965, double_interleave(cos_raw_3d) | t_cos},
                                                nullptr,
                                                {{"auto_broadcast", "numpy"}});
    auto mul_sin = GenPattern<opset1::Multiply>({flatten_Reshape_1198, double_interleave(sin_raw_3d) | t_sin},
                                                nullptr,
                                                {{"auto_broadcast", "numpy"}});

    // *cos + *sin
    auto rotary_emb = GenPattern<opset1::Add>({mul_cos, mul_sin}, nullptr, {{"auto_broadcast", "numpy"}});

    auto slice_Slice_971 = GenSlice(view_Reshape, ndims, int32_max, 1, 3);
    auto cat_Concat_1211 = GenPattern<opset1::Concat>({rotary_emb, slice_Slice_971}, nullptr, {{"axis", -1}});
    auto permute_Transpose_1213 = GenPattern<opset1::Transpose>({cat_Concat_1211, {0, 2, 1, 3}});

    auto result = permute_Transpose_1213;

    matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        CALLBACK_LOG(m);

        const auto& pattern_map = m.get_pattern_value_map();
        auto root = m.get_match_root();
        std::map<std::string, double> symbol_name2value;
        if (!validate_matched_symbols(m, symbol_name2value)) {
            return false;
        }

        RoPENode::Config config;
        OutputVector new_args;
        config.ndims = symbol_name2value["ndims"];

        config.output_trans0213 = true;

        // input is [B,L,H,S]
        new_args.push_back(pattern_map.at(view_Reshape));
        if (pattern_map.count(cos_raw_3d)) {
            config.cos_is_raw3d = true;
            new_args.push_back(pattern_map.at(cos_raw_3d));
        } else {
            new_args.push_back(pattern_map.at(t_cos));
        }

        if (pattern_map.count(sin_raw_3d)) {
            config.sin_is_raw3d = true;
            new_args.push_back(pattern_map.at(sin_raw_3d));
        } else {
            new_args.push_back(pattern_map.at(t_sin));
        }

        auto old_node = root;

        auto new_node = std::make_shared<RoPENode>(new_args, config);
        new_node->set_friendly_name(old_node->get_friendly_name());
        ov::replace_node(old_node, new_node);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(result, matcher_name);
    this->register_matcher(m, callback);
}