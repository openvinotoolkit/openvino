// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/fuse_rotary_positional_embeddings.hpp"

#include <climits>
#include <cstdint>
#include <limits>
#include <variant>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/gather_elements.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/sin.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/shape_of_base.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/opsets/opset1_decl.hpp"
#include "openvino/opsets/opset6_decl.hpp"
#include "openvino/opsets/opset8_decl.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/common_util.hpp"
#include "ov_ops/rotary_positional_embeddings.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "transformations/symbolic_transformations/symbolic_optimizations.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov::pass;
using namespace ov::pass::pattern;
using namespace ov::op;

ov::pass::RoPEFusion::RoPEFusion(bool support_2d_rope) : m_support_2d_rope(support_2d_rope) {}

bool ov::pass::RoPEFusion::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_MODEL_SCOPE(RoPEFusion);
    ov::pass::SymbolicOptimizations symbolic_optimizations(false, get_pass_config());

    auto symbolic_ctx_manager = symbolic_optimizations.get_manager();

    symbolic_ctx_manager->register_pass<ov::pass::RoPEFusionFlux>();
    symbolic_ctx_manager->register_pass<ov::pass::RoPEFusionGPTNEOX>();
    symbolic_ctx_manager->register_pass<ov::pass::RoPEFusionGPTJ>();
    // optional heads & tails are fused in separate matcher pass,
    // after RoPENode has been created.
    symbolic_ctx_manager->register_pass<ov::pass::RoPEFusionCosSinPreprocess>();
    symbolic_ctx_manager->register_pass<ov::pass::RoPEFusionIOSlicing>();
    symbolic_ctx_manager->register_pass<ov::pass::RoPEFusionPreprocess>();

    symbolic_ctx_manager->register_pass<ov::pass::RoPEFusionChatGLM>(false);
    if (m_support_2d_rope) {
        symbolic_ctx_manager->register_pass<ov::pass::RoPEFusionChatGLM>(true);
        symbolic_ctx_manager->register_pass<ov::pass::RoPEFusionChatGLMHF>();
    }
    symbolic_ctx_manager->register_pass<ov::pass::RoPEFusionQwen>(0);
    symbolic_ctx_manager->register_pass<ov::pass::RoPEFusionQwen>(1);

    symbolic_ctx_manager->register_pass<ov::pass::RoPEShareCosSin>();

    return symbolic_optimizations.run_on_model(model);
}

static std::shared_ptr<ov::Node> gen_chatglm_const() {
    using namespace pattern;

    auto pred = value_matches("-1, head_cnt, 1, ndims/2, 1") || value_matches("1, -1, head_cnt, ndims/2, 1") ||
                value_matches("0, 0, 0, ndims/2, 1");
    return wrap_type<v0::Constant>(pred);
}

ov::pass::RoPEFusionFlux::RoPEFusionFlux() {
    MATCHER_SCOPE(RoPEFusionFlux);
    // x[?,24,?,128]
    // x1 = reshape(x, [?,24,?,64,2])
    // x1_0, x1_1 = split(x1, -1)
    // x2 = concat(x1_0, x1_1 * (-1), -1)
    // x3 = reshape(x2, [?,24,?,128])
    // y1 = x * t_cos
    // y2 = x3 * t_sin
    // y = y1 + y2
    auto x = pattern::any_input(pattern::rank_equals(4) && pattern::shape_matches("[PRESERVED_DIMS..., head_size]"));
    auto t_cos = pattern::any_input(pattern::rank_equals(4));
    auto t_sin = pattern::any_input(pattern::rank_equals(4));

    auto x1 = pattern::wrap_type<opset1::Reshape>({x, pattern::any_input()},
                                                  pattern::shape_matches("[PRESERVED_DIMS..., ?, 2]"));
    auto split = pattern::wrap_type<opset1::Split>({x1, -1}, {{"num_splits", 2}});
    split->set_output_size(2);

    // 3 versions of mulitply by -1 depending on transformations execution prior to this pass
    auto opt_squeeze = pattern::optional<opset1::Squeeze>({split->output(1), -1});
    auto x1_1_neg = pattern::wrap_type<opset1::Multiply>({opt_squeeze, -1}, {{"auto_broadcast", "numpy"}});
    auto opt_squeeze_1 = pattern::optional<opset1::Squeeze>({x1_1_neg, -1});
    auto opt_unsqueeze = pattern::optional<opset1::Unsqueeze>({opt_squeeze_1, -1});

    auto x2 = pattern::wrap_type<opset1::Concat>({opt_unsqueeze, split->output(0)}, {{"axis", -1}});
    auto x3 = pattern::wrap_type<opset1::Reshape>({x2, pattern::any_input()},
                                                  pattern::shape_matches("[PRESERVED_DIMS..., head_size]"));

    auto y1 = pattern::wrap_type<opset1::Multiply>({x, t_cos}, {{"auto_broadcast", "numpy"}});
    auto y2 = pattern::wrap_type<opset1::Multiply>({x3, t_sin}, {{"auto_broadcast", "numpy"}});

    auto result = pattern::wrap_type<opset1::Add>({y1, y2}, {{"auto_broadcast", "numpy"}});

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto root = m.get_match_root();

        auto symbols = m.get_symbols();
        auto num_heads = symbols["PRESERVED_DIMS"].g()[1];
        auto head_size = symbols["head_size"];
        if (!num_heads.is_static() || !head_size.is_static()) {
            return false;
        }

        op::internal::RoPE::Config config;
        config.head_cnt = static_cast<size_t>(num_heads.i());
        config.head_size = static_cast<size_t>(head_size.i());
        config.rotary_ndims = config.head_size;
        config.is_interleaved = true;
        config.output_trans0213 = false;

        OutputVector new_args;
        new_args.push_back(pattern_map.at(x));
        new_args.push_back(pattern_map.at(t_cos));
        new_args.push_back(pattern_map.at(t_sin));

        auto old_node = root;
        auto new_node = std::make_shared<op::internal::RoPE>(new_args, config);
        new_node->set_friendly_name(old_node->get_friendly_name());
        ov::copy_runtime_info({pattern_map.at(x1).get_node_shared_ptr(),
                               pattern_map.at(split).get_node_shared_ptr(),
                               pattern_map.at(x2).get_node_shared_ptr(),
                               pattern_map.at(x3).get_node_shared_ptr(),
                               pattern_map.at(y1).get_node_shared_ptr(),
                               pattern_map.at(y2).get_node_shared_ptr(),
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
using symbol_variant = std::variant<float, int32_t, int64_t, std::string>;

static std::string ParseSymbolVariant(std::vector<symbol_variant> values) {
    std::vector<std::string> symbol_strings;
    symbol_strings.reserve(values.size());
    for (auto& value : values) {
        if (std::holds_alternative<float>(value)) {
            symbol_strings.push_back(std::to_string(std::get<float>(value)));
        } else if (std::holds_alternative<int>(value)) {
            symbol_strings.push_back(std::to_string(std::get<int>(value)));
        } else if (std::holds_alternative<int32_t>(value)) {
            symbol_strings.push_back(std::to_string(std::get<int32_t>(value)));
        } else if (std::holds_alternative<int64_t>(value)) {
            symbol_strings.push_back(std::to_string(std::get<int64_t>(value)));
        } else {
            symbol_strings.push_back(std::get<std::string>(value));
        }
    }

    return ov::util::join(symbol_strings);
}

static std::shared_ptr<ov::Node> NewGenSlice(std::shared_ptr<ov::Node> data,
                                             symbol_variant start,
                                             symbol_variant stop,
                                             symbol_variant step,
                                             size_t axis) {
    auto slice_start = ParseSymbolVariant({start});
    auto slice_stop = ParseSymbolVariant({stop});
    auto slice_step = ParseSymbolVariant({step});
    auto slice_axis = ParseSymbolVariant({static_cast<int64_t>(axis)});

    auto opt1 = pattern::wrap_type<ov::opset8::Slice>({data, slice_start, slice_stop, slice_step, slice_axis});

    std::vector<symbol_variant> vbegin(axis + 1, 0);
    std::vector<symbol_variant> vend(axis + 1, 0);
    std::vector<symbol_variant> vstride(axis + 1, 1);

    vbegin[axis] = start;
    vend[axis] = stop;
    vstride[axis] = step;

    auto begin = ParseSymbolVariant(vbegin);
    auto end = ParseSymbolVariant(vend);
    auto stride = ParseSymbolVariant(vstride);

    std::vector<int64_t> begin_mask(axis + 1, 1);
    std::vector<int64_t> end_mask(axis + 1, 1);
    std::vector<int64_t> new_axis_mask;
    std::vector<int64_t> shrink_axis_mask;
    std::vector<int64_t> ellipsis_mask;

    begin_mask[axis] = 0;
    end_mask[axis] = 0;

    auto opt2 = pattern::wrap_type<ov::op::v1::StridedSlice>({data, begin, end, stride},
                                                             {{"begin_mask", begin_mask},
                                                              {"end_mask", end_mask},
                                                              {"new_axis_mask", new_axis_mask},
                                                              {"shrink_axis_mask", shrink_axis_mask},
                                                              {"ellipsis_mask", ellipsis_mask}});

    return opt1 | opt2;
}

static std::shared_ptr<ov::Node> NewGenStridedSlice(std::shared_ptr<ov::Node> data,
                                                    const pattern::PatternOp& start,
                                                    const pattern::PatternOp& stop,
                                                    const pattern::PatternOp& step,
                                                    size_t axis) {
    std::vector<int64_t> begin_mask(axis + 1, 1);
    std::vector<int64_t> end_mask(axis + 1, 1);
    std::vector<int64_t> new_axis_mask;
    std::vector<int64_t> shrink_axis_mask;
    std::vector<int64_t> ellipsis_mask;

    begin_mask[axis] = 0;
    end_mask[axis] = 0;

    return pattern::wrap_type<ov::op::v1::StridedSlice>({data, start, stop, step},
                                                        {{"begin_mask", begin_mask},
                                                         {"end_mask", end_mask},
                                                         {"new_axis_mask", new_axis_mask},
                                                         {"shrink_axis_mask", shrink_axis_mask},
                                                         {"ellipsis_mask", ellipsis_mask}});
}

ov::pass::RoPEFusionGPTNEOX::RoPEFusionGPTNEOX() {
    using namespace ov::op;
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
    auto x = pattern::any_input(pattern::rank_equals(4));
    auto x_or_cos1 = pattern::any_input(pattern::rank_equals(4));
    auto x_or_cos2 = pattern::any_input(pattern::rank_equals(4));
    auto t_sin = pattern::any_input(pattern::rank_equals(4));

    auto varsplit = pattern::wrap_type<v1::VariadicSplit>({x, 3, {"half_ndims", "?"}});
    varsplit->set_output_size(2);

    auto int32_max = std::numeric_limits<std::int32_t>::max();

    auto x2 = NewGenSlice(x, "half_ndims", int32_max, 1, 3);
    auto x2neg = pattern::wrap_type<v1::Multiply>({x2 | varsplit->output(1), -1.0f}, {{"auto_broadcast", "numpy"}});
    auto x1 = NewGenSlice(x, 0, "half_ndims", 1, 3);
    auto x_rotate_half = pattern::wrap_type<v0::Concat>({x2neg, x1 | varsplit->output(0)}, {{"axis", -1}});

    auto mul_cos = pattern::wrap_type<v1::Multiply>({x_or_cos1, x_or_cos2}, {{"auto_broadcast", "numpy"}});
    auto mul_sin = pattern::wrap_type<v1::Multiply>({x_rotate_half, t_sin}, {{"auto_broadcast", "numpy"}});

    auto result = pattern::wrap_type<v1::Add>({mul_cos, mul_sin}, {{"auto_broadcast", "numpy"}});

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
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

        auto symbols = m.get_symbols();
        auto half_ndims = symbols["half_ndims"];
        if (!half_ndims.is_integer()) {
            return false;
        }

        op::internal::RoPE::Config config;
        OutputVector new_args;
        config.rotary_ndims = 2ul * static_cast<size_t>(half_ndims.i());

        new_args.push_back(pattern_map.at(x));
        new_args.push_back(v_cos);
        new_args.push_back(pattern_map.at(t_sin));
        auto old_node = root;
        auto new_node = std::make_shared<internal::RoPE>(new_args, config);
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

    auto cos_const = pattern::wrap_type<v0::Constant>(pattern::type_matches(element::f32));
    auto sin_const = pattern::wrap_type<v0::Constant>(pattern::type_matches(element::f32));

    auto node_batch_size = pattern::any_input(pattern::type_matches(element::i32) && pattern::shape_matches("[1]"));
    auto tile_batch = pattern::any_input(pattern::type_matches(element::i32) && pattern::shape_matches("[1]"));
    auto gather_positions = pattern::any_input(pattern::type_matches(element::i32) && pattern::rank_equals(4));

    auto prepare_cos_sin_gptneox = [&](std::shared_ptr<Node> const_tab) {
        auto slice = pattern::wrap_type<v8::Slice>({const_tab, {0}, node_batch_size, {1}, {0}});
        auto strided_slice = NewGenStridedSlice(const_tab, {0}, node_batch_size, {1}, 0);
        return pattern::wrap_type<v6::GatherElements>({strided_slice | slice, gather_positions}, {{"axis", 2}});
    };

    auto seq_len = pattern::any_input(pattern::type_matches(element::i32) && pattern::shape_matches("[1]"));
    auto gather_positions_2d = pattern::any_input(pattern::type_matches(element::i32) && pattern::rank_equals(2));

    auto prepare_cos_sin_llama = [&](std::shared_ptr<Node> const_tab) {
        auto ScatterUpdate = pattern::wrap_type<v3::ScatterUpdate>({{0, 0, 0}, 2, seq_len, 0});
        auto slice_Slice = pattern::wrap_type<v8::Slice>({const_tab, {0}, seq_len, {1}, {2}});
        auto slice_StridedSlice = NewGenStridedSlice(const_tab, {0, 0, 0}, ScatterUpdate, {1, 1, 1}, 2);
        auto squeeze = pattern::wrap_type<v1::Reshape>({slice_StridedSlice | slice_Slice, pattern::any_input()},
                                                       pattern::shape_matches("[?, head_dims]"));
        auto index_Gather = pattern::wrap_type<v8::Gather>({squeeze, gather_positions_2d, 0}, {{"batch_dims", 0}});

        // another simplified pattern for gathering at position_ids
        auto slice_Slice2 = pattern::wrap_type<v8::Slice>({const_tab, {0}, seq_len, {1}, {0}});
        auto slice_StridedSlice2 = NewGenStridedSlice(const_tab, {0}, seq_len, {1}, 0);
        auto index_Gather2 =
            pattern::wrap_type<v8::Gather>({slice_Slice2 | slice_StridedSlice2, gather_positions_2d, 0},
                                           {{"batch_dims", 0}});

        auto unsqueeze = pattern::wrap_type<v1::Reshape>({index_Gather | index_Gather2, pattern::any_input()},
                                                         pattern::shape_matches("[1, 1, ?, head_dims]"));
        auto unsqueeze2 = pattern::wrap_type<v0::Unsqueeze>({index_Gather2, 1});

        return unsqueeze2 | unsqueeze;
    };

    auto cos_tab = prepare_cos_sin_gptneox(cos_const) | prepare_cos_sin_llama(cos_const);
    auto sin_tab = prepare_cos_sin_gptneox(sin_const) | prepare_cos_sin_llama(sin_const);

    auto x = pattern::any_input(pattern::rank_equals(4));
    auto rope = pattern::wrap_type<op::internal::RoPE>({x, cos_tab, sin_tab});

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
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
    auto data = pattern::any_input(pattern::rank_equals(4));
    auto varsplit = pattern::wrap_type<op::v1::VariadicSplit>({data, 3, {"ndims", "?"}});
    varsplit->set_output_size(2);

    auto x = NewGenSlice(data, 0, "ndims", 1, 3);
    auto y = NewGenSlice(data, "ndims", int32_max, 1, 3);
    auto x_emb =
        pattern::wrap_type<op::internal::RoPE>({x | varsplit->output(0), pattern::any_input(), pattern::any_input()}) |
        pattern::wrap_type<op::internal::RoPE>(
            {x | varsplit->output(0), pattern::any_input(), pattern::any_input(), pattern::any_input()});
    auto result = pattern::wrap_type<op::v0::Concat>({x_emb, y | varsplit->output(1)}, {{"axis", -1}});

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto root = m.get_match_root();

        auto rope_node = as_type_ptr<op::internal::RoPE>(root->input_value(0).get_node_shared_ptr());
        if (!rope_node)
            return false;

        auto symbols = m.get_symbols();
        auto ndims = symbols["ndims"];
        if (!ndims.is_static() || !ndims.is_integer())
            return false;

        const auto& config = rope_node->get_config();
        if (config.rotary_ndims != static_cast<size_t>(ndims.i()))
            return false;

        // remove slice & concat
        rope_node->input(0).replace_source_output(pattern_map.at(data));
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

// gptneox-preprocess of input data
ov::pass::RoPEFusionPreprocess::RoPEFusionPreprocess() {
    MATCHER_SCOPE(RoPEFusionPreprocess);

    // Pattern for input to be sliced (for models with combined QKV projection)
    auto input_to_slice = pattern::any_input(pattern::rank_equals(4));
    // Pattern for input to be transposed (for models that transpose before RoPE)
    auto input_to_trans = pattern::any_input(pattern::rank_equals(4));

    // Slice input if needed: [B, L, H, S] -> [B, L', H, S]
    auto input_slice = NewGenSlice(input_to_slice, {"slice_start"}, {"slice_stop"}, 1, 3);

    // Transpose input if needed: [B, L, H, S] -> [B, H, L, S]
    auto x = pattern::wrap_type<op::v1::Transpose>({input_slice | input_to_trans, {0, 2, 1, 3}});

    // RoPE node: supports both 3 and 4 inputs
    auto result =
        pattern::wrap_type<op::internal::RoPE>({x, pattern::any_input(), pattern::any_input()}) |
        pattern::wrap_type<op::internal::RoPE>({x, pattern::any_input(), pattern::any_input(), pattern::any_input()});

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto root = m.get_match_root();
        auto rope_node = as_type_ptr<op::internal::RoPE>(root);
        if (!rope_node)
            return false;

        auto symbols = m.get_symbols();
        auto slice_start = symbols["slice_start"];
        auto slice_stop = symbols["slice_stop"];

        auto config = rope_node->get_config();
        if (pattern_map.count(input_to_slice) && slice_start.is_integer() && slice_stop.is_integer()) {
            config.slice_start = static_cast<size_t>(slice_start.i());
            config.slice_stop = static_cast<size_t>(slice_stop.i());
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
    auto m = std::make_shared<pattern::Matcher>(result, matcher_name);
    this->register_matcher(m, callback);
}

auto const_idx_predicate = [](pattern::PatternSymbolMap&, const ov::Output<ov::Node>& out) {
    if (auto const_node = ov::as_type_ptr<ov::opset1::Constant>(out.get_node_shared_ptr())) {
        const auto& vec = const_node->get_vector<int32_t>();
        int32_t v = 0;
        for (size_t i = 0; i < vec.size(); i += 2, v++)
            if (vec[i] != v || vec[i + 1] != v)
                return false;
        return true;
    }
    return false;
};

static std::shared_ptr<ov::Node> repeat_interleave_pattern(const ov::Output<ov::Node>& var_split_output) {
    auto unsqueeze = pattern::wrap_type<ov::opset1::Reshape>({var_split_output, {"dim0", "dim1", "1", "32"}}) |
                     pattern::wrap_type<ov::opset1::Unsqueeze>({var_split_output, 2});
    // repeate cos/sin table
    auto const_idx =
        pattern::wrap_type<ov::opset1::Constant>(pattern::type_matches(ov::element::i32) && const_idx_predicate);
    return pattern::wrap_type<ov::opset8::Gather>({unsqueeze, const_idx, 3}, {{"batch_dims", 0}});
}

ov::pass::RoPEFusionGPTJ::RoPEFusionGPTJ() {
    MATCHER_SCOPE(RoPEFusionGPTJ);

    auto gather_sin_cos = pattern::any_input(pattern::type_matches(ov::element::f32));
    auto varsplit = pattern::wrap_type<opset1::VariadicSplit>({gather_sin_cos, -1, {"ndims/2", "-1"}});
    varsplit->set_output_size(2);
    auto repeat_interleave_sin = repeat_interleave_pattern(varsplit->output(0));
    auto repeat_interleave_cos = repeat_interleave_pattern(varsplit->output(1));

    auto view_Reshape = pattern::any_input(pattern::rank_equals(4));
    auto slice_Slice_965 = NewGenSlice(view_Reshape, 0, "ndims", 1, 3);
    // view_Reshape : B,L,H,S
    auto varsplit_view_Reshape = pattern::wrap_type<opset1::VariadicSplit>({view_Reshape, 3, {"ndims", "end"}});
    varsplit_view_Reshape->set_output_size(2);
    // x interleave (-x[:,:,:, 1::2], x[:,:,:, 0::2])
    auto slice_Slice_1174 = NewGenSlice(slice_Slice_965 | varsplit_view_Reshape->output(0), 1, INT_MAX, 2, 3);

    auto neg_Multiply_1177 =
        pattern::wrap_type<opset1::Multiply>({slice_Slice_1174, -1.0f}, {{"auto_broadcast", "numpy"}});
    auto Unsqueeze_65524 = pattern::wrap_type<opset1::Unsqueeze>({neg_Multiply_1177, -1});
    auto Unsqueeze_28998 = pattern::wrap_type<opset1::Reshape>({neg_Multiply_1177, {"-1", "1", "head_num", "32", "1"}},
                                                               {{"special_zero", false}});

    auto slice_Slice_1168 = NewGenSlice(slice_Slice_965 | varsplit_view_Reshape->output(0), 0, INT_MAX, 2, 3);
    auto Unsqueeze_65525 = pattern::wrap_type<opset1::Unsqueeze>({slice_Slice_1168, -1});
    auto Unsqueeze_28999 = pattern::wrap_type<opset1::Reshape>({slice_Slice_1168, {"-1", "1", "head_num", "32", "1"}},
                                                               {{"special_zero", false}});
    auto stack_1182 =
        pattern::wrap_type<opset1::Concat>({Unsqueeze_65524 | Unsqueeze_28998, Unsqueeze_65525 | Unsqueeze_28999},
                                           {{"axis", -1}});

    auto ShapeOf_169068 = pattern::wrap_type<opset1::ShapeOf>({stack_1182});
    auto flatten_Slice_1194 = NewGenSlice(ShapeOf_169068, 0, 3, 1, 0);
    auto flatten_Concat_1197 = pattern::wrap_type<opset1::Concat>({flatten_Slice_1194, {-1}}, {{"axis", 0}});
    // If with special zero, no need to use shapeof to get full shape
    auto flatten_Reshape_1198 = pattern::wrap_type<opset1::Reshape>({stack_1182, flatten_Concat_1197});
    auto flatten_Reshape_Zero =
        pattern::wrap_type<opset1::Reshape>({stack_1182, ov::pass::pattern::any_input()}, {{"special_zero", true}});

    // x*cos [B,L,H,ndims]
    auto mul_cos = pattern::wrap_type<opset1::Multiply>(
        {slice_Slice_965 | varsplit_view_Reshape->output(0), repeat_interleave_cos},
        {{"auto_broadcast", "numpy"}});
    auto mul_sin =
        pattern::wrap_type<opset1::Multiply>({flatten_Reshape_1198 | flatten_Reshape_Zero, repeat_interleave_sin},
                                             {{"auto_broadcast", "numpy"}});

    // *cos + *sin
    auto rotary_emb = pattern::wrap_type<opset1::Add>({mul_cos, mul_sin}, {{"auto_broadcast", "numpy"}});

    auto slice_Slice_971 = NewGenSlice(view_Reshape, "ndims", INT_MAX, 1, 3);
    auto result = pattern::wrap_type<opset1::Concat>({rotary_emb, slice_Slice_971 | varsplit_view_Reshape->output(1)},
                                                     {{"axis", -1}});

    matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto root = m.get_match_root();
        auto symbols = m.get_symbols();

        auto ndims = symbols["ndims"];
        auto ndims_over_2 = symbols["ndims/2"];
        if (!ndims.is_integer() || !ndims_over_2.is_integer() || ndims_over_2.i() * 2 != ndims.i()) {
            return false;
        }

        op::internal::RoPE::Config config;
        OutputVector new_args;
        NodeVector rt_from = {pattern_map.at(varsplit).get_node_shared_ptr(),
                              pattern_map.at(repeat_interleave_sin).get_node_shared_ptr(),
                              pattern_map.at(repeat_interleave_cos).get_node_shared_ptr(),
                              pattern_map.at(neg_Multiply_1177).get_node_shared_ptr(),
                              pattern_map.at(stack_1182).get_node_shared_ptr(),
                              pattern_map.at(mul_cos).get_node_shared_ptr(),
                              pattern_map.at(mul_sin).get_node_shared_ptr(),
                              pattern_map.at(rotary_emb).get_node_shared_ptr(),
                              pattern_map.at(result).get_node_shared_ptr()};
        config.rotary_ndims = static_cast<size_t>(ndims.i());

        // Fuse output transpose to Rope.
        auto root_target_inputs = root->output(0).get_target_inputs();
        if (root_target_inputs.size() == 1) {
            auto target_node = root_target_inputs.begin()->get_node()->shared_from_this();
            if (auto transpose = ov::as_type_ptr<op::v1::Transpose>(target_node)) {
                auto axes = transpose->input_value(1).get_node_shared_ptr();
                auto axes_const = ov::as_type_ptr<op::v0::Constant>(axes);
                if (axes_const && axes_const->cast_vector<int64_t>() == std::vector<int64_t>{0, 2, 1, 3}) {
                    config.output_trans0213 = true;
                    rt_from.push_back(target_node);
                    root = target_node;
                }
            }
        }
        config.is_interleaved = true;
        // input is [B,L,H,S]
        new_args.push_back(pattern_map.at(view_Reshape));
        // sin_cos table (gathered with positions) [1, L, 64]
        new_args.push_back(pattern_map.at(gather_sin_cos));
        new_args.push_back(pattern_map.at(gather_sin_cos));
        auto old_node = root;
        auto new_node = std::make_shared<op::internal::RoPE>(new_args, config);
        new_node->set_friendly_name(old_node->get_friendly_name());
        ov::copy_runtime_info(rt_from, new_node);
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

ov::pass::RoPEFusionChatGLM::RoPEFusionChatGLM(const bool support_2d_rope) {
    MATCHER_SCOPE(RoPEFusionChatGLM);

    //  [seq_length, batch_size, input_size(will be cropped to match hidden state size)]
    //  [batch_size, seq_length, input_size] support_2d_rope
    auto qkv_linear = pattern::any_input(pattern::rank_equals(3));
    auto seq_length = pattern::any_input(pattern::type_matches(element::i32) && pattern::shape_matches("[1]"));
    // [max_pos_embeddings, batch_size, half_rotary_dims, 2]
    // [batch_size, max_pos_embeddings, half_rotary_dims, 2] support_2d_rope
    auto cos_sin_cache = pattern::any_input(pattern::rank_equals(4));

    auto qkv_proj =
        pattern::wrap_type<v1::VariadicSplit>({qkv_linear, -1, {"total_size_q", "total_size_k", "total_size_v"}});
    qkv_proj->set_output_size(3);
    auto reshape_pattern_const = pattern::wrap_type<v0::Constant>(pattern::value_matches("0, 0, head_cnt, head_size"));
    auto cur_key =
        pattern::wrap_type<v1::Reshape>({qkv_proj, reshape_pattern_const},
                                        {{"special_zero", true}});
    std::shared_ptr<ov::Node> input_key = nullptr;
    // Extended the RoPE to a two-dimensional form to accommodate the 2D positional encoding in GLM.
    // Calculate positional embedding independent of batch and each head
    if (support_2d_rope) {
        // Get transposed key [batch, head_cnt, seq_length, head_size]
        // For Models, where SDPA to PagedAttention transformation was applied,
        // all sequences have the size == 1, we move sequences to the batch, this is the PagedAttention specific,
        // so seq_length dim will be always 1, this means that Transpose is unnecessary and Reshape op can be used.
        auto transposed_cur_key =
            pattern::wrap_type<v1::Reshape>({qkv_proj, {"-1", "head_cnt", "1", "head_size"}},
                                            {{"special_zero", false}});
        // Transpose for SDPA version:
        input_key = pattern::wrap_type<v1::Transpose>({cur_key, {0, 2, 1, 3}}) | transposed_cur_key;
    } else {
        // Get key [seq_length, batch, head_cnt, head_size]
        input_key = std::move(cur_key);
    }

    auto slice0 = NewGenSlice(input_key, 0, "ndims", 1, 3);
    auto var_split0 = pattern::wrap_type<v1::VariadicSplit>({input_key, 3, {"ndims", "end"}});
    var_split0->set_output_size(2);

    // rotate half
    std::shared_ptr<ov::Node> reshape0 = nullptr;
    if (support_2d_rope) {
        auto const_target_shape0 =
            pattern::wrap_type<v0::Constant>(pattern::value_matches("0, head_cnt, 0, ndims/2, 2"));
        reshape0 = pattern::wrap_type<v1::Reshape>({slice0 | var_split0->output(0), const_target_shape0},
                                                   {{"special_zero", true}});
    } else {
        auto concat0 =
            pattern::wrap_type<v0::Concat>({seq_length, {-1}, {"head_cnt"}, {"ndims/2"}, {2}}, {{"axis", 0}});
        auto const_target_shape1 =
            pattern::wrap_type<v0::Constant>(pattern::value_matches("0, 0, head_cnt, ndims/2, 2"));
        auto const_target_shape2 =
            pattern::wrap_type<v0::Constant>(pattern::value_matches("seq_len, batch, head_cnt, ndims/2, 2"));
        reshape0 = pattern::wrap_type<v1::Reshape>(
            {slice0 | var_split0->output(0), concat0 | const_target_shape1 | const_target_shape2});
    }

    auto x_even = pattern::wrap_type<v8::Gather>({reshape0, 0, -1}, {{"batch_dims", 0}});
    auto x_odd = pattern::wrap_type<v8::Gather>({reshape0, 1, -1}, {{"batch_dims", 0}});

    auto var_split1 = pattern::wrap_type<v1::VariadicSplit>({cos_sin_cache, 0, {"0", "end"}});
    var_split1->set_output_size(2);

    std::shared_ptr<ov::Node> reshape1 = nullptr;
    if (support_2d_rope) {
        // Slice cos_sin_cache to support 2-dimentional RoPE
        auto scatter_update0 = pattern::wrap_type<v3::ScatterUpdate>({{0, 0}, {1}, seq_length, {0}}, {});
        auto slice1 = pattern::wrap_type<v8::Slice>({cos_sin_cache, {0}, seq_length, {1}, {1}});
        auto slice2 = pattern::wrap_type<v8::Slice>({cos_sin_cache, {0, 0}, scatter_update0, {1, 1}, {0}});
        auto ss_stop = pattern::wrap_type<v0::Constant>();
        auto strided_slice0 = NewGenStridedSlice(cos_sin_cache, {0, 0}, ss_stop | scatter_update0, {1, 1}, 1);

        auto concat1 = pattern::wrap_type<v0::Concat>({{-1}, {1}, seq_length, {"ndims/2"}, {2}}, {{"axis", 0}});
        auto const_target_shape3 =
            pattern::wrap_type<v0::Constant>(pattern::value_matches("batch, 1, seq_len, ndims/2, 2"));

        // [batch, 1, seq_length, half_rotary_dims, 2]
        reshape1 = pattern::wrap_type<v1::Reshape>(
            {strided_slice0 | slice1 | slice2 | var_split1->output(0), concat1 | const_target_shape3});
    } else {
        auto concat2 = pattern::wrap_type<v0::Concat>({seq_length, {-1}, {1}, {"ndims/2"}, {2}}, {{"axis", 0}});
        auto const_target_shape4 = pattern::wrap_type<v0::Constant>(pattern::value_matches("1, -1, 1, ndims/2, 2"));
        auto const_target_shape5 =
            pattern::wrap_type<v0::Constant>(pattern::value_matches("seq_len, batch, 1, ndims/2, 2"));

        auto slice3 = pattern::wrap_type<v8::Slice>({cos_sin_cache, {0}, seq_length, {1}, {0}});
        auto strided_slice1 = NewGenStridedSlice(cos_sin_cache, {0}, seq_length, {1}, 0);

        // [seq_length, 1, batch, half_rotary_dims, 2]
        reshape1 = pattern::wrap_type<v1::Reshape>(
            {strided_slice1 | slice3 | var_split1->output(0), concat2 | const_target_shape4 | const_target_shape5});
    }

    auto cos_tab = pattern::wrap_type<v8::Gather>({reshape1, 0, -1}, {{"batch_dims", 0}});
    auto x_even_cos = pattern::wrap_type<v1::Multiply>({x_even, cos_tab}, {{"auto_broadcast", "numpy"}});

    auto sin_tab = pattern::wrap_type<v8::Gather>({reshape1, 1, -1}, {{"batch_dims", 0}});
    auto x_odd_sin = pattern::wrap_type<v1::Multiply>({x_odd, sin_tab}, {{"auto_broadcast", "numpy"}});
    auto neg_x_odd_sin = pattern::wrap_type<v1::Multiply>({x_odd_sin, -1.000000f}, {{"auto_broadcast", "numpy"}});
    auto add0 = pattern::wrap_type<v1::Add>({x_even_cos, neg_x_odd_sin}, {{"auto_broadcast", "numpy"}});
    auto y_even = pattern::wrap_type<v0::Unsqueeze>({add0, -1}) |
                  pattern::wrap_type<v1::Reshape>({add0, gen_chatglm_const()}, {{"special_zero", false}});
    auto x_odd_cos = pattern::wrap_type<v1::Multiply>({x_odd, cos_tab}, {{"auto_broadcast", "numpy"}});
    auto x_even_sin = pattern::wrap_type<v1::Multiply>({x_even, sin_tab}, {{"auto_broadcast", "numpy"}});
    auto add1 = pattern::wrap_type<v1::Add>({x_odd_cos, x_even_sin}, {{"auto_broadcast", "numpy"}});
    auto y_odd = pattern::wrap_type<v0::Unsqueeze>({add1, -1}) |
                 pattern::wrap_type<v1::Reshape>({add1, gen_chatglm_const()}, {{"special_zero", false}});

    auto concat2 = pattern::wrap_type<v0::Concat>({y_even, y_odd}, {{"axis", -1}});

    auto shape_of0 = pattern::wrap_type<ov::op::util::ShapeOfBase>({concat2});
    auto slice4 = NewGenSlice(shape_of0, 0, 3, 1, 0);
    auto concat3 = pattern::wrap_type<v0::Concat>({slice4, {-1}}, {{"axis", 0}});

    std::shared_ptr<ov::Node> const_target_shape6 = nullptr;
    std::shared_ptr<ov::Node> reshape2 = nullptr;
    if (support_2d_rope) {
        // [batch, head_cnt, length, half_rotary_dims, 2]
        const_target_shape6 =
            pattern::wrap_type<v0::Constant>(pattern::value_matches("batch, head_cnt, seq_len, ndims") ||
                                             pattern::value_matches("0, head_cnt, 0, ndims"));
        reshape2 = pattern::wrap_type<v1::Reshape>({concat2, concat3 | const_target_shape6}, {{"special_zero", true}});
    } else {
        // [length, batch, head_cnt, half_rotary_dims, 2]
        auto const_target_shape7 = pattern::wrap_type<v0::Constant>(pattern::value_matches("0, 0, head_cnt, ndims"));
        const_target_shape6 =
            pattern::wrap_type<v0::Constant>(pattern::value_matches("seq_len, batch, head_cnt, ndims"));
        reshape2 = pattern::wrap_type<v1::Reshape>({concat2, concat3 | const_target_shape6 | const_target_shape7},
                                                   {{"special_zero", true}});
    }
    auto slice5 = NewGenSlice(input_key, "ndims", INT_MAX, 1, 3);

    auto concat4 = pattern::wrap_type<v0::Concat>({reshape2, slice5 | var_split0->output(1)}, {{"axis", -1}});
    auto result = concat4 | reshape2;

    matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto root = m.get_match_root();
        auto symbols = m.get_symbols();
        
        auto ndims_over_2 = symbols["ndims/2"];
        auto ndims = symbols["ndims"];
        auto head_cnt = symbols["head_cnt"];
        auto head_size = symbols["head_size"];
        auto total_size_q = symbols["total_size_q"];
        auto total_size_k = symbols["total_size_k"];

        if (!ndims_over_2.is_integer() || !ndims.is_integer() || !head_cnt.is_integer() || !head_size.is_integer() ||
            !total_size_q.is_integer() || !total_size_k.is_integer() || ndims_over_2.i() * 2 != ndims.i()) {
            return false;
        }
        
        op::internal::RoPE::Config config;
        OutputVector new_args;
        config.rotary_ndims = static_cast<size_t>(ndims.i());
        config.is_chatglm = true;
        config.support_2d_rope = support_2d_rope;
        config.use_rope_cache = true;
        config.head_cnt = static_cast<size_t>(head_cnt.i());
        config.head_size = static_cast<size_t>(head_size.i());

        const auto& qkv_proj_node = pattern_map.at(qkv_proj);
        const size_t qkv_proj_output_id = qkv_proj_node.get_index();
        if (qkv_proj_output_id == 0) {
            // query : split output id == 0
            config.slice_start = 0;
            config.slice_stop = static_cast<size_t>(total_size_q.i());
        } else if (qkv_proj_output_id == 1) {
            // key : split output id == 1
            config.slice_start = static_cast<size_t>(total_size_q.i());
            config.slice_stop = static_cast<size_t>(config.slice_start + static_cast<size_t>(total_size_k.i()));
        } else {
            return false;
        }

        if (ov::is_type<opset1::Reshape>(root)) {
            if (config.rotary_ndims != config.head_size)
                return false;
        }

        new_args.push_back(pattern_map.at(qkv_linear));
        new_args.push_back(pattern_map.at(cos_sin_cache));
        new_args.push_back(pattern_map.at(cos_sin_cache));

        auto old_node = root;

        auto new_node = std::make_shared<op::internal::RoPE>(new_args, config);
        new_node->set_friendly_name(old_node->get_friendly_name());
        ov::copy_runtime_info({root->get_input_node_shared_ptr(0), root}, new_node);
        ov::replace_node(old_node, new_node);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(result, matcher_name);
    this->register_matcher(m, callback);
}

ov::pass::RoPEFusionChatGLMHF::RoPEFusionChatGLMHF() {
    using namespace ov::op;
    MATCHER_SCOPE(RoPEFusionChatGLMHF);

    auto qk_linear = pattern::any_input(pattern::shape_matches("[?, 1, ?]"));
    auto cos = pattern::any_input(pattern::shape_matches("[?, 1, 1, ?]"));
    auto sin = pattern::any_input(pattern::shape_matches("[?, 1, 1, ?]"));

    auto reshape = pattern::wrap_type<v1::Reshape>({qk_linear, pattern::any_input()},
                                                   pattern::shape_matches("[?, head_cnt, 1, head_size]"),
                                                   {{"special_zero", false}});
    auto slice_1 = NewGenSlice(reshape, 0, "ndims", 1, 3);

    auto const_idx =
        pattern::wrap_type<ov::opset1::Constant>(pattern::type_matches(ov::element::i32) && const_idx_predicate);
    auto repeat_interleave_cos = pattern::wrap_type<v8::Gather>({cos, const_idx, -1}, {{"batch_dims", 0}});
    auto repeat_interleave_sin = pattern::wrap_type<v8::Gather>({sin, const_idx, -1}, {{"batch_dims", 0}});

    auto multiply = pattern::wrap_type<v1::Multiply>({slice_1, repeat_interleave_cos}, {{"auto_broadcast", "numpy"}});
    auto slice_2 = NewGenSlice(slice_1, 1, INT_MAX, 2, 3);
    auto neg = pattern::wrap_type<v1::Multiply>({slice_2, -1}, {{"auto_broadcast", "numpy"}});
    auto unsqueeze_1 = pattern::wrap_type<v1::Reshape>({neg, pattern::any_input()},
                                                       pattern::shape_matches("[?, head_cnt, 1, ndims/2, 1]"),
                                                       {{"special_zero", false}});
    auto slice_3 = NewGenSlice(slice_1, 0, INT_MAX, 2, 3);
    auto unsqueeze_2 = pattern::wrap_type<v1::Reshape>({slice_3, pattern::any_input()},
                                                       pattern::shape_matches("[?, head_cnt, 1, ndims/2, 1]"),
                                                       {{"special_zero", false}});
    auto stack = pattern::wrap_type<v0::Concat>({unsqueeze_1, unsqueeze_2}, {{"axis", -1}});
    auto flatten = pattern::wrap_type<v1::Reshape>({stack, pattern::any_input()},
                                                   pattern::shape_matches("[?, head_cnt, 1, ndims]"),
                                                   {{"special_zero", true}});
    auto multiply_1 = pattern::wrap_type<v1::Multiply>({flatten, repeat_interleave_sin}, {{"auto_broadcast", "numpy"}});
    auto add = pattern::wrap_type<v1::Add>({multiply, multiply_1}, {{"auto_broadcast", "numpy"}});

    auto slice_5 = NewGenSlice(reshape, "ndims", INT_MAX, 1, 3);
    auto result = pattern::wrap_type<v0::Concat>({add, slice_5}, {{"axis", -1}});

    matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto root = m.get_match_root();

        auto symbols = m.get_symbols();
        auto ndims = symbols["ndims"];
        auto head_cnt = symbols["head_cnt"];
        auto head_size = symbols["head_size"];
        auto half_ndims = symbols["ndims/2"];
        if (!ndims.is_integer() || !head_cnt.is_integer() || !head_size.is_integer() || !half_ndims.is_integer() ||
            half_ndims.i() * 2 != ndims.i()) {
            return false;
        }

        op::internal::RoPE::Config config;
        OutputVector new_args;
        config.rotary_ndims = static_cast<size_t>(ndims.i());
        config.is_chatglm = true;
        config.support_2d_rope = true;
        config.head_cnt = static_cast<size_t>(head_cnt.i());
        config.head_size = static_cast<size_t>(head_size.i());

        new_args.push_back(pattern_map.at(qk_linear));
        new_args.push_back(pattern_map.at(cos));
        new_args.push_back(pattern_map.at(sin));

        auto old_node = root;
        auto new_node = std::make_shared<internal::RoPE>(new_args, config);
        new_node->set_friendly_name(old_node->get_friendly_name());
        ov::copy_runtime_info({root->get_input_node_shared_ptr(0), root}, new_node);
        ov::replace_node(old_node, new_node);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(result, matcher_name);
    this->register_matcher(m, callback);
}

ov::pass::RoPEFusionQwen::RoPEFusionQwen(int split_output_id) {
    using namespace ov::op;
    MATCHER_SCOPE(RoPEFusionQwen);

    // rotary_emb_cos & rotary_emb_sin are sliced by present kv-length (past-kv-length + cur_len)
    auto rotary_emb_cos = pattern::any_input(pattern::shape_matches("[1, ?, 1, ?]"));  // [1,..4096,1,128]
    auto rotary_emb_sin = pattern::any_input(pattern::shape_matches("[1, ?, 1, ?]"));  // [1,..4096,1,128]
    auto qkv_proj = pattern::any_input(pattern::shape_matches("[?, ?, ?]"));           // [?,?,12288]
    auto position_ids = pattern::any_input();

    auto ListUnpack_410_VariadicSplit =
        pattern::wrap_type<v1::VariadicSplit>({qkv_proj, 2, {"head_cnt*head_size", "head_cnt*head_size", "?"}});
    ListUnpack_410_VariadicSplit->set_output_size(3);
    // B,L,H,S
    auto view_Reshape_424 =
        pattern::wrap_type<v1::Reshape>({ListUnpack_410_VariadicSplit->output(split_output_id), pattern::any_input()},
                                        pattern::shape_matches("[?, ?, head_cnt, head_size]"),
                                        {{"special_zero", true}});
    auto slice_Slice_543 = NewGenSlice(view_Reshape_424, 0, "head_size", 1, 3);

    auto ShapeOf_485735 = pattern::wrap_type<ov::op::util::ShapeOfBase>({pattern::any_input()}, {});
    auto Multiply_567524 = pattern::wrap_type<v1::Multiply>({ShapeOf_485735, -1}, {{"auto_broadcast", "numpy"}});
    auto Gather_377635 = pattern::wrap_type<v8::Gather>({Multiply_567524, 1, 0}, {{"batch_dims", 0}});

    auto ShapeOf_409241 = pattern::wrap_type<ov::op::util::ShapeOfBase>({pattern::any_input()}, {});
    auto Gather_311651 = pattern::wrap_type<v8::Gather>({ShapeOf_409241, 1, 0}, {{"batch_dims", 0}});
    auto neg_Multiply = pattern::wrap_type<v1::Multiply>({Gather_311651, -1}, {{"auto_broadcast", "numpy"}});

    auto ScatterUpdate_463814 = pattern::wrap_type<v3::ScatterUpdate>({{0, 0}, 1, Gather_377635 | neg_Multiply, 0});
    auto slice_Slice_446 = pattern::wrap_type<v8::Slice>({rotary_emb_cos, Gather_377635 | neg_Multiply, INT_MAX, 1, 1});

    auto gather_cos_by_pos_ids = pattern::wrap_type<v8::Gather>({rotary_emb_cos, position_ids, 1}, {{"batch_dims", 0}});
    auto reshape_cos_to_expected_layout = pattern::wrap_type<v1::Reshape>({gather_cos_by_pos_ids, pattern::any_input()},
                                                                          pattern::shape_matches("[?, 1, 1, 128]"),
                                                                          {{"special_zero", false}});

    auto slice_StridedSlice_446 = NewGenStridedSlice(rotary_emb_cos, ScatterUpdate_463814, {0, INT_MAX}, {1, 1}, 1);
    auto mul_Multiply_552 = pattern::wrap_type<v1::Multiply>(
        {slice_Slice_543, slice_StridedSlice_446 | slice_Slice_446 | reshape_cos_to_expected_layout},
        {{"auto_broadcast", "numpy"}});

    auto reshape_opt1 = [&](std::shared_ptr<Node> input_BLHS) {
        auto ShapeOf_485814 = pattern::wrap_type<v3::ShapeOf>({input_BLHS}, {});
        auto Gather_377647 = pattern::wrap_type<v8::Gather>({ShapeOf_485814, 1, 0}, {{"batch_dims", 0}});
        // batch-size, we don't care
        auto Gather_377641 =
            pattern::any_input(pattern::type_matches(ov::element::i32) && pattern::shape_matches("[1]"));
        auto ListConstruct_581_Concat =
            pattern::wrap_type<v0::Concat>({Gather_377641, Gather_377647, "head_cnt", 2, "head_size/2"}, {{"axis", 0}});
        auto Gather_391791 = pattern::wrap_type<v8::Gather>({ShapeOf_485814, {0, 1}, 0}, {{"batch_dims", 0}});
        auto ListConstruct_522_Concat = pattern::wrap_type<v0::Concat>({Gather_391791, 32, 2, 64}, {{"axis", 0}});

        auto reshape_Reshape_577 = pattern::wrap_type<v1::Reshape>({input_BLHS, pattern::any_input()},
                                                                   pattern::shape_matches("[?, 2, head_size/2]"),
                                                                   {{"special_zero", true}});
        return pattern::wrap_type<v1::Reshape>(
            {reshape_Reshape_577, ListConstruct_581_Concat | ListConstruct_522_Concat},
            {{"special_zero", false}});
    };

    auto reshape_special = pattern::wrap_type<v1::Reshape>(
        {slice_Slice_543, pattern::any_input()},
        pattern::shape_matches("[..., 0, 2, head_size/2]") || pattern::shape_matches("[..., head_cnt, 2, head_size/2]"),
        {{"special_zero", true}});

    auto ListUnpack_586_Split =
        pattern::wrap_type<v1::Split>({reshape_opt1(slice_Slice_543) | reshape_special, -2}, {{"num_splits", 2}});
    ListUnpack_586_Split->set_output_size(2);
    auto Multiply_567527 =
        pattern::wrap_type<v1::Multiply>({ListUnpack_586_Split->output(1), -1.0f}, {{"auto_broadcast", "numpy"}});
    auto ListUnpack_586_Squeeze_0 = pattern::wrap_type<v0::Squeeze>({Multiply_567527, -2});
    auto ListUnpack_586_Squeeze = pattern::wrap_type<v0::Squeeze>({ListUnpack_586_Split->output(0), -2});

    auto ListUnpack_Squeeze_0_1 = pattern::wrap_type<v1::Reshape>({Multiply_567527, pattern::any_input()},
                                                                  pattern::shape_matches("[?, 1, 32, 64]"),
                                                                  {{"special_zero", false}});
    auto ListUnpack_Squeeze_1 = pattern::wrap_type<v1::Reshape>({ListUnpack_586_Split->output(0), pattern::any_input()},
                                                                pattern::shape_matches("[?, 1, 32, 64]"),
                                                                {{"special_zero", false}});

    auto cat_Concat_593 = pattern::wrap_type<v0::Concat>(
        {ListUnpack_586_Squeeze_0 | ListUnpack_Squeeze_0_1, ListUnpack_586_Squeeze | ListUnpack_Squeeze_1},
        {{"axis", -1}});
    auto slice_StridedSlice_470 = NewGenStridedSlice(rotary_emb_sin, ScatterUpdate_463814, {0, INT_MAX}, {1, 1}, 1);
    auto slice_Slice_470 = pattern::wrap_type<v8::Slice>({rotary_emb_sin, Gather_377635 | neg_Multiply, INT_MAX, 1, 1});
    auto gather_sin_by_pos_ids = pattern::wrap_type<v8::Gather>({rotary_emb_sin, position_ids, 1}, {{"batch_dims", 0}});
    auto reshape_sin_to_expected_layout = pattern::wrap_type<v1::Reshape>({gather_sin_by_pos_ids, pattern::any_input()},
                                                                          pattern::shape_matches("[?, 1, 1, 128]"),
                                                                          {{"special_zero", false}});
    auto mul_Multiply_594 = pattern::wrap_type<v1::Multiply>(
        {cat_Concat_593, slice_StridedSlice_470 | slice_Slice_470 | reshape_sin_to_expected_layout},
        {{"auto_broadcast", "numpy"}});
    auto result = pattern::wrap_type<v1::Add>({mul_Multiply_552, mul_Multiply_594}, {{"auto_broadcast", "numpy"}});

    matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto root = m.get_match_root();

        auto symbols = m.get_symbols();
        auto head_cnt = symbols["head_cnt"];
        auto head_size = symbols["head_size"];
        auto head_size_over_2 = symbols["head_size/2"];
        auto head_cnt_by_head_size = symbols["head_cnt*head_size"];
        if (!head_cnt.is_integer() || !head_size.is_integer() || !head_size_over_2.is_integer() ||
            !head_cnt_by_head_size.is_integer() || head_size_over_2.i() * 2 != head_size.i() ||
            head_cnt.i() * head_size.i() != head_cnt_by_head_size.i()) {
            return false;
        }
        op::internal::RoPE::Config config;
        OutputVector new_args;
        config.is_qwen = true;
        config.head_cnt = static_cast<size_t>(head_cnt.i());
        config.head_size = static_cast<size_t>(head_size.i());
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

        ov::NodeVector rt_from = {pattern_map.at(Multiply_567527).get_node_shared_ptr(),
                                  pattern_map.at(cat_Concat_593).get_node_shared_ptr(),
                                  pattern_map.at(mul_Multiply_594).get_node_shared_ptr(),
                                  pattern_map.at(result).get_node_shared_ptr()};

        if (pattern_map.count(position_ids)) {
            new_args.push_back(pattern_map.at(position_ids));
            config.gather_position_arg_id = 3;
            rt_from.push_back(pattern_map.at(ListUnpack_Squeeze_0_1).get_node_shared_ptr());
            rt_from.push_back(pattern_map.at(ListUnpack_Squeeze_1).get_node_shared_ptr());
        } else {
            rt_from.push_back(pattern_map.at(ListUnpack_586_Squeeze_0).get_node_shared_ptr());
            rt_from.push_back(pattern_map.at(ListUnpack_586_Squeeze).get_node_shared_ptr());
        }
        auto old_node = root;
        auto new_node = std::make_shared<internal::RoPE>(new_args, config);
        new_node->set_friendly_name(old_node->get_friendly_name());
        ov::copy_runtime_info(rt_from, new_node);
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

    std::vector<std::shared_ptr<Node>> inputs = {pattern::any_input(), pattern::any_input()};

    // Broadcast pattern
    auto const_broadcast_axes =
        pattern::wrap_type<op::v0::Constant>(pattern::type_matches(element::u8) && pattern::value_matches("{0}"));
    auto broadcast =
        pattern::wrap_type<op::v1::Broadcast>({"{1.000000f}", inputs[0], const_broadcast_axes}, {{"mode", "numpy"}});

    // Multiply pattern (expand broadcast)
    auto const_inv_freq = pattern::wrap_type<op::v0::Constant>();  // Pattern for the constant inverse frequency
    auto multiply = pattern::wrap_type<op::v1::Multiply>({const_inv_freq, broadcast}, {{"auto_broadcast", "numpy"}});

    // MatMul pattern
    auto matmul =
        pattern::wrap_type<op::v0::MatMul>({multiply, inputs[1]}, {{"transpose_a", false}, {"transpose_b", false}});

    // Transpose pattern
    auto transpose = pattern::wrap_type<op::v1::Transpose>({matmul, {0, 2, 1}});

    // Concat pattern
    auto concat = pattern::wrap_type<op::v0::Concat>({transpose, transpose}, {{"axis", -1}});

    // Cosine and Sine patterns
    auto cos = pattern::wrap_type<op::v0::Cos>({concat});
    auto sin = pattern::wrap_type<op::v0::Sin>({concat});

    // Unsqueeze result pattern (cos or sin)
    auto result = pattern::wrap_type<op::v0::Unsqueeze>({cos | sin, {1}});

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto root = m.get_match_root();

        // Find the current inverse frequency constant
        auto it = pattern_map.find(const_inv_freq);
        if (it == pattern_map.end()) {
            return false;
        }
        auto cur_inv_freq = as_type_ptr<op::v0::Constant>(it->second.get_node_shared_ptr());
        if (!cur_inv_freq) {
            return false;
        }

        // On first match, store the shared inputs and constant
        if (!m_inv_freq) {
            for (size_t i = 0; i < m_shared_inputs.size(); i++) {
                auto it = pattern_map.find(inputs[i]);
                if (it == pattern_map.end())
                    return false;
                m_shared_inputs[i] = it->second.get_node_shared_ptr();
            }
            m_inv_freq = cur_inv_freq;
        }

        // check consts are the same as the one to be shared.
        if (cur_inv_freq->get_element_type() != m_inv_freq->get_element_type())
            return false;
        if (cur_inv_freq->get_shape() != m_inv_freq->get_shape())
            return false;
        auto global_inv_freq = ov::as_type_ptr<op::v0::Constant>(m_inv_freq);

        auto cmp_error =
            memcmp(cur_inv_freq->get_data_ptr(), global_inv_freq->get_data_ptr(), global_inv_freq->get_byte_size());
        if (cmp_error != 0)
            return false;
        // check all inputs are the same as the one to be shared.
        for (size_t i = 0; i < inputs.size(); i++) {
            auto it = pattern_map.find(inputs[i]);
            if (it == pattern_map.end())
                return false;
            if (m_shared_inputs[i] != it->second.get_node_shared_ptr())
                return false;
        }

        // Store the first matched sin/cos node for sharing
        bool is_sin_matched = pattern_map.find(sin) != pattern_map.end();
        if (is_sin_matched && !m_shared_sin0) {
            m_shared_sin0 = root;
            return false;
        }
        if (!is_sin_matched && !m_shared_cos0) {
            m_shared_cos0 = root;
            return false;
        }

        // all inputs & consts are same, we can safely share the subgraph
        // Just for record, the pattern uses cos | sin as root node. This means that we could match both cases.
        // There we use find to decides whether cons or sin is used
        auto replacement = is_sin_matched ? m_shared_sin0 : m_shared_cos0;
        replace_node(root, replacement);
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(result, matcher_name);
    this->register_matcher(m, callback);
}
