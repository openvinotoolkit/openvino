// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pre_compute.hpp"

#include "../../logging.hpp"
#include "../../orc.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "ov_ops/rotary_positional_embeddings.hpp"
#include "transformations/common_optimizations/fuse_rotary_positional_embeddings.hpp"

namespace opp = ov::pass::pattern;
namespace pre_compute = ov::npuw::patterns::pre_compute;

namespace {
// TODO: copied from common tests
// Writes `rows` rows of a rotate_half-style sin/cos LUT, row i holding the
// coefficients for absolute position i.
// When duplicate=true (LLama2-style rotate_half), the row is inv_freq_size*2 wide and
// its second half mirrors the first (torch.cat([freqs, freqs], dim=-1)). When
// duplicate=false (GPT-style) the row is inv_freq_size wide with no mirroring.
//
// Coefficients are computed in f32 and stored as f16 - the precision the graph-side
// RoPE cache has always used.
static void writeCosSinRows(const std::vector<float>& inverse_freq_fp32,
                            bool duplicate,
                            size_t rows,
                            ov::float16* pcos,
                            ov::float16* psin) {
    const size_t inv_freq_size = inverse_freq_fp32.size();
    const size_t row_width = duplicate ? inv_freq_size * 2 : inv_freq_size;
    std::fill_n(pcos, rows * row_width, ov::float16{0.0f});
    std::fill_n(psin, rows * row_width, ov::float16{0.0f});

    // rotate_half style cos/sin table:
    //   y1 = cos(m*xita_i) * x1 - sin(m*xita_i) * x2
    //   y2 = cos(m*xita_i) * x2 + sin(m*xita_i) * x1
    //
    for (size_t k = 0; k < inv_freq_size; k++) {
        auto xita_i = inverse_freq_fp32[k];
        ov::float16* row_sin = psin;
        ov::float16* row_cos = pcos;
        for (size_t m = 0; m < rows; m++, row_sin += row_width, row_cos += row_width) {
            row_cos[k] = ov::float16{std::cos(xita_i * static_cast<float>(m))};
            row_sin[k] = ov::float16{std::sin(xita_i * static_cast<float>(m))};
            if (duplicate) {
                row_cos[k + inv_freq_size] = row_cos[k];
                row_sin[k + inv_freq_size] = row_sin[k];
            }
        }
    }
}

// Builds a [1, max_position_embeddings, rotary_ndims] cos/sin LUT pair as graph
// Constants. Used by every RoPE flavour except LongRoPE, which gets host-fed model
// inputs instead.
static ov::OutputVector makeCosSinCache(const size_t max_position_embeddings,
                                        const std::shared_ptr<ov::Node> inverse_frequencies,
                                        bool duplicate = true) {
    const auto inverse_freq_fp32 = ov::as_type_ptr<ov::op::v0::Constant>(inverse_frequencies)->cast_vector<float>();
    const size_t rotary_ndims = duplicate ? inverse_freq_fp32.size() * 2 : inverse_freq_fp32.size();

    const ov::Shape table_shape{1, max_position_embeddings, rotary_ndims};
    ov::Tensor lut_cos(ov::element::f16, table_shape);
    ov::Tensor lut_sin(ov::element::f16, table_shape);
    writeCosSinRows(inverse_freq_fp32,
                    duplicate,
                    max_position_embeddings,
                    lut_cos.data<ov::float16>(),
                    lut_sin.data<ov::float16>());

    auto Cos = std::make_shared<ov::op::v0::Constant>(lut_cos);
    auto Sin = std::make_shared<ov::op::v0::Constant>(lut_sin);

    return {Cos, Sin};
}

static ov::NodeVector calculate_freq(const std::shared_ptr<ov::Node> short_factor_node,
                                     const std::shared_ptr<ov::Node> long_factor_node,
                                     const std::shared_ptr<ov::Node> multiply_node,
                                     const std::shared_ptr<ov::Node> power_node) {
    const auto short_factor = ov::as_type_ptr<ov::op::v0::Constant>(short_factor_node)->cast_vector<float>();
    const auto long_factor = ov::as_type_ptr<ov::op::v0::Constant>(long_factor_node)->cast_vector<float>();
    const auto multiply_const = ov::as_type_ptr<ov::op::v0::Constant>(multiply_node)->cast_vector<float>();
    const auto power_const = ov::as_type_ptr<ov::op::v0::Constant>(power_node)->cast_vector<float>();
    auto factor_size = short_factor.size();

    OPENVINO_ASSERT(short_factor.size() == multiply_const.size() && long_factor.size() == multiply_const.size(),
                    "Invalid constants for LongRopePatternPhi_v5, expected same size for short_factor, long_factor "
                    "and multiply_const");
    OPENVINO_ASSERT(power_const.size() == 1,
                    "Invalid constants for LongRopePatternPhi_v5, expected single value for power_const");

    std::vector<float> freq(factor_size, 0.0f);
    std::vector<float> freq_long(factor_size, 0.0f);
    for (size_t i = 0; i < factor_size; i++) {
        freq[i] = std::pow(short_factor[i] * multiply_const[i], power_const[0]);
        freq_long[i] = std::pow(long_factor[i] * multiply_const[i], power_const[0]);
    }

    auto inv_freq = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape({factor_size}), freq);
    auto inv_freq_long = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape({factor_size}), freq_long);

    return {inv_freq, inv_freq_long};
}

void replaceSinCosByCache(int max_prompt_len, const ov::OutputVector& cache, const pre_compute::RopePatternDesc* rpe) {
    auto inv_freq_size = ov::shape_size(rpe->matched_inv_freq->get_shape());

    LOG_VERB("Making sin-cos cache of size: " << max_prompt_len << "x" << inv_freq_size);

    // Step 1: Define axis (gather along axis 1)
    auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});

    // Step 2: Apply Gather for cos and sin
    auto gather_cos = std::make_shared<ov::op::v8::Gather>(cache[0], rpe->matched_position_ids, axis);
    auto gather_sin = std::make_shared<ov::op::v8::Gather>(cache[1], rpe->matched_position_ids, axis);
    LOG_VERB("Created gather op facilitate LUT search: " << gather_cos->get_name() << ", " << gather_cos->get_shape());

    // Step 2: convert fp16->fp32
    auto cos_fp32 = std::make_shared<ov::op::v0::Convert>(gather_cos, ov::element::f32);
    auto sin_fp32 = std::make_shared<ov::op::v0::Convert>(gather_sin, ov::element::f32);

    // Create the squeeze operation required after gather
    auto squeeze_cos = std::make_shared<ov::op::v0::Squeeze>(cos_fp32, axis);
    auto squeeze_sin = std::make_shared<ov::op::v0::Squeeze>(sin_fp32, axis);

    LOG_VERB("Created squeeze_cos op to reduce axis=1: " << squeeze_cos->get_name() << ", "
                                                         << squeeze_cos->get_shape());
    LOG_VERB("Created squeeze_sin op to reduce axis=1: " << squeeze_sin->get_name() << ", "
                                                         << squeeze_sin->get_shape());

    LOG_VERB("Rope cos detected at: " << rpe->matched_cos->get_name() << ", replacing by cache node: "
                                      << gather_cos->get_name() << ", " << gather_cos->get_shape());
    LOG_VERB("Rope sin detected at: " << rpe->matched_sin->get_name() << ", replacing by cache node: "
                                      << gather_sin->get_name() << ", " << gather_sin->get_shape());

    // replacing sin with gather op
    ov::replace_node(rpe->matched_cos, squeeze_cos);
    ov::replace_node(rpe->matched_sin, squeeze_sin);

    // disconnecting gather from rest or subgraph started from concat_1
    auto gather_input_to_concat = rpe->matched_concat->input(0);
    gather_input_to_concat.get_source_output().remove_target_input(gather_input_to_concat);
}

}  // namespace

ov::npuw::patterns::pre_compute::RopePatternLLama2::RopePatternLLama2() : matcher("sin-cos-matcher") {
    auto shape_of = opp::wrap_type<ov::op::v3::ShapeOf>({opp::any_input()});
    auto gather = opp::wrap_type<ov::op::v8::Gather>({shape_of, opp::any_input(), opp::any_input()});
    auto concat_1 = opp::wrap_type<ov::op::v0::Concat>({gather, opp::any_input(), opp::any_input()});
    // here we can seen inverse frequencies as a parameter or constant depending on partitioner passes
    auto inv_freq = opp::wrap_type<ov::op::v0::Constant>();
    auto inv_freq_convert = opp::optional<ov::op::v0::Convert>({inv_freq->output(0)});
    auto broadcast = opp::wrap_type<ov::op::v3::Broadcast>({inv_freq_convert, concat_1});
    auto position_ids = opp::wrap_type<ov::op::v0::Parameter>();
    auto unsqueeze = opp::wrap_type<ov::op::v0::Unsqueeze>({position_ids, opp::wrap_type<ov::op::v0::Constant>()});
    auto convert = opp::wrap_type<ov::op::v0::Convert>({unsqueeze});
    auto matmul = opp::wrap_type<ov::op::v0::MatMul>({broadcast, convert});
    auto transpose = opp::wrap_type<ov::op::v1::Transpose>({matmul, opp::any_input()});
    // Optional Concat between Transpose and Sin/Cos: present in LLama2-style RoPE
    // (torch.cat([freqs, freqs], dim=-1)), absent in GPT-style RoPE (inv_freq shape [1, n, 1]).
    auto concat_2 = opp::optional<ov::op::v0::Concat>({transpose->output(0), opp::any_input()});
    auto output_sin = opp::wrap_type<ov::op::v0::Sin>({concat_2});
    auto output_cos = opp::wrap_type<ov::op::v0::Cos>({concat_2});

    init_cb = [=](const auto& matches) {
        const auto& map_sin = matches.at(output_sin)[0];
        const auto& map_cos = matches.at(output_cos)[0];

        this->matched_position_ids = map_sin.at(position_ids).get_node_shared_ptr();
        this->matched_concat = map_sin.at(concat_1).get_node_shared_ptr();
        this->matched_inv_freq = map_sin.at(inv_freq).get_node_shared_ptr();

        this->matched_cos = map_cos.at(output_cos).get_node_shared_ptr();
        this->matched_sin = map_sin.at(output_sin).get_node_shared_ptr();

        // Determine if freq duplication was applied by inspecting the actual graph:
        // sin's direct input is Concat (LLama2-style) or Transpose (GPT-style, no dup).
        auto sin_input = this->matched_sin->input_value(0).get_node_shared_ptr();
        this->duplicate_freqs = ov::is_type<ov::op::v0::Concat>(sin_input);

        LOG_VERB("Rope found : sin=" << matched_sin->get_name() << ", cos=" << matched_cos->get_name()
                                     << " (duplicate_freqs=" << duplicate_freqs << ")");

        return true;
    };

    matcher.register_patterns({output_sin, output_cos}, make_matcher_callback());
}

ov::npuw::patterns::pre_compute::LongRopePatternPhi::LongRopePatternPhi() : matcher("sin-cos-matcher") {
    auto MakeConstant = []() {
        return opp::wrap_type<ov::op::v0::Constant>();
    };

    auto make_select_pattern = [&](const std::shared_ptr<ov::Node>& position_ids,
                                   const std::shared_ptr<ov::Node>& inv_freq_short,
                                   const std::shared_ptr<ov::Node>& inv_freq_long) {
        auto red_max = opp::wrap_type<ov::op::v1::ReduceMax>({position_ids, MakeConstant()});
        auto cond_offset = MakeConstant();
        auto add = opp::wrap_type<ov::op::v1::Add>({red_max, cond_offset});
        auto context_limit = MakeConstant();
        // max(position_ids) + 1 <= original_max_position_embeddings
        auto leq = opp::wrap_type<ov::op::v1::LessEqual>({add, context_limit});

        auto inv_freq_short_conv = opp::optional<ov::op::v0::Convert>({inv_freq_short->output(0)});
        auto inv_freq_long_conv = opp::optional<ov::op::v0::Convert>({inv_freq_long->output(0)});

        // max(position_ids) + 1 <= original_max_position_embeddings ? short_factor : long_factor;
        auto select = opp::wrap_type<ov::op::v1::Select>({leq, inv_freq_short_conv, inv_freq_long_conv});
        auto unsqueeze = opp::optional<ov::op::v0::Unsqueeze>({select, MakeConstant()});
        auto unsqueeze_1 = opp::optional<ov::op::v0::Unsqueeze>({unsqueeze, MakeConstant()});

        return std::make_tuple(unsqueeze_1, leq, red_max, context_limit, cond_offset);
    };

    auto position_ids = opp::wrap_type<ov::op::v0::Parameter>();

    auto inv_freq_short = MakeConstant();
    auto inv_freq_long = MakeConstant();

    auto select_cond_max_pos_id = make_select_pattern(position_ids, inv_freq_short, inv_freq_long);
    auto select = std::get<0>(select_cond_max_pos_id);
    auto cond = std::get<1>(select_cond_max_pos_id);
    auto max_pos_id = std::get<2>(select_cond_max_pos_id);
    auto context_limit = std::get<3>(select_cond_max_pos_id);
    auto cond_offset = std::get<4>(select_cond_max_pos_id);

    auto shape_of = opp::wrap_type<ov::op::v3::ShapeOf>({opp::any_input()});
    auto gather = opp::wrap_type<ov::op::v8::Gather>({shape_of, opp::any_input(), opp::any_input()});
    auto concat_1 = opp::wrap_type<ov::op::v0::Concat>({gather, opp::any_input(), opp::any_input()});
    // here we can seen inverse frequencies as a parameter or constant depending on partitioner passes
    auto broadcast = opp::wrap_type<ov::op::v3::Broadcast>({select, concat_1});
    auto unsqueeze = opp::wrap_type<ov::op::v0::Unsqueeze>({position_ids, MakeConstant()});
    auto convert = opp::wrap_type<ov::op::v0::Convert>({unsqueeze});
    auto matmul = opp::wrap_type<ov::op::v0::MatMul>({broadcast, convert});
    auto transpose = opp::wrap_type<ov::op::v1::Transpose>({matmul, opp::any_input()});
    auto concat_2 = opp::wrap_type<ov::op::v0::Concat>({transpose, opp::any_input()});
    auto output_sin = opp::wrap_type<ov::op::v0::Sin>({concat_2});
    auto output_cos = opp::wrap_type<ov::op::v0::Cos>({concat_2});

    init_cb = [=](const auto& matches) {
        const auto& map_sin = matches.at(output_sin)[0];
        const auto& map_cos = matches.at(output_cos)[0];

        this->matched_position_ids = map_sin.at(position_ids).get_node_shared_ptr();
        this->matched_concat = map_sin.at(concat_1).get_node_shared_ptr();
        this->matched_inv_freq = map_sin.at(inv_freq_short).get_node_shared_ptr();
        this->matched_inv_freq_long = map_sin.at(inv_freq_long).get_node_shared_ptr();
        this->matched_context_limit = map_sin.at(context_limit).get_node_shared_ptr();
        this->matched_cond_offset = map_sin.at(cond_offset).get_node_shared_ptr();
        this->matched_cond = map_sin.at(cond).get_node_shared_ptr();
        this->max_pos_id = map_sin.at(max_pos_id).get_node_shared_ptr();

        this->matched_cos = map_cos.at(output_cos).get_node_shared_ptr();
        this->matched_sin = map_sin.at(output_sin).get_node_shared_ptr();

        LOG_VERB("Rope found : sin=" << matched_sin->get_name() << ", cos=" << matched_cos->get_name());

        return true;
    };

    matcher.register_patterns({output_sin, output_cos}, make_matcher_callback());
}

ov::npuw::patterns::pre_compute::LongRopePatternPhi_v5::LongRopePatternPhi_v5() : matcher("sin-cos-matcher") {
    auto MakeConstant = []() {
        return opp::wrap_type<ov::op::v0::Constant>();
    };

    auto make_select_pattern = [&](const std::shared_ptr<ov::Node>& position_ids,
                                   const std::shared_ptr<ov::Node>& short_factor,
                                   const std::shared_ptr<ov::Node>& long_factor,
                                   const std::shared_ptr<ov::Node>& multiply_const,
                                   const std::shared_ptr<ov::Node>& power_const) {
        auto red_max = opp::wrap_type<ov::op::v1::ReduceMax>({position_ids, MakeConstant()});
        auto cond_offset = MakeConstant();
        auto add = opp::wrap_type<ov::op::v1::Add>({red_max, cond_offset});
        auto context_limit = MakeConstant();
        // max(position_ids) + 1 > original_max_position_embeddings
        auto greater = opp::wrap_type<ov::op::v1::Greater>({add, context_limit});

        auto short_factor_conv = opp::optional<ov::op::v0::Convert>({short_factor->output(0)});
        auto long_factor_conv = opp::optional<ov::op::v0::Convert>({long_factor->output(0)});

        // max(position_ids) + 1 > original_max_position_embeddings ? long_factor : short_factor;
        auto select = opp::wrap_type<ov::op::v1::Select>({greater, long_factor_conv, short_factor_conv});
        auto multiply = opp::wrap_type<ov::op::v1::Multiply>({select, multiply_const});
        auto power = opp::wrap_type<ov::op::v1::Power>({multiply, power_const});
        auto unsqueeze = opp::optional<ov::op::v0::Unsqueeze>({power, MakeConstant()});
        auto unsqueeze_1 = opp::optional<ov::op::v0::Unsqueeze>({unsqueeze, MakeConstant()});

        return std::make_tuple(unsqueeze_1, greater, red_max, context_limit, cond_offset);
    };

    auto position_ids = opp::wrap_type<ov::op::v0::Parameter>();

    auto short_factor = MakeConstant();
    auto long_factor = MakeConstant();

    auto multiply_const = MakeConstant();
    auto power_const = MakeConstant();

    auto select_cond_max_pos_id =
        make_select_pattern(position_ids, short_factor, long_factor, multiply_const, power_const);
    auto select = std::get<0>(select_cond_max_pos_id);
    auto cond = std::get<1>(select_cond_max_pos_id);
    auto max_pos_id = std::get<2>(select_cond_max_pos_id);
    auto context_limit = std::get<3>(select_cond_max_pos_id);
    auto cond_offset = std::get<4>(select_cond_max_pos_id);

    auto shape_of = opp::wrap_type<ov::op::v3::ShapeOf>({opp::any_input()});
    auto gather = opp::wrap_type<ov::op::v8::Gather>({shape_of, opp::any_input(), opp::any_input()});
    auto concat_1 = opp::wrap_type<ov::op::v0::Concat>({gather, opp::any_input(), opp::any_input()});
    // here we can seen inverse frequencies as a parameter or constant depending on partitioner passes
    auto broadcast = opp::wrap_type<ov::op::v3::Broadcast>({select, concat_1});
    auto unsqueeze = opp::wrap_type<ov::op::v0::Unsqueeze>({position_ids, MakeConstant()});
    auto convert = opp::wrap_type<ov::op::v0::Convert>({unsqueeze});
    auto matmul = opp::wrap_type<ov::op::v0::MatMul>({broadcast, convert});
    auto transpose = opp::wrap_type<ov::op::v1::Transpose>({matmul, opp::any_input()});
    auto concat_2 = opp::wrap_type<ov::op::v0::Concat>({transpose, opp::any_input()});
    auto output_sin = opp::wrap_type<ov::op::v0::Sin>({concat_2});
    auto output_cos = opp::wrap_type<ov::op::v0::Cos>({concat_2});

    init_cb = [=](const auto& matches) {
        const auto& map_sin = matches.at(output_sin)[0];
        const auto& map_cos = matches.at(output_cos)[0];

        this->matched_position_ids = map_sin.at(position_ids).get_node_shared_ptr();
        this->matched_concat = map_sin.at(concat_1).get_node_shared_ptr();
        this->matched_short_factor = map_sin.at(short_factor).get_node_shared_ptr();
        this->matched_long_factor = map_sin.at(long_factor).get_node_shared_ptr();
        this->matched_context_limit = map_sin.at(context_limit).get_node_shared_ptr();
        this->matched_cond_offset = map_sin.at(cond_offset).get_node_shared_ptr();
        this->matched_multiply_const = map_sin.at(multiply_const).get_node_shared_ptr();
        this->matched_power_const = map_sin.at(power_const).get_node_shared_ptr();
        this->matched_cond = map_sin.at(cond).get_node_shared_ptr();
        this->max_pos_id = map_sin.at(max_pos_id).get_node_shared_ptr();

        this->matched_cos = map_cos.at(output_cos).get_node_shared_ptr();
        this->matched_sin = map_sin.at(output_sin).get_node_shared_ptr();

        LOG_VERB("Rope found : sin=" << matched_sin->get_name() << ", cos=" << matched_cos->get_name());

        return true;
    };

    matcher.register_patterns({output_sin, output_cos}, make_matcher_callback());
}

namespace {
// Reads the original_max_position_embeddings constant captured by a LongRoPE pattern.
uint64_t read_context_limit(const std::shared_ptr<ov::Node>& limit_node) {
    auto limit_const = ov::as_type_ptr<ov::op::v0::Constant>(limit_node);
    OPENVINO_ASSERT(limit_const, "Invalid LongRoPE match, expected constant context limit");

    const auto limit_values = limit_const->cast_vector<int64_t>();
    OPENVINO_ASSERT(limit_values.size() == 1, "Invalid LongRoPE context limit, expected a single scalar value");
    OPENVINO_ASSERT(limit_values.front() >= 0, "Invalid LongRoPE context limit, expected a non-negative value");
    return static_cast<uint64_t>(limit_values.front());
}

// The mode switch is re-evaluated on the host as max(position_ids) >= context_limit,
// which only reproduces the graph's `max(position_ids) + offset` comparison for an
// offset of exactly 1. The matchers accept any constant there, so refuse to rewrite a
// model whose offset differs rather than silently binding the wrong coefficients.
void check_cond_offset(const std::shared_ptr<ov::Node>& offset_node) {
    auto offset_const = ov::as_type_ptr<ov::op::v0::Constant>(offset_node);
    OPENVINO_ASSERT(offset_const, "Invalid LongRoPE match, expected a constant position-id offset");

    const auto offset_values = offset_const->cast_vector<int64_t>();
    OPENVINO_ASSERT(offset_values.size() == 1 && offset_values.front() == 1,
                    "Unsupported LongRoPE model: the short/long factor switch is only supported in its "
                    "max(position_ids) + 1 vs original_max_position_embeddings form");
}

template <typename PatternT>
std::optional<uint64_t> extract_context_limit(const std::shared_ptr<ov::Model>& model) {
    auto long_rope = std::make_shared<PatternT>();
    std::optional<uint64_t> context_limit;
    long_rope->transform_cb = [&]() {
        check_cond_offset(long_rope->matched_cond_offset);
        const auto matched_limit = read_context_limit(long_rope->matched_context_limit);
        if (context_limit.has_value()) {
            OPENVINO_ASSERT(context_limit.value() == matched_limit,
                            "Inconsistent LongRoPE context limits detected in the model");
        } else {
            context_limit = matched_limit;
        }
    };
    long_rope->run_on_model(model);
    return context_limit;
}
}  // namespace

std::optional<uint64_t> ov::npuw::patterns::pre_compute::extract_longrope_context_limit(
    const std::shared_ptr<ov::Model>& model) {
    if (auto limit = extract_context_limit<LongRopePatternPhi_v5>(model)) {
        return limit;
    }
    return extract_context_limit<LongRopePatternPhi>(model);
}

ov::npuw::patterns::pre_compute::RopeCacheMatcher::RopeCacheMatcher(const uint32_t max_prompt_len,
                                                                    const std::shared_ptr<ov::Model>& model,
                                                                    LongRopeCosSin* out_tables) {
    auto rpe = std::make_shared<RopePatternLLama2>();

    rpe->transform_cb = [&]() {
        auto cache = makeCosSinCache(max_prompt_len, rpe->matched_inv_freq, rpe->duplicate_freqs);
        replaceSinCosByCache(max_prompt_len, cache, rpe.get());
    };
    rpe->run_on_model(model);

    // LongRoPE cos/sin LUTs are model inputs, not Constants: the host picks the
    // short/long factor mode and binds the matching rows, so no in-graph Select - and
    // hence no npuw_longrope_input scalar - is created for either LongRoPE pattern.
    std::shared_ptr<ov::op::v0::Parameter> lr_cos_param;
    std::shared_ptr<ov::op::v0::Parameter> lr_sin_param;

    // Replaces the matched Sin/Cos with a Gather from two freshly created f16 model
    // inputs, and reports their layout plus both inverse-frequency sets to out_tables.
    auto make_lut_inputs = [&](const pre_compute::RopePatternDesc* rpe_desc,
                               const std::vector<float>& inv_freq_short,
                               const std::vector<float>& inv_freq_long) {
        const size_t rotary_ndims = inv_freq_short.size() * 2;
        OPENVINO_ASSERT(inv_freq_short.size() == inv_freq_long.size(),
                        "Invalid LongRoPE match, expected same size for the short and long factors");

        const ov::Shape lut_shape{1, max_prompt_len, rotary_ndims};
        lr_cos_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, lut_shape);
        lr_sin_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, lut_shape);
        lr_cos_param->set_friendly_name(longrope_cos_input);
        lr_sin_param->set_friendly_name(longrope_sin_input);

        replaceSinCosByCache(max_prompt_len, {lr_cos_param, lr_sin_param}, rpe_desc);

        if (out_tables) {
            out_tables->max_len = max_prompt_len;
            out_tables->rotary_ndims = rotary_ndims;
            out_tables->inv_freq_short = inv_freq_short;
            out_tables->inv_freq_long = inv_freq_long;
        }
    };

    auto long_rpe = std::make_shared<LongRopePatternPhi>();

    long_rpe->transform_cb = [&]() {
        if (lr_cos_param) {
            return;  // one cos/sin cache per model
        }
        check_cond_offset(long_rpe->matched_cond_offset);
        // Unlike v5, here the matched constants already are the inverse frequencies.
        make_lut_inputs(long_rpe.get(),
                        ov::as_type_ptr<ov::op::v0::Constant>(long_rpe->matched_inv_freq)->cast_vector<float>(),
                        ov::as_type_ptr<ov::op::v0::Constant>(long_rpe->matched_inv_freq_long)->cast_vector<float>());
    };
    long_rpe->run_on_model(model);

    auto long_rpe_v5 = std::make_shared<LongRopePatternPhi_v5>();

    long_rpe_v5->transform_cb = [&]() {
        if (lr_cos_param) {
            return;  // one cos/sin cache per model
        }
        check_cond_offset(long_rpe_v5->matched_cond_offset);
        auto inv_freq = calculate_freq(long_rpe_v5->matched_short_factor,
                                       long_rpe_v5->matched_long_factor,
                                       long_rpe_v5->matched_multiply_const,
                                       long_rpe_v5->matched_power_const);

        // WA: to get correct sin-cos cache size
        long_rpe_v5->matched_inv_freq = inv_freq[0];
        make_lut_inputs(long_rpe_v5.get(),
                        ov::as_type_ptr<ov::op::v0::Constant>(inv_freq[0])->cast_vector<float>(),
                        ov::as_type_ptr<ov::op::v0::Constant>(inv_freq[1])->cast_vector<float>());
    };
    long_rpe_v5->run_on_model(model);

    if (lr_cos_param) {
        model->add_parameters({lr_cos_param, lr_sin_param});
        for (auto&& input : model->inputs()) {
            if (input.get_node() == lr_cos_param.get()) {
                input.set_names({lr_cos_param->get_friendly_name()});
            } else if (input.get_node() == lr_sin_param.get()) {
                input.set_names({lr_sin_param->get_friendly_name()});
            }
        }
    }
    model->validate_nodes_and_infer_types();
}

ov::npuw::patterns::pre_compute::RopeInverseFreq::RopeInverseFreq(
    ov::npuw::patterns::pre_compute::RopeInverseFreq::Results need_freq_consts,
    const std::shared_ptr<ov::Model>& model) {
    auto rpe = std::make_shared<ov::npuw::patterns::pre_compute::RopePatternLLama2>();

    rpe->transform_cb = [&]() {
        if (auto inverse_freq_constant = ov::as_type_ptr<ov::op::v0::Constant>(rpe->matched_inv_freq)) {
            LOG_VERB("Inverse Frequences Constant found: " << inverse_freq_constant->get_name());
            need_freq_consts.get().push_back(inverse_freq_constant);
            return true;
        }
        return false;  // root hasnt changed
    };
    rpe->run_on_model(model);
}

bool ov::npuw::patterns::pre_compute::RopeCache::run_on_model(const std::shared_ptr<ov::Model>& model) {
    ov::npuw::patterns::pre_compute::RopeCacheMatcher ropeCache(m_max_prompt_len, model, &m_host_tables);
    return true;
}

void ov::npuw::patterns::pre_compute::LongRopeCosSin::rebuild_tables() {
    cos = {};
    sin = {};
    if (max_len == 0 || rotary_ndims == 0 || inv_freq_short.empty()) {
        return;
    }
    const size_t modes = has_long ? 2u : 1u;
    const ov::Shape shape{1, modes * max_len, rotary_ndims};
    cos = ov::Tensor(ov::element::f16, shape);
    sin = ov::Tensor(ov::element::f16, shape);

    writeCosSinRows(inv_freq_short, true, max_len, cos.data<ov::float16>(), sin.data<ov::float16>());
    if (has_long) {
        writeCosSinRows(inv_freq_long,
                        true,
                        max_len,
                        cos.data<ov::float16>() + max_len * rotary_ndims,
                        sin.data<ov::float16>() + max_len * rotary_ndims);
    }
}

namespace {
ov::Tensor longrope_rows(ov::Tensor& table,
                         size_t max_len,
                         size_t rotary_ndims,
                         bool has_long,
                         size_t lut_len,
                         bool is_long) {
    OPENVINO_ASSERT(table && lut_len <= max_len, "LongRoPE LUT does not cover the requested rows");
    const size_t row_offset = (is_long && has_long) ? max_len : 0u;
    auto* rows = table.data<ov::float16>() + row_offset * rotary_ndims;
    return ov::Tensor(ov::element::f16, ov::Shape{1, lut_len, rotary_ndims}, rows);
}
}  // namespace

ov::Tensor ov::npuw::patterns::pre_compute::LongRopeCosSin::cos_rows(size_t lut_len, bool is_long) {
    return longrope_rows(cos, max_len, rotary_ndims, has_long, lut_len, is_long);
}

ov::Tensor ov::npuw::patterns::pre_compute::LongRopeCosSin::sin_rows(size_t lut_len, bool is_long) {
    return longrope_rows(sin, max_len, rotary_ndims, has_long, lut_len, is_long);
}

void ov::npuw::patterns::pre_compute::LongRopeCosSin::serialize(ov::npuw::orc::Stream& stream) {
    stream & max_len & rotary_ndims & has_long & inv_freq_short & inv_freq_long;
    if (stream.input()) {
        // Deserialization imports already-compiled child models, so RopeCache never runs
        // again - rebuild the tables here, otherwise the imported graph's npuw_lr_cos/sin
        // inputs would have nothing to bind to.
        rebuild_tables();
    }
}
