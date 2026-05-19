// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pre_compute.hpp"

#include "../../logging.hpp"
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
static ov::OutputVector makeCosSinCache(const size_t max_position_embeddings,
                                        const std::shared_ptr<ov::Node> inverse_frequencies) {
    const auto inverse_freq_fp32 = ov::as_type_ptr<ov::op::v0::Constant>(inverse_frequencies)->cast_vector<float>();
    const size_t rotary_ndims = ov::shape_size(inverse_frequencies->get_shape()) * 2;

    std::vector<ov::float16> lut_sin(max_position_embeddings * rotary_ndims, 0.0f);
    std::vector<ov::float16> lut_cos(max_position_embeddings * rotary_ndims, 0.0f);

    // rotate_half style cos/sin table:
    //   y1 = cos(m*xita_i) * x1 - sin(m*xita_i) * x2
    //   y2 = cos(m*xita_i) * x2 + sin(m*xita_i) * x1
    //
    for (size_t i = 0, k = 0; i < rotary_ndims; i += 2, k++) {
        auto xita_i = inverse_freq_fp32[i >> 1];
        ov::float16* psin = lut_sin.data();
        ov::float16* pcos = lut_cos.data();
        for (size_t m = 0; m < max_position_embeddings; m++, psin += rotary_ndims, pcos += rotary_ndims) {
            auto vsin = std::sin(xita_i * m);
            auto vcos = std::cos(xita_i * m);
            pcos[k] = ov::float16{vcos};
            pcos[k + rotary_ndims / 2] = pcos[k];

            psin[k] = ov::float16{vsin};
            psin[k + rotary_ndims / 2] = psin[k];
        }
    }

    auto Cos =
        ov::op::v0::Constant::create(ov::element::f16, ov::Shape({1, max_position_embeddings, rotary_ndims}), lut_cos);
    auto Sin =
        ov::op::v0::Constant::create(ov::element::f16, ov::Shape({1, max_position_embeddings, rotary_ndims}), lut_sin);

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

    std::vector<float> freq(factor_size, 0.0f);
    std::vector<float> freq_long(factor_size, 0.0f);
    for (size_t i = 0; i < factor_size; i++) {
        freq[i] = std::pow(short_factor[i] * multiply_const[i], power_const[0]);
        freq_long[i] = std::pow(long_factor[i] * multiply_const[i], power_const[0]);
    }

    auto inv_freq =
        std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape({factor_size}), freq);
    auto inv_freq_long =
        std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape({factor_size}), freq_long);

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
    auto concat_2 = opp::wrap_type<ov::op::v0::Concat>({transpose, opp::any_input()});
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

        LOG_VERB("Rope found : sin=" << matched_sin->get_name() << ", cos=" << matched_cos->get_name());

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
        auto add = opp::wrap_type<ov::op::v1::Add>({red_max, MakeConstant()});
        // max(position_ids) + 1 <= original_max_position_embeddings
        auto leq = opp::wrap_type<ov::op::v1::LessEqual>({add, MakeConstant()});

        auto inv_freq_short_conv = opp::optional<ov::op::v0::Convert>({inv_freq_short->output(0)});
        auto inv_freq_long_conv = opp::optional<ov::op::v0::Convert>({inv_freq_long->output(0)});

        // max(position_ids) + 1 <= original_max_position_embeddings ? short_factor : long_factor;
        auto select = opp::wrap_type<ov::op::v1::Select>({leq, inv_freq_short_conv, inv_freq_long_conv});
        auto unsqueeze = opp::optional<ov::op::v0::Unsqueeze>({select, MakeConstant()});
        auto unsqueeze_1 = opp::optional<ov::op::v0::Unsqueeze>({unsqueeze, MakeConstant()});

        return std::make_tuple(unsqueeze_1, leq, red_max);
    };

    auto position_ids = opp::wrap_type<ov::op::v0::Parameter>();

    auto inv_freq_short = MakeConstant();
    auto inv_freq_long = MakeConstant();

    auto select_cond_max_pos_id = make_select_pattern(position_ids, inv_freq_short, inv_freq_long);
    auto select = std::get<0>(select_cond_max_pos_id);
    auto cond = std::get<1>(select_cond_max_pos_id);
    auto max_pos_id = std::get<2>(select_cond_max_pos_id);

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
        auto add = opp::wrap_type<ov::op::v1::Add>({red_max, MakeConstant()});
        // max(position_ids) + 1 > original_max_position_embeddings
        auto greater = opp::wrap_type<ov::op::v1::Greater>({add, MakeConstant()});

        auto short_factor_conv = opp::optional<ov::op::v0::Convert>({short_factor->output(0)});
        auto long_factor_conv = opp::optional<ov::op::v0::Convert>({long_factor->output(0)});

        // max(position_ids) + 1 > original_max_position_embeddings ? long_factor : short_factor;
        auto select = opp::wrap_type<ov::op::v1::Select>({greater, long_factor_conv, short_factor_conv});
        auto multiply = opp::wrap_type<ov::op::v1::Multiply>({select, multiply_const});
        auto power = opp::wrap_type<ov::op::v1::Power>({multiply, power_const});
        auto unsqueeze = opp::optional<ov::op::v0::Unsqueeze>({power, MakeConstant()});
        auto unsqueeze_1 = opp::optional<ov::op::v0::Unsqueeze>({unsqueeze, MakeConstant()});

        return std::make_tuple(unsqueeze_1, greater, red_max);
    };

    auto position_ids = opp::wrap_type<ov::op::v0::Parameter>();

    auto short_factor = MakeConstant();
    auto long_factor = MakeConstant();

    auto multiply_const = MakeConstant();
    auto power_const = MakeConstant();

    auto select_cond_max_pos_id = make_select_pattern(position_ids, short_factor, long_factor, multiply_const, power_const);
    auto select = std::get<0>(select_cond_max_pos_id);
    auto cond = std::get<1>(select_cond_max_pos_id);
    auto max_pos_id = std::get<2>(select_cond_max_pos_id);

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

ov::npuw::patterns::pre_compute::RopeCacheMatcher::RopeCacheMatcher(const uint32_t max_prompt_len,
                                                                    const std::shared_ptr<ov::Model>& model,
                                                                    const std::string& longrope_input_name) {
    auto rpe = std::make_shared<RopePatternLLama2>();

    rpe->transform_cb = [&]() {
        auto cache = makeCosSinCache(max_prompt_len, rpe->matched_inv_freq);
        replaceSinCosByCache(max_prompt_len, cache, rpe.get());
    };
    rpe->run_on_model(model);

    auto long_rpe = std::make_shared<LongRopePatternPhi>();

    std::shared_ptr<ov::op::v0::Parameter> max_pos_id_param;
    long_rpe->transform_cb = [&]() {
        auto cache_short = makeCosSinCache(max_prompt_len, long_rpe->matched_inv_freq);
        auto cache_long = makeCosSinCache(max_prompt_len, long_rpe->matched_inv_freq_long);

        auto select_cos = std::make_shared<ov::op::v1::Select>(long_rpe->matched_cond, cache_short[0], cache_long[0]);
        auto select_sin = std::make_shared<ov::op::v1::Select>(long_rpe->matched_cond, cache_short[1], cache_long[1]);

        replaceSinCosByCache(max_prompt_len, {select_cos, select_sin}, long_rpe.get());

        auto max_pos_id_out = long_rpe->max_pos_id->output(0);
        max_pos_id_param.reset(new ov::op::v0::Parameter(max_pos_id_out.get_element_type(), {1}));
        max_pos_id_param->set_friendly_name(longrope_input_name);
        max_pos_id_out.replace(max_pos_id_param->output(0));
    };
    long_rpe->run_on_model(model);

    auto long_rpe_v5 = std::make_shared<LongRopePatternPhi_v5>();

    long_rpe_v5->transform_cb = [&]() {
        auto inv_freq = calculate_freq(long_rpe_v5->matched_short_factor,
                                       long_rpe_v5->matched_long_factor,
                                       long_rpe_v5->matched_multiply_const,
                                       long_rpe_v5->matched_power_const);

        auto cache_short = makeCosSinCache(max_prompt_len, inv_freq[0]);
        auto cache_long = makeCosSinCache(max_prompt_len, inv_freq[1]);

        auto select_cos = std::make_shared<ov::op::v1::Select>(long_rpe_v5->matched_cond, cache_long[0], cache_short[0]);
        auto select_sin = std::make_shared<ov::op::v1::Select>(long_rpe_v5->matched_cond, cache_long[1], cache_short[1]);

        // WA: to get correct sin-cos cache size
        long_rpe_v5->matched_inv_freq = inv_freq[0];
        replaceSinCosByCache(max_prompt_len, {select_cos, select_sin}, long_rpe_v5.get());

        auto max_pos_id_out = long_rpe_v5->max_pos_id->output(0);
        max_pos_id_param.reset(new ov::op::v0::Parameter(max_pos_id_out.get_element_type(), {1}));
        max_pos_id_param->set_friendly_name(longrope_input_name);
        max_pos_id_out.replace(max_pos_id_param->output(0));
    };
    long_rpe_v5->run_on_model(model);

    if (max_pos_id_param) {
        model->add_parameters({max_pos_id_param});
        for (auto&& input : model->inputs()) {
            if (input.get_node() == max_pos_id_param.get()) {
                input.set_names({max_pos_id_param->get_friendly_name()});
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
    ov::npuw::patterns::pre_compute::RopeCacheMatcher ropeCache(m_max_prompt_len, model, m_longrope_input_name);
    return true;
}
