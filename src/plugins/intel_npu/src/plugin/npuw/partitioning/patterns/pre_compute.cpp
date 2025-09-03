// Copyright (C) 2024 Intel Corporation
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

namespace {
template <typename T>
std::shared_ptr<ov::Node> makeConst(const ov::element::Type& type,
                                    const ov::Shape& shape,
                                    const std::vector<T>& values) {
    return std::make_shared<ov::op::v0::Constant>(type, shape, values);
}
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
    auto Cos = makeConst(ov::element::f16, ov::Shape({1, max_position_embeddings, rotary_ndims}), lut_cos);
    auto Sin = makeConst(ov::element::f16, ov::Shape({1, max_position_embeddings, rotary_ndims}), lut_sin);

    return {Cos, Sin};
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

ov::npuw::patterns::pre_compute::RopeCacheMatcher::RopeCacheMatcher(const uint32_t max_prompt_len,
                                                                    const std::shared_ptr<ov::Model>& model) {
    auto rpe = std::make_shared<RopePatternLLama2>();

    rpe->transform_cb = [&]() {
        auto inv_freq_size = ov::shape_size(rpe->matched_inv_freq->get_shape());

        LOG_VERB("Making sin-cos cache of size: " << max_prompt_len << "x" << inv_freq_size);

        // shapes  that matches max possible position
        auto cache = makeCosSinCache(max_prompt_len, rpe->matched_inv_freq);

        // Step 1: Define axis (gather along axis 1)
        auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});

        // Step 2: Apply Gather for cos and sin
        auto gather_cos = std::make_shared<ov::op::v8::Gather>(cache[0], rpe->matched_position_ids, axis);
        auto gather_sin = std::make_shared<ov::op::v8::Gather>(cache[1], rpe->matched_position_ids, axis);
        LOG_VERB("Created gather op facilitate LUT search: " << gather_cos->get_name() << ", "
                                                             << gather_cos->get_shape());

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
    };
    rpe->run_on_model(model);
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
    ov::npuw::patterns::pre_compute::RopeCacheMatcher ropeCache(m_max_prompt_len, model);
    return true;
}
