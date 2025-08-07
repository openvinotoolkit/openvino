// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "../../logging.hpp"
#include "pre_compute.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/op/ops.hpp"
#include "ov_ops/rotary_positional_embeddings.hpp"
#include "transformations/common_optimizations/fuse_rotary_positional_embeddings.hpp"
#include "openvino/pass/manager.hpp"

namespace opp = ov::pass::pattern;

namespace {
    uint32_t align_to(uint32_t value, uint32_t alignment) {
        return (value + alignment - 1) & ~(alignment - 1);
    }
    template <typename T>
    std::shared_ptr<ov::Node> makeConst(const ov::element::Type& type, const ov::Shape& shape, const std::vector<T>& values) {
        return std::make_shared<ov::op::v0::Constant>(type, shape, values);
    }
    //TODO: copied from common tests
    static ov::OutputVector makeCosSinCache(size_t max_position_embeddings, std::shared_ptr<ov::Node> inverse_frequencies) {
        const auto inverse_freq_fp32 = ov::as_type_ptr<ov::op::v0::Constant>(inverse_frequencies)->cast_vector<float>();
        size_t rotary_ndims = ov::shape_size(inverse_frequencies->get_shape());

        std::vector<float> lut_sin(max_position_embeddings * inverse_freq_fp32.size(), 0.0f);
        std::vector<float> lut_cos(max_position_embeddings * inverse_freq_fp32.size(), 0.0f);

        // rotate_half style cos/sin table:
        //   y1 = cos(m*xita_i) * x1 - sin(m*xita_i) * x2
        //   y2 = cos(m*xita_i) * x2 + sin(m*xita_i) * x1
        //
        for (size_t i = 0, k = 0; i < rotary_ndims; i += 2, k++) {
            //auto xita_i = 1.0 / std::pow(10000.0, static_cast<double>(i) / rotary_ndims);
            auto xita_i = inverse_freq_fp32[i];
            float* psin = lut_sin.data();
            float* pcos = lut_cos.data();
            for (size_t m = 0; m < max_position_embeddings; m++, psin += rotary_ndims, pcos += rotary_ndims) {
                auto vsin = std::sin(xita_i * m);
                auto vcos = std::cos(xita_i * m);
                pcos[k] = static_cast<float>(vcos);
                pcos[k + rotary_ndims / 2] = pcos[k];

                psin[k] = static_cast<float>(vsin);
                psin[k + rotary_ndims / 2] = psin[k];
            }
        }
        auto Cos = makeConst(ov::element::f32, ov::Shape({1, max_position_embeddings, rotary_ndims}), lut_cos);
        auto Sin = makeConst(ov::element::f32, ov::Shape({1, max_position_embeddings, rotary_ndims}), lut_sin);

        return {Cos, Sin};
    }
}  // namespace

ov::npuw::patterns::pre_compute::RPEPattern::RPEPattern() {
    auto shape_of = opp::wrap_type<ov::op::v3::ShapeOf>({opp::any_input()});
    auto gather = opp::wrap_type<ov::op::v8::Gather>({shape_of, opp::any_input(), opp::any_input()});
    auto concat_1 = opp::wrap_type<ov::op::v0::Concat>({gather, opp::any_input(), opp::any_input()});
    //here we can seen inverse frequencies as a parameter or constant depending on partitioner passes
    auto inv_freq = opp::wrap_type<ov::op::v0::Constant>();
    auto inv_freq_convert = opp::optional<ov::op::v0::Convert>({inv_freq->output(0)});
    auto broadcast = opp::wrap_type<ov::op::v3::Broadcast>({inv_freq_convert, concat_1});
    auto position_ids = opp::wrap_type<ov::op::v0::Parameter>();
    //auto broadcast = opp::wrap_type<ov::op::v3::Broadcast>({opp::wrap_type<ov::op::v0::Parameter>(), concat_1});
    auto unsqueeze = opp::wrap_type<ov::op::v0::Unsqueeze>({position_ids, opp::wrap_type<ov::op::v0::Constant>()});
    auto convert = opp::wrap_type<ov::op::v0::Convert>({unsqueeze});
    auto matmul = opp::wrap_type<ov::op::v0::MatMul>({broadcast, convert});
    auto transpose = opp::wrap_type<ov::op::v1::Transpose>({matmul, opp::any_input()});
    auto concat_2 = opp::wrap_type<ov::op::v0::Concat>({transpose, opp::any_input()});
    auto sin_cos = opp::wrap_type<ov::op::v0::Sin, ov::op::v0::Cos>({concat_2});

    pattern = sin_cos;
    callback_keep = [=](ov::pass::pattern::Matcher& m, RPEPattern & actual) {
        auto& node_to_output = m.get_pattern_value_map();

        actual.matched_position_ids = node_to_output.at(position_ids).get_node_shared_ptr();

        auto matched_concat_2 = node_to_output.at(concat_2).get_node_shared_ptr();
        auto matched_sin_cos = node_to_output.at(sin_cos).get_node_shared_ptr();

        // TODO: maybe check for it's dims
        actual.matched_inv_freq = node_to_output.at(inv_freq).get_node_shared_ptr();

        // locating sin and cos
        auto concat_users = matched_concat_2->output(0).get_target_inputs();
        if (concat_users.size() != 2) {
            LOG_INFO("Rope sin_cos invalid at: "<< matched_concat_2->get_name());
            return false;
        }
        auto concat_it = concat_users.begin();
        auto cos_node = concat_it->get_node();
        concat_it++;
        auto sin_node  = concat_it->get_node();

        if (ov::as_type<ov::op::v0::Sin>(sin_node) == nullptr) {
            std::swap(sin_node, cos_node);
        }
        if (ov::as_type<ov::op::v0::Sin>(sin_node) == nullptr ||
            ov::as_type<ov::op::v0::Cos>(cos_node) == nullptr) {
            LOG_INFO("Rope sin_cos nodes are not valid: sin=" << sin_node->get_name() << ", cos=" << cos_node->get_name());
            return false;
        }
        actual.matched_cos = cos_node->shared_from_this();
        actual.matched_sin = sin_node->shared_from_this();

        return true;
    };
}

ov::npuw::patterns::pre_compute::SinCosLLama2::SinCosLLama2(const uint32_t max_prompt_len) {
    auto rpe = std::make_shared<RPEPattern>();

    auto callback = [=](ov::pass::pattern::Matcher& m) {
        // TODO: how to avoid this call
        if (!rpe->callback(m)) {
            return false;
        }
        auto inv_freq_size = ov::shape_size(rpe->matched_inv_freq->get_shape());

        LOG_INFO("Making sin-cos cache for tensor size: " << max_prompt_len << "x" << inv_freq_size);

        // TODO: take frequencies from node
        // fixed shapes for now that matches max possible position
        auto cache = makeCosSinCache(max_prompt_len, rpe->matched_inv_freq);

        // Step 3: Define axis (gather along axis 1)
        auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1});

        // Step 4: Apply Gather for cos and sin
        auto gather_cos = std::make_shared<ov::op::v8::Gather>(cache[0], rpe->matched_position_ids, axis);
        auto gather_sin = std::make_shared<ov::op::v8::Gather>(cache[1], rpe->matched_position_ids, axis);
        LOG_INFO("Created gather op facilitate LUT search: "<< gather_cos->get_name() << ", " << gather_cos->get_shape());

        // Create the squeeze operation required after gather
        auto squeeze_cos = std::make_shared<ov::op::v0::Squeeze>(gather_cos, axis);
        auto squeeze_sin = std::make_shared<ov::op::v0::Squeeze>(gather_sin, axis);

        LOG_INFO("Created squeeze op to reduce axis=1: "<< squeeze_cos->get_name() << ", " << squeeze_cos->get_shape());

        // making concat with itselfs
        std::vector<std::shared_ptr<ov::Node>> to_concat_cos {squeeze_cos, squeeze_cos};
        std::vector<std::shared_ptr<ov::Node>> to_concat_sin {squeeze_sin, squeeze_sin};

        auto concat_cos = std::make_shared<ov::op::v0::Concat>(to_concat_cos, -1);
        auto concat_sin = std::make_shared<ov::op::v0::Concat>(to_concat_sin, -1);

        LOG_INFO("Rope detected at: "<< rpe->matched_cos->get_name() << ", replacing by cache node: "
            << concat_cos->get_name() << ", " << concat_cos->get_shape());
        LOG_INFO("Rope detected at: "<< rpe->matched_sin->get_name() << ", replacing by cache node: "
            << concat_sin->get_name() << ", " << concat_sin->get_shape());

        // replacing with unsquese folowed sin and cos
        ov::replace_node(rpe->matched_cos->shared_from_this(), concat_cos);
        ov::replace_node(rpe->matched_sin->shared_from_this(), concat_sin);

        return true;  // root changed
    };
    register_matcher(std::make_shared<opp::Matcher>(rpe->pattern, "TagSinCos"), std::move(callback));
}


ov::npuw::patterns::pre_compute::RopeInverseFreq::RopeInverseFreq(
    ov::npuw::patterns::pre_compute::RopeInverseFreq::Results need_freq_consts) {
    auto rpe = std::make_shared<ov::npuw::patterns::pre_compute::RPEPattern>();

    auto callback = [=](ov::pass::pattern::Matcher& m) {
        // TODO: how to avoid this call
        if (!rpe->callback(m)) {
            return false;
        }
        if (auto inverse_freq_constant = ov::as_type_ptr<ov::op::v0::Constant>(rpe->matched_inv_freq)) {
            LOG_INFO("Inverse Frequences Constant found: " << inverse_freq_constant->get_name());
            need_freq_consts.get().push_back(inverse_freq_constant);
            return true;
        }
        return false; //root hasnt changed
    };
    register_matcher(std::make_shared<opp::Matcher>(rpe->pattern, "TagSinCos_InverseFrequencies"), std::move(callback));
}

bool ov::npuw::patterns::pre_compute::RopeCache::run_on_model(const std::shared_ptr<ov::Model>& model) {
    LOG_VERB("RopeCache building=" << m_build_cache);

    if (m_build_cache) {
        ov::pass::GraphRewrite rewr;
        rewr.add_matcher<ov::npuw::patterns::pre_compute::SinCosLLama2>(m_max_prompt_len);
        return rewr.run_on_model(model);
    }
    return true;
}
