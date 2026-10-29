// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/fuse_qsa_indexer.hpp"

#include <algorithm>
#include <cmath>
#include <optional>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/qsa_indexer.hpp"
#include "openvino/op/qsa_prepare.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/rms_norm.hpp"
#include "openvino/op/sparse_attention_indexer.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace {

struct NormPath {
    ov::Output<ov::Node> data;
    ov::Output<ov::Node> weight;
    float epsilon = 0.0f;
    ov::NodeVector nodes;
};

std::optional<float> scalar_constant(const ov::Output<ov::Node>& value) {
    const auto constant = ov::as_type_ptr<ov::op::v0::Constant>(value.get_node_shared_ptr());
    if (!constant || constant->get_shape().size() != 0) {
        return std::nullopt;
    }
    const auto values = constant->cast_vector<float>();
    return values.empty() ? std::nullopt : std::optional<float>(values.front());
}

bool normalizes_last_dimension(const ov::Output<ov::Node>& axes_output) {
    const auto axes = ov::as_type_ptr<ov::op::v0::Constant>(axes_output.get_node_shared_ptr());
    if (!axes) {
        return false;
    }
    const auto axis_values = axes->cast_vector<int64_t>();
    return axis_values.size() == 1 && (axis_values.front() == -1 || axis_values.front() == 3);
}

std::optional<NormPath> match_rms_norm(const ov::Output<ov::Node>& output) {
    if (const auto rms_norm = ov::as_type_ptr<ov::op::internal::RMSNorm>(output.get_node_shared_ptr())) {
        if (rms_norm->get_input_size() < 3 || !normalizes_last_dimension(rms_norm->input_value(1))) {
            return std::nullopt;
        }
        return NormPath{rms_norm->input_value(0),
                        rms_norm->input_value(2),
                        static_cast<float>(rms_norm->get_epsilon()),
                        {rms_norm}};
    }

    const auto multiply = ov::as_type_ptr<ov::op::v1::Multiply>(output.get_node_shared_ptr());
    if (!multiply) {
        return std::nullopt;
    }
    auto divide = ov::as_type_ptr<ov::op::v1::Divide>(multiply->input_value(0).get_node_shared_ptr());
    auto weight = multiply->input_value(1);
    if (!divide) {
        divide = ov::as_type_ptr<ov::op::v1::Divide>(multiply->input_value(1).get_node_shared_ptr());
        weight = multiply->input_value(0);
    }
    const auto sqrt =
        divide ? ov::as_type_ptr<ov::op::v0::Sqrt>(divide->input_value(1).get_node_shared_ptr()) : nullptr;
    const auto add = sqrt ? ov::as_type_ptr<ov::op::v1::Add>(sqrt->input_value(0).get_node_shared_ptr()) : nullptr;
    if (!divide || !sqrt || !add) {
        return std::nullopt;
    }

    auto reduce = ov::as_type_ptr<ov::op::v1::ReduceMean>(add->input_value(0).get_node_shared_ptr());
    auto epsilon = scalar_constant(add->input_value(1));
    if (!reduce || !epsilon) {
        reduce = ov::as_type_ptr<ov::op::v1::ReduceMean>(add->input_value(1).get_node_shared_ptr());
        epsilon = scalar_constant(add->input_value(0));
    }
    if (!reduce || !epsilon) {
        return std::nullopt;
    }
    const auto power = ov::as_type_ptr<ov::op::v1::Power>(reduce->input_value(0).get_node_shared_ptr());
    const auto exponent = power ? scalar_constant(power->input_value(1)) : std::nullopt;
    const auto axes = ov::as_type_ptr<ov::op::v0::Constant>(reduce->input_value(1).get_node_shared_ptr());
    if (!power || !exponent || *exponent != 2.0f || !axes || !reduce->get_keep_dims() ||
        divide->input_value(0) != power->input_value(0)) {
        return std::nullopt;
    }
    const auto axis_values = axes->cast_vector<int64_t>();
    if (axis_values.size() != 1 || (axis_values[0] != -1 && axis_values[0] != 3)) {
        return std::nullopt;
    }

    return NormPath{power->input_value(0), weight, *epsilon, {multiply, divide, sqrt, add, reduce, power}};
}

bool has_single_consumer(const ov::Output<ov::Node>& output) {
    return ov::pass::pattern::consumers_count(1)(output);
}

bool has_single_consumer_each_output(const std::shared_ptr<ov::Node>& node) {
    for (size_t i = 0; i < node->get_output_size(); ++i) {
        if (!has_single_consumer(node->output(i))) {
            return false;
        }
    }
    return true;
}

bool valid_epsilon(float epsilon) {
    return std::isfinite(epsilon) && epsilon >= 1e-6f && epsilon <= 1e-5f;
}

bool has_supported_dimensions(const ov::Output<ov::Node>& query, const ov::Output<ov::Node>& raw_keys) {
    const auto query_shape = query.get_partial_shape();
    const auto keys_shape = raw_keys.get_partial_shape();
    if (!query_shape.rank().is_static() || query_shape.rank().get_length() != 4) {
        return false;
    }
    if (!keys_shape.rank().is_static() || keys_shape.rank().get_length() != 3) {
        return false;
    }
    if (!query_shape[1].is_static() || query_shape[1].get_length() != 4 || !query_shape[3].is_static() ||
        query_shape[3].get_length() != 128) {
        return false;
    }
    if (!keys_shape[2].is_static() || keys_shape[2].get_length() != 128) {
        return false;
    }
    return query_shape[0].compatible(keys_shape[0]) && query_shape[2].compatible(keys_shape[1]);
}

}  // namespace

ov::pass::FuseQSAIndexer::FuseQSAIndexer() {
    MATCHER_SCOPE(FuseQSAIndexer);
    using namespace ov::pass::pattern;

    const auto query_norm_pattern = any_input([](const ov::Output<ov::Node>& output) {
        const auto norm = match_rms_norm(output);
        if (!norm || !valid_epsilon(norm->epsilon) || ov::util::is_empty_constant_tensor(norm->weight)) {
            return false;
        }
        return std::all_of(norm->nodes.begin(), norm->nodes.end(), [](const auto& node) {
            return has_single_consumer(node->output(0));
        });
    });

    const auto prepare_inputs = PatternOps{any_input(),
                                           any_input(),
                                           any_input(),
                                           any_input(),
                                           any_input(),
                                           any_input(),
                                           any_input(),
                                           any_input()};
    const auto prepare_pattern = wrap_type<ov::op::v17::QSAPrepare>(
        prepare_inputs,
        consumers_count(1) &&
            [](const ov::Output<ov::Node>& output) {
                const auto prepare = ov::as_type_ptr<ov::op::v17::QSAPrepare>(output.get_node_shared_ptr());
                OPENVINO_ASSERT(prepare, "FuseQSAIndexer: matched preparation must be QSAPrepare");
                if (!valid_epsilon(prepare->get_epsilon()) ||
                    ov::util::is_empty_constant_tensor(prepare->input_value(5))) {
                    return false;
                }
                return has_single_consumer_each_output(prepare);
            },
        {{"key_block_size", int64_t{4}}});
    const auto indexer_inputs = PatternOps{query_norm_pattern, prepare_pattern, any_input()};
    const auto indexer_pattern = wrap_type<ov::op::v17::SparseAttentionIndexer>(
        indexer_inputs,
        [](const ov::Output<ov::Node>& output) {
            const auto indexer = ov::as_type_ptr<ov::op::v17::SparseAttentionIndexer>(output.get_node_shared_ptr());
            OPENVINO_ASSERT(indexer, "FuseQSAIndexer: matched indexer must be SparseAttentionIndexer");
            if (indexer->get_token_budget() > 512) {
                return false;
            }
            const auto prepare =
                ov::as_type_ptr<ov::op::v17::QSAPrepare>(indexer->input_value(1).get_node_shared_ptr());
            const auto norm = match_rms_norm(indexer->input_value(0));
            return prepare && norm && indexer->input_value(1).get_index() == 0 &&
                   indexer->input_value(2) == prepare->input_value(2) &&
                   has_supported_dimensions(norm->data, prepare->input_value(0));
        },
        {{"key_block_size", int64_t{4}}, {"num_selection_heads", int64_t{1}}});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& matcher) {
        const auto indexer = ov::as_type_ptr<ov::op::v17::SparseAttentionIndexer>(matcher.get_match_root());
        OPENVINO_ASSERT(indexer, "FuseQSAIndexer: matcher root must be SparseAttentionIndexer");
        const auto prepare = ov::as_type_ptr<ov::op::v17::QSAPrepare>(indexer->input_value(1).get_node_shared_ptr());
        OPENVINO_ASSERT(prepare, "FuseQSAIndexer: matched preparation input must be QSAPrepare");
        const auto norm = match_rms_norm(indexer->input_value(0));
        OPENVINO_ASSERT(norm, "FuseQSAIndexer: matched query must have supported RMSNorm provenance");

        ov::op::internal::QSAIndexer::Config config;
        config.key_block_size = prepare->get_key_block_size();
        config.token_budget = indexer->get_token_budget();
        config.num_selection_heads = indexer->get_num_selection_heads();
        config.q_epsilon = norm->epsilon;
        config.k_epsilon = prepare->get_epsilon();
        config.is_causal = true;

        auto fused = std::make_shared<ov::op::internal::QSAIndexer>(ov::OutputVector{norm->data,
                                                                                     prepare->input_value(0),
                                                                                     prepare->input_value(3),
                                                                                     prepare->input_value(4),
                                                                                     prepare->input_value(2),
                                                                                     norm->weight,
                                                                                     prepare->input_value(5),
                                                                                     prepare->input_value(1),
                                                                                     prepare->input_value(6),
                                                                                     prepare->input_value(7)},
                                                                    config);
        fused->set_friendly_name(indexer->get_friendly_name());

        ov::NodeVector nodes{prepare, indexer};
        nodes.insert(nodes.end(), norm->nodes.begin(), norm->nodes.end());
        ov::copy_runtime_info(nodes, fused);

        for (size_t i = 0; i < 2; ++i) {
            indexer->output(i).replace(fused->output(i));
        }
        for (size_t i = 0; i < 3; ++i) {
            prepare->output(i).replace(fused->output(i + 2));
        }
        return true;
    };

    register_matcher(std::make_shared<Matcher>(indexer_pattern, matcher_name), callback);
}
