// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fc_bias_fusion.hpp"
#include <cstdint>
#include <memory>

#include "openvino/op/reshape.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "ov_ops/fully_connected.hpp"
#include "ov_ops/fully_connected_compressed.hpp"
#include "ov_ops/fully_connected_quantized.hpp"
#include "ov_ops/fully_connected_quantized_legacy.hpp"
#include "ov_ops/placeholder.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

#include "transformations/utils/utils.hpp"

#include "itt.hpp"

ov::intel_cpu::FullyConnectedBiasFusion::FullyConnectedBiasFusion() {
    MATCHER_SCOPE(FullyConnectedBiasFusion);
    auto any = ov::pass::pattern::any_input();
    auto input = any;
    auto weights = ov::pass::pattern::any_input(ov::pass::pattern::has_static_shape());
    auto ph = ov::pass::pattern::wrap_type<ov::op::internal::Placeholder>();

    auto has_single_consumer = [](ov::Output<ov::Node> output) {
        return ov::pass::pattern::consumers_count(1)(output);
    };

    auto m_fc =
        ov::pass::pattern::wrap_type<ov::op::internal::FullyConnected>({input, weights, ph}, has_single_consumer);

    auto m_fc_ql = ov::pass::pattern::wrap_type<ov::op::internal::FullyConnectedQuantizedLegacy>(
        {
            input,
            weights,
            ph,
            ov::pass::pattern::any_input(),
            ov::pass::pattern::any_input(),
        },
        has_single_consumer);

    auto m_fc_c = ov::pass::pattern::wrap_type<ov::op::internal::FullyConnectedCompressed>(
        {
            input,
            weights,
            ph,
            ov::pass::pattern::any_input(),
        },
        has_single_consumer);

    auto m_fc_or = std::make_shared<pass::pattern::op::Or>(
        OutputVector{
            m_fc,
            m_fc_ql,
            m_fc_c
        });

    auto m_bias = ov::pass::pattern::any_input(ov::pass::pattern::has_static_shape());
    auto m_add = ov::pass::pattern::wrap_type<ov::op::v1::Add>({m_fc_or, m_bias});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher &m) {
        auto& pattern_to_output = m.get_pattern_value_map();

        auto add = pattern_to_output[m_add].get_node_shared_ptr();
        auto bias = pattern_to_output[m_bias].get_node_shared_ptr();
        auto fc = pattern_to_output.count(m_fc) ? pattern_to_output[m_fc].get_node_shared_ptr()
            : pattern_to_output.count(m_fc_ql) ? pattern_to_output[m_fc_ql].get_node_shared_ptr()
            : pattern_to_output[m_fc_c].get_node_shared_ptr();

        if (transformation_callback(fc)) {
            return false;
        }

        if (!std::dynamic_pointer_cast<ov::op::v0::Constant>(bias)) {
            return false;
        }

        ov::Shape bias_shape(bias->get_shape());
        ov::PartialShape output_shape(fc->get_output_partial_shape(0));
        size_t bias_size = ov::shape_size(bias_shape);
        auto rank = output_shape.rank().get_length();
        if (rank == 0 || output_shape[rank - 1].is_dynamic()) {
            return false;
        }

        if (bias_shape.empty() || static_cast<int64_t>(bias_shape.back()) != output_shape[rank - 1].get_length() || bias_shape.back() != bias_size) {
            return false;
        }

        ov::NodeVector new_ops;

        std::shared_ptr<ov::Node> final_bias = bias;
        if (bias_shape.size() >= 2) {
            auto reshape_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ 1 }, { -1 });
            final_bias = ov::op::util::make_try_fold<ov::op::v1::Reshape>(final_bias, reshape_const, true);
            new_ops.push_back(final_bias);
        }

        std::shared_ptr<ov::Node> fc_with_bias;

        // so we don't need to down cast here
        auto fc_node = std::dynamic_pointer_cast<ov::op::internal::FullyConnected>(fc);
        fc_with_bias = fc_node->fuse_bias(final_bias);

        new_ops.push_back(fc_with_bias);

        fc_with_bias->set_friendly_name(add->get_friendly_name());
        ov::copy_runtime_info({fc, add}, new_ops);
        ov::replace_node(add, fc_with_bias);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(m_add, matcher_name);
    this->register_matcher(m, callback);
}
