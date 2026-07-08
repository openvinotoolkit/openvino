// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "keep_nms_boundary_precision.hpp"

#include <string>
#include <unordered_set>

#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/non_max_suppression.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/disable_precision_conversion.hpp"

namespace ov::intel_gpu {
namespace {

constexpr char kTargetNmsPrefix[] = "torchvision::nms/";

bool has_target_nms_prefix(const std::shared_ptr<ov::Node>& node) {
    return node && node->get_friendly_name().rfind(kTargetNmsPrefix, 0) == 0;
}

bool is_local_nms_helper(const std::shared_ptr<ov::Node>& node) {
    return ov::is_type<ov::op::v1::Add>(node) || ov::is_type<ov::op::v1::Multiply>(node) ||
           ov::is_type<ov::op::v0::Convert>(node) || ov::is_type<ov::op::v1::ReduceMax>(node) ||
           ov::is_type<ov::op::v1::Reshape>(node) || ov::is_type<ov::op::v0::Unsqueeze>(node);
}

void mark_fp32_chain(const std::shared_ptr<ov::Node>& node,
                     std::unordered_set<const ov::Node*>& visited,
                     size_t remaining_depth) {
    if (!node || !visited.insert(node.get()).second) {
        return;
    }

    ov::disable_conversion(node, ov::element::f16);

    if (remaining_depth == 0) {
        return;
    }

    for (const auto& input : node->inputs()) {
        auto source = input.get_source_output().get_node_shared_ptr();
        if (!source) {
            continue;
        }

        if (ov::is_type<ov::op::v0::Constant>(source)) {
            ov::disable_conversion(source, ov::element::f16);
            continue;
        }

        if (has_target_nms_prefix(source) || is_local_nms_helper(source)) {
            mark_fp32_chain(source, visited, remaining_depth - 1);
        }
    }
}

}  // namespace

KeepNMSBoundaryPrecision::KeepNMSBoundaryPrecision() {
    using namespace ov::pass::pattern;

    auto boxes_offset_add_m = wrap_type<ov::op::v1::Add>();
    auto boxes_reshape_m = wrap_type<ov::op::v1::Reshape>({boxes_offset_add_m, any_input()});
    auto scores_unsqueeze_m = wrap_type<ov::op::v0::Unsqueeze>({any_input(), any_input()});
    auto nms_m = wrap_type<ov::op::v9::NonMaxSuppression>({boxes_reshape_m,
                                                           scores_unsqueeze_m,
                                                           any_input(),
                                                           any_input(),
                                                           any_input()});

    ov::matcher_pass_callback callback = [this, boxes_offset_add_m, boxes_reshape_m, scores_unsqueeze_m, nms_m](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto nms = ov::as_type_ptr<ov::op::v9::NonMaxSuppression>(pattern_map.at(nms_m).get_node_shared_ptr());
        if (!nms || transformation_callback(nms) || !has_target_nms_prefix(nms)) {
            return false;
        }

        auto boxes_reshape = pattern_map.at(boxes_reshape_m).get_node_shared_ptr();
        auto boxes_offset_add = pattern_map.at(boxes_offset_add_m).get_node_shared_ptr();
        auto scores_unsqueeze = pattern_map.at(scores_unsqueeze_m).get_node_shared_ptr();

        std::unordered_set<const ov::Node*> visited;
        mark_fp32_chain(nms, visited, 1);
        mark_fp32_chain(boxes_reshape, visited, 5);
        mark_fp32_chain(boxes_offset_add, visited, 5);
        mark_fp32_chain(scores_unsqueeze, visited, 2);

        for (size_t index = 2; index < nms->get_input_size(); ++index) {
            auto source = nms->input_value(index).get_node_shared_ptr();
            if (source) {
                ov::disable_conversion(source, ov::element::f16);
            }
        }

        return true;
    };

    auto matcher = std::make_shared<Matcher>(nms_m, "KeepNMSBoundaryPrecision");
    register_matcher(matcher, callback);
}

}  // namespace ov::intel_gpu