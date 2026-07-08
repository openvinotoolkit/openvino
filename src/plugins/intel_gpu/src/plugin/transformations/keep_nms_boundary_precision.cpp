// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "keep_nms_boundary_precision.hpp"

#include <memory>
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

bool is_const_one_like(const std::shared_ptr<ov::Node>& node) {
    if (!node) {
        return false;
    }

    if (const auto constant = ov::as_type_ptr<ov::op::v0::Constant>(node)) {
        return constant->cast_vector<float>() == std::vector<float>{1.0f};
    }

    const auto convert = ov::as_type_ptr<ov::op::v0::Convert>(node);
    if (!convert) {
        return false;
    }

    const auto constant = ov::as_type_ptr<ov::op::v0::Constant>(convert->input_value(0).get_node_shared_ptr());
    return constant && constant->cast_vector<float>() == std::vector<float>{1.0f};
}

// The lowered pattern may materialize `1` either as a plain Constant or as a
// Constant followed by Convert (for example when a compressed constant is
// decompressed back to fp32 before it is added to ReduceMax(boxes)).
bool is_integral_to_fp_convert(const std::shared_ptr<ov::Node>& node) {
    const auto convert = ov::as_type_ptr<ov::op::v0::Convert>(node);
    if (!convert) {
        return false;
    }

    const auto input_type = convert->input_value(0).get_element_type();
    const auto output_type = convert->get_output_element_type(0);
    return input_type.is_integral_number() && output_type.is_real();
}

template <typename T>
std::shared_ptr<T> get_node_if(const ov::Output<ov::Node>& output) {
    return ov::as_type_ptr<T>(output.get_node_shared_ptr());
}

// Match the lowered batched-NMS boxes path:
//   boxes_for_nms = Add(boxes, Unsqueeze(offsets))
// The surrounding callback validates that `offsets` is the class-dependent
// coordinate shift used to collapse per-class NMS into a single NMS call.
bool match_batched_nms_offsets(const std::shared_ptr<ov::Node>& boxes_offset_add,
                               std::shared_ptr<ov::Node>& boxes_source,
                               std::shared_ptr<ov::Node>& offsets_unsqueeze) {
    const auto add = ov::as_type_ptr<ov::op::v1::Add>(boxes_offset_add);
    if (!add) {
        return false;
    }

    const auto first = add->input_value(0).get_node_shared_ptr();
    const auto second = add->input_value(1).get_node_shared_ptr();

    if (ov::is_type<ov::op::v0::Unsqueeze>(first)) {
        offsets_unsqueeze = first;
        boxes_source = second;
        return true;
    }

    if (ov::is_type<ov::op::v0::Unsqueeze>(second)) {
        offsets_unsqueeze = second;
        boxes_source = first;
        return true;
    }

    return false;
}

// Validate the full class-aware offsets trick typically lowered from
// batched/multi-class NMS frontends:
//   offsets = Convert(class_ids) * (ReduceMax(boxes) + 1)
//   boxes_for_nms = boxes + Unsqueeze(offsets)
//
// Why this matters:
// - `ReduceMax(boxes) + 1` produces a stride larger than any coordinate in the
//   current boxes tensor.
// - multiplying that stride by `class_ids` moves boxes from different classes
//   into disjoint coordinate ranges.
// - a single NMS call can then behave like per-class NMS, because boxes from
//   different classes no longer overlap after the shift.
//
// Example:
// - assume two boxes overlap in the original image:
//     class 0: [10, 10, 20, 20]
//     class 1: [11, 11, 21, 21]
// - if the maximum coordinate in the tensor is 100, then the stride is 101
// - the class 0 box keeps offset 0 * 101 = 0 and stays [10, 10, 20, 20]
// - the class 1 box gets offset 1 * 101 = 101 and becomes
//     [112, 112, 122, 122]
// The boxes overlapped before the shift, but no longer overlap after it, so a
// single NMS call behaves like class-wise NMS.
// Matching this structure is more stable than relying on a frontend-specific
// friendly name such as `torchvision::nms/...`.
bool matches_batched_nms_chain(const std::shared_ptr<ov::Node>& boxes_offset_add) {
    std::shared_ptr<ov::Node> boxes_source;
    std::shared_ptr<ov::Node> offsets_unsqueeze;
    if (!match_batched_nms_offsets(boxes_offset_add, boxes_source, offsets_unsqueeze)) {
        return false;
    }

    const auto unsqueeze = ov::as_type_ptr<ov::op::v0::Unsqueeze>(offsets_unsqueeze);
    const auto multiply = unsqueeze ? get_node_if<ov::op::v1::Multiply>(unsqueeze->input_value(0)) : nullptr;
    if (!multiply) {
        return false;
    }

    std::shared_ptr<ov::Node> class_ids_convert;
    std::shared_ptr<ov::Node> max_plus_one;
    for (size_t index = 0; index < 2; ++index) {
        auto lhs = multiply->input_value(index).get_node_shared_ptr();
        auto rhs = multiply->input_value(1 - index).get_node_shared_ptr();
        // The offsets trick multiplies two specific ingredients:
        // 1) integer class ids converted to floating point
        // 2) a scalar stride computed as ReduceMax(boxes) + 1
        if (is_integral_to_fp_convert(lhs) && ov::is_type<ov::op::v1::Add>(rhs)) {
            class_ids_convert = lhs;
            max_plus_one = rhs;
            break;
        }
    }

    if (!class_ids_convert || !max_plus_one) {
        return false;
    }

    const auto add = ov::as_type_ptr<ov::op::v1::Add>(max_plus_one);
    if (!add) {
        return false;
    }

    std::shared_ptr<ov::Node> reduce_max;
    for (size_t index = 0; index < 2; ++index) {
        auto lhs = add->input_value(index).get_node_shared_ptr();
        auto rhs = add->input_value(1 - index).get_node_shared_ptr();
        // `ReduceMax(boxes) + 1` is the key part of the coordinate stride: the
        // `+1` guarantees that boxes shifted by neighboring class ids land in
        // non-overlapping coordinate ranges.
        if (ov::is_type<ov::op::v1::ReduceMax>(lhs) && is_const_one_like(rhs)) {
            reduce_max = lhs;
            break;
        }
    }

    if (!reduce_max) {
        return false;
    }

    const auto reduce_max_boxes = reduce_max->input_value(0).get_node_shared_ptr();
    // Tie the stride back to the same `boxes` tensor that is later shifted.
    // This avoids matching an unrelated ReduceMax/Add chain in the neighborhood.
    return reduce_max_boxes == boxes_source;
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

        if (is_local_nms_helper(source)) {
            mark_fp32_chain(source, visited, remaining_depth - 1);
        }
    }
}

}  // namespace

KeepNMSBoundaryPrecision::KeepNMSBoundaryPrecision() {
    using namespace ov::pass::pattern;

    // Coarse candidate: a boxes path feeding NMS through Add -> Reshape and a
    // scores path feeding NMS through Unsqueeze. The callback below performs
    // the stricter batched-NMS offsets validation before marking anything.
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
        if (!nms || transformation_callback(nms)) {
            return false;
        }

        auto boxes_reshape = pattern_map.at(boxes_reshape_m).get_node_shared_ptr();
        auto boxes_offset_add = pattern_map.at(boxes_offset_add_m).get_node_shared_ptr();
        auto scores_unsqueeze = pattern_map.at(scores_unsqueeze_m).get_node_shared_ptr();

        // Reject generic NMS tails. We only want the precision-sensitive
        // lowered batched-NMS/class-aware offsets variant.
        if (!matches_batched_nms_chain(boxes_offset_add)) {
            return false;
        }

        std::unordered_set<const ov::Node*> visited;
        // Keep the local NMS tail in fp32 once the structure is confirmed.
        // We start from the matched NMS inputs and walk only through the small
        // helper subgraph around the offsets computation.
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