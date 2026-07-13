// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_batched_nms_to_multiclass_nms.hpp"

#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/multiclass_nms.hpp"
#include "openvino/op/non_zero.hpp"
#include "openvino/op/non_max_suppression.hpp"
#include "openvino/op/one_hot.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/op/util/multi_subgraph_base.hpp"
#include "ov_ops/multiclass_nms_ie_internal.hpp"
#include "transformations/rt_info/disable_precision_conversion.hpp"

namespace ov::intel_gpu {
namespace {

constexpr const char* static_class_count_key = "intel_gpu_batched_nms_static_class_count";
constexpr const char* prefix_limit_key = "intel_gpu_batched_nms_prefix_limit";

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

bool matches_batched_nms_chain(const std::shared_ptr<ov::Node>& boxes_offset_add,
                               ov::Output<ov::Node>& boxes_source,
                               ov::Output<ov::Node>& class_ids_source) {
    std::shared_ptr<ov::Node> boxes_source_node;
    std::shared_ptr<ov::Node> offsets_unsqueeze;
    if (!match_batched_nms_offsets(boxes_offset_add, boxes_source_node, offsets_unsqueeze)) {
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
        if (ov::is_type<ov::op::v1::ReduceMax>(lhs) && is_const_one_like(rhs)) {
            reduce_max = lhs;
            break;
        }
    }

    if (!reduce_max) {
        return false;
    }

    if (reduce_max->input_value(0).get_node_shared_ptr() != boxes_source_node) {
        return false;
    }

    boxes_source = boxes_source_node;
    class_ids_source = class_ids_convert->input_value(0);
    return true;
}

template <typename T>
bool get_scalar_from_const_source(const ov::Output<ov::Node>& output, T& value) {
    auto node = output.get_node_shared_ptr();
    if (const auto convert = ov::as_type_ptr<ov::op::v0::Convert>(node)) {
        return get_scalar_from_const_source(convert->input_value(0), value);
    }

    const auto constant = ov::as_type_ptr<ov::op::v0::Constant>(node);
    if (!constant) {
        return false;
    }

    const auto values = constant->cast_vector<T>();
    if (values.size() != 1) {
        return false;
    }

    value = values[0];
    return true;
}

bool is_scalar_constant_value(const ov::Output<ov::Node>& output, int64_t expected) {
    int64_t value = 0;
    return get_scalar_from_const_source(output, value) && value == expected;
}

bool infer_class_count_from_nonzero_indices(const ov::Output<ov::Node>& output, int64_t& class_count) {
    const auto gather = get_node_if<ov::op::v8::Gather>(output);
    if (!gather || !is_scalar_constant_value(gather->input_value(1), 1) ||
        !is_scalar_constant_value(gather->input_value(2), 1)) {
        return false;
    }

    const auto transpose = get_node_if<ov::op::v1::Transpose>(gather->input_value(0));
    const auto non_zero = transpose ? get_node_if<ov::op::v3::NonZero>(transpose->input_value(0)) : nullptr;
    if (!non_zero) {
        return false;
    }

    const auto input_shape = non_zero->get_input_partial_shape(0);
    if (input_shape.rank().is_dynamic() || input_shape.rank().get_length() != 2 || input_shape[1].is_dynamic()) {
        return false;
    }

    class_count = input_shape[1].get_length();
    return class_count > 0;
}

bool infer_prefix_limit(const ov::Output<ov::Node>& output, int64_t& prefix_limit) {
    prefix_limit = 0;
    const auto& consumers = output.get_target_inputs();
    if (consumers.empty()) {
        return false;
    }

    for (const auto& consumer : consumers) {
        const auto slice = ov::as_type_ptr<ov::op::v8::Slice>(consumer.get_node()->shared_from_this());
        int64_t start = 0;
        int64_t stop = 0;
        int64_t step = 0;
        int64_t axis = 0;
        if (!slice || !get_scalar_from_const_source(slice->input_value(1), start) || start != 0 ||
            !get_scalar_from_const_source(slice->input_value(2), stop) || stop <= 0 ||
            !get_scalar_from_const_source(slice->input_value(3), step) || step != 1 ||
            !get_scalar_from_const_source(slice->input_value(4), axis) || axis != 0) {
            return false;
        }
        prefix_limit = std::max(prefix_limit, stop);
    }
    return true;
}

}  // namespace

bool MarkBatchedNmsStaticClassCount::run_on_model(const std::shared_ptr<ov::Model>& model) {
    bool marked = false;
    for (const auto& node : model->get_ordered_ops()) {
        const auto subgraph = ov::as_type_ptr<ov::op::util::MultiSubGraphOp>(node);
        if (!subgraph) {
            continue;
        }

        const auto& bodies = subgraph->get_functions();
        for (size_t body_index = 0; body_index < bodies.size(); ++body_index) {
            const auto& parameters = bodies[body_index]->get_parameters();
            for (const auto& input_desc : subgraph->get_input_descriptions(body_index)) {
                int64_t class_count = 0;
                const auto source = subgraph->input(input_desc->m_input_index).get_source_output();
                if (infer_class_count_from_nonzero_indices(source, class_count)) {
                    parameters[input_desc->m_body_parameter_index]->get_rt_info()[static_class_count_key] = class_count;
                    marked = true;
                }
            }

            for (const auto& output_desc : subgraph->get_output_descriptions(body_index)) {
                int64_t prefix_limit = 0;
                if (infer_prefix_limit(subgraph->output(output_desc->m_output_index), prefix_limit)) {
                    const auto& result = bodies[body_index]->get_results()[output_desc->m_body_value_index];
                    result->input_value(0).get_node_shared_ptr()->get_rt_info()[prefix_limit_key] = prefix_limit;
                    marked = true;
                }
            }
            marked |= run_on_model(bodies[body_index]);
        }
    }
    return marked;
}

ConvertBatchedNmsToMulticlassNms::ConvertBatchedNmsToMulticlassNms() {
    using namespace ov::pass::pattern;

    auto boxes_offset_add_m = wrap_type<ov::op::v1::Add>();
    auto boxes_reshape_m = wrap_type<ov::op::v1::Reshape>({boxes_offset_add_m, any_input()});
    auto scores_unsqueeze_m = wrap_type<ov::op::v0::Unsqueeze>({any_input(), any_input()});
    auto nms_m = wrap_type<ov::op::v9::NonMaxSuppression>({boxes_reshape_m,
                                                           scores_unsqueeze_m,
                                                           any_input(),
                                                           any_input(),
                                                           any_input()});
    auto gather_m = wrap_type<ov::op::v8::Gather>({nms_m, any_input(), any_input()});
    auto squeeze_m = wrap_type<ov::op::v0::Squeeze>({gather_m, any_input()});

    ov::matcher_pass_callback callback = [this,
                                          boxes_offset_add_m,
                                          boxes_reshape_m,
                                          scores_unsqueeze_m,
                                          nms_m,
                                          gather_m,
                                          squeeze_m](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto squeeze = ov::as_type_ptr<ov::op::v0::Squeeze>(pattern_map.at(squeeze_m).get_node_shared_ptr());
        auto gather = ov::as_type_ptr<ov::op::v8::Gather>(pattern_map.at(gather_m).get_node_shared_ptr());
        auto nms = ov::as_type_ptr<ov::op::v9::NonMaxSuppression>(pattern_map.at(nms_m).get_node_shared_ptr());
        auto boxes_offset_add = pattern_map.at(boxes_offset_add_m).get_node_shared_ptr();
        auto boxes_reshape = ov::as_type_ptr<ov::op::v1::Reshape>(pattern_map.at(boxes_reshape_m).get_node_shared_ptr());
        auto scores_unsqueeze = ov::as_type_ptr<ov::op::v0::Unsqueeze>(pattern_map.at(scores_unsqueeze_m).get_node_shared_ptr());

        if (!squeeze || !gather || !nms || !boxes_reshape || !scores_unsqueeze || transformation_callback(squeeze)) {
            return false;
        }

        if (!is_scalar_constant_value(gather->input_value(1), 2) ||
            !is_scalar_constant_value(gather->input_value(2), 1) ||
            !is_scalar_constant_value(squeeze->input_value(1), 1)) {
            return false;
        }

        ov::Output<ov::Node> boxes_source;
        ov::Output<ov::Node> class_ids_source;
        if (!matches_batched_nms_chain(boxes_offset_add, boxes_source, class_ids_source)) {
            return false;
        }

        const auto raw_scores = scores_unsqueeze->input_value(0);
        if (boxes_source.get_partial_shape().rank().is_static() && boxes_source.get_partial_shape().rank().get_length() != 2) {
            return false;
        }
        if (raw_scores.get_partial_shape().rank().is_static() && raw_scores.get_partial_shape().rank().get_length() != 1) {
            return false;
        }
        if (class_ids_source.get_partial_shape().rank().is_static() && class_ids_source.get_partial_shape().rank().get_length() != 1) {
            return false;
        }

        int64_t max_output_boxes = 0;
        float iou_threshold = 0.0f;
        float score_threshold = 0.0f;
        if (!get_scalar_from_const_source(nms->input_value(2), max_output_boxes) ||
            !get_scalar_from_const_source(nms->input_value(3), iou_threshold) ||
            !get_scalar_from_const_source(nms->input_value(4), score_threshold)) {
            return false;
        }

        if (max_output_boxes < 0 || max_output_boxes > std::numeric_limits<int>::max()) {
            return false;
        }

        const auto& class_ids_rt_info = class_ids_source.get_node_shared_ptr()->get_rt_info();
        const auto class_count_it = class_ids_rt_info.find(static_class_count_key);
        if (class_count_it == class_ids_rt_info.end()) {
            return false;
        }
        const auto class_count = class_count_it->second.as<int64_t>();

        const auto& squeeze_rt_info = squeeze->get_rt_info();
        const auto prefix_limit_it = squeeze_rt_info.find(prefix_limit_key);
        if (prefix_limit_it == squeeze_rt_info.end()) {
            return false;
        }
        const auto prefix_limit = prefix_limit_it->second.as<int64_t>();
        if (prefix_limit <= 0 || prefix_limit > std::numeric_limits<int>::max()) {
            return false;
        }

        ov::NodeVector new_ops;

        auto boxes_f32 = std::make_shared<ov::op::v0::Convert>(boxes_source, ov::element::f32);
        auto scores_f32 = std::make_shared<ov::op::v0::Convert>(raw_scores, ov::element::f32);
        auto classes_count = ov::op::v0::Constant::create(class_ids_source.get_element_type(), ov::Shape{}, {class_count});

        auto one_hot = std::make_shared<ov::op::v1::OneHot>(
            class_ids_source,
            classes_count,
            ov::op::v0::Constant::create(ov::element::boolean, ov::Shape{}, {true}),
            ov::op::v0::Constant::create(ov::element::boolean, ov::Shape{}, {false}),
            -1);

        auto score_expand_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
        auto scores_2d = std::make_shared<ov::op::v0::Unsqueeze>(scores_f32, score_expand_axis);
        auto masked_score = ov::op::v0::Constant::create(ov::element::f32,
                                                        ov::Shape{},
                                                        {-std::numeric_limits<float>::infinity()});
        auto class_wise_scores_nc = std::make_shared<ov::op::v1::Select>(
            one_hot,
            scores_2d,
            masked_score);
        auto scores_transpose = std::make_shared<ov::op::v1::Transpose>(
            class_wise_scores_nc,
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {1, 0}));
        auto batch_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
        auto class_wise_scores = std::make_shared<ov::op::v0::Unsqueeze>(scores_transpose, batch_axis);
        auto boxes_for_multiclass = std::make_shared<ov::op::v1::Reshape>(boxes_f32, boxes_reshape->input_value(1), false);

        ov::op::util::MulticlassNmsBase::Attributes attrs;
        attrs.sort_result_type = ov::op::util::MulticlassNmsBase::SortResultType::SCORE;
        attrs.sort_result_across_batch = false;
        attrs.output_type = nms->get_output_element_type(0);
        attrs.iou_threshold = iou_threshold;
        attrs.score_threshold = score_threshold;
        attrs.nms_top_k = static_cast<int>(max_output_boxes);
        attrs.keep_top_k = static_cast<int>(prefix_limit);
        attrs.background_class = -1;
        attrs.nms_eta = 1.0f;
        attrs.normalized = true;

        auto multiclass_nms = std::make_shared<ov::op::internal::MulticlassNmsIEInternal>(boxes_for_multiclass,
                                                  class_wise_scores,
                                                  attrs);
        multiclass_nms->set_friendly_name(nms->get_friendly_name() + "/MulticlassNms");

        for (const auto& fp32_node : ov::NodeVector{boxes_f32,
                                                   scores_f32,
                                                   scores_2d,
                                                   masked_score,
                                                   class_wise_scores_nc,
                                                   scores_transpose,
                                                   class_wise_scores,
                                                   boxes_for_multiclass,
                                                   multiclass_nms}) {
            ov::disable_conversion(fp32_node, ov::element::f16);
        }

        auto valid_selected_indices = std::make_shared<ov::op::v8::Slice>(
            multiclass_nms->output(1),
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0}),
            multiclass_nms->output(2),
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1}),
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0}));

        auto selected_box_indices = std::make_shared<ov::op::v8::Gather>(
            valid_selected_indices,
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0}),
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1}));
        selected_box_indices->set_friendly_name(squeeze->get_friendly_name());

        new_ops.insert(new_ops.begin(), {boxes_f32,
                         scores_f32,
                         classes_count,
                                         one_hot,
                                         scores_2d,
                                         masked_score,
                                         class_wise_scores_nc,
                                         scores_transpose,
                                         class_wise_scores,
                                         boxes_for_multiclass,
                                         multiclass_nms,
                                         valid_selected_indices,
                                         selected_box_indices});

        ov::copy_runtime_info(m.get_matched_nodes(), new_ops);
        ov::replace_node(squeeze, selected_box_indices);
        return true;
    };

    auto matcher = std::make_shared<Matcher>(squeeze_m, "ConvertBatchedNmsToMulticlassNms");
    register_matcher(matcher, callback);
}

}  // namespace ov::intel_gpu

