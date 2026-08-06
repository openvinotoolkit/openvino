// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_batched_nms_to_multiclass_nms.hpp"

#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "openvino/core/except.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiclass_nms.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/non_max_suppression.hpp"
#include "openvino/op/non_zero.hpp"
#include "openvino/op/one_hot.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/multi_subgraph_base.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/pp.hpp"
#include "ov_ops/multiclass_nms_ie_internal.hpp"
#include "ov_ops/nms_ie_internal.hpp"
#include "transformations/rt_info/disable_precision_conversion.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::intel_gpu {
namespace {

constexpr const char* static_class_count_key = "intel_gpu_batched_nms_static_class_count";
constexpr const char* prefix_limit_key = "intel_gpu_batched_nms_prefix_limit";

bool is_const_one_like(const std::shared_ptr<ov::Node>& node) {
    if (!node) {
        return false;
    }
    const auto constant = ov::util::get_constant_from_source(node->output(0));
    return ov::op::util::has_constant_value<float>(constant, 1.0f);
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
bool get_scalar_from_const_source(const ov::Output<ov::Node>& output, T& value) {
    const auto constant = ov::util::get_constant_from_source(output);
    if (!constant || ov::shape_size(constant->get_shape()) != 1) {
        return false;
    }

    value = constant->cast_vector<T>(1)[0];
    return true;
}

bool is_scalar_constant_value(const ov::Output<ov::Node>& output, int64_t expected) {
    int64_t value = 0;
    return get_scalar_from_const_source(output, value) && value == expected;
}

bool infer_class_count_from_nonzero_indices(const ov::Output<ov::Node>& output, int64_t& class_count) {
    const auto gather = ov::as_type_ptr<ov::op::util::GatherBase>(output.get_node_shared_ptr());
    if (!gather || !is_scalar_constant_value(gather->input_value(1), 1) || !is_scalar_constant_value(gather->input_value(2), 1)) {
        return false;
    }

    const auto transpose = ov::as_type_ptr<ov::op::v1::Transpose>(gather->input_value(0).get_node_shared_ptr());
    const auto non_zero = transpose ? ov::as_type_ptr<ov::op::v3::NonZero>(transpose->input_value(0).get_node_shared_ptr()) : nullptr;
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
        if (!slice || !get_scalar_from_const_source(slice->input_value(1), start) || start != 0 || !get_scalar_from_const_source(slice->input_value(2), stop) ||
            stop <= 0 || !get_scalar_from_const_source(slice->input_value(3), step) || step != 1 ||
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
        OPENVINO_ASSERT(bodies.size() <= static_cast<size_t>(std::numeric_limits<int>::max()), "Unexpected number of subgraph bodies: ", bodies.size());
        for (int body_index = 0; body_index < static_cast<int>(bodies.size()); ++body_index) {
            const auto body_index_us = static_cast<size_t>(body_index);
            const auto& body = bodies[body_index_us];
            const auto& parameters = body->get_parameters();
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
                    const auto& result = body->get_results()[output_desc->m_body_value_index];
                    result->input_value(0).get_node_shared_ptr()->get_rt_info()[prefix_limit_key] = prefix_limit;
                    marked = true;
                }
            }
            marked |= run_on_model(body);
        }
    }
    return marked;
}

ConvertBatchedNmsToMulticlassNms::ConvertBatchedNmsToMulticlassNms() {
    using namespace ov::pass::pattern;
    using ov::pass::operator|;

    auto boxes_source_m = any_input(rank_equals(2));
    auto class_ids_source_m = any_input(rank_equals(1));
    auto reduce_max_m = wrap_type<ov::op::v1::ReduceMax>({boxes_source_m, any_input()});
    auto const_one_m = any_input(is_const_one_like);
    auto max_plus_one_m = wrap_type<ov::op::v1::Add>({reduce_max_m, const_one_m});
    auto class_ids_convert_m = optional<ov::op::v0::Convert>({class_ids_source_m}, is_integral_to_fp_convert);
    auto offsets_multiply_m = wrap_type<ov::op::v1::Multiply>({class_ids_convert_m, max_plus_one_m});

    // match Unsqueeze, or Reshape if optimized after CommonOptimizations
    auto offsets_unsqueeze_m = wrap_type<ov::op::v0::Unsqueeze, ov::op::v1::Reshape>({offsets_multiply_m, any_input()});
    auto boxes_offset_add_m = wrap_type<ov::op::v1::Add>({offsets_unsqueeze_m, boxes_source_m});
    auto boxes_reshape_m = wrap_type<ov::op::v1::Reshape>({boxes_offset_add_m, any_input()});
    auto raw_scores_m = any_input(rank_equals(1));
    auto scores_unsqueeze_m = wrap_type<ov::op::v0::Unsqueeze, ov::op::v1::Reshape>({raw_scores_m, any_input()});

    // match NMSIEInternal for a static model, or op::v9 NMS for a dynamic model.
    auto nms_m = wrap_type<ov::op::v9::NonMaxSuppression, ov::op::internal::NonMaxSuppressionIEInternal>(
        {boxes_reshape_m, scores_unsqueeze_m, any_input(), any_input(), any_input()});
    auto nms_output_m = optional<ov::op::v0::Convert>(nms_m);
    auto gather_indices_m = wrap_type<ov::op::v0::Constant>(value_matches("2"));
    auto gather_axis_m = wrap_type<ov::op::v0::Constant>(value_matches("1"));
    auto gather_m = wrap_type<ov::op::util::GatherBase>({nms_output_m, gather_indices_m, gather_axis_m});
    auto squeeze_axis_m = wrap_type<ov::op::v0::Constant>(value_matches("1"));
    auto squeeze_m = wrap_type<ov::op::v0::Squeeze>({gather_m, squeeze_axis_m}) | wrap_type<ov::op::v1::Reshape>({gather_m, any_input()});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto squeeze = pattern_map.at(squeeze_m).get_node_shared_ptr();
        auto gather = ov::as_type_ptr<ov::op::util::GatherBase>(pattern_map.at(gather_m).get_node_shared_ptr());
        auto nms = pattern_map.at(nms_m).get_node_shared_ptr();
        auto boxes_reshape = ov::as_type_ptr<ov::op::v1::Reshape>(pattern_map.at(boxes_reshape_m).get_node_shared_ptr());
        auto scores_unsqueeze = pattern_map.at(scores_unsqueeze_m).get_node_shared_ptr();

        if (!squeeze || !gather || !nms || !boxes_reshape || !scores_unsqueeze || transformation_callback(squeeze)) {
            return false;
        }

        auto boxes_source = pattern_map.at(boxes_source_m);
        auto class_ids_source = pattern_map.at(class_ids_source_m);
        const auto raw_scores = pattern_map.at(raw_scores_m);

        int64_t max_output_boxes = 0;
        float iou_threshold = 0.0f;
        float score_threshold = 0.0f;
        if (!get_scalar_from_const_source(nms->input_value(2), max_output_boxes) || !get_scalar_from_const_source(nms->input_value(3), iou_threshold) ||
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

        auto one_hot = std::make_shared<ov::op::v1::OneHot>(class_ids_source,
                                                            classes_count,
                                                            ov::op::v0::Constant::create(ov::element::boolean, ov::Shape{}, {true}),
                                                            ov::op::v0::Constant::create(ov::element::boolean, ov::Shape{}, {false}),
                                                            -1);

        auto score_expand_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
        auto scores_2d = std::make_shared<ov::op::v0::Unsqueeze>(scores_f32, score_expand_axis);
        auto masked_score = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {-std::numeric_limits<float>::infinity()});
        auto class_wise_scores_nc = std::make_shared<ov::op::v1::Select>(one_hot, scores_2d, masked_score);
        auto scores_transpose =
            std::make_shared<ov::op::v1::Transpose>(class_wise_scores_nc, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {1, 0}));
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

        auto multiclass_nms = std::make_shared<ov::op::internal::MulticlassNmsIEInternal>(boxes_for_multiclass, class_wise_scores, attrs);
        multiclass_nms->set_friendly_name(nms->get_friendly_name() + "/MulticlassNms");

        auto valid_selected_indices = std::make_shared<ov::op::v8::Slice>(multiclass_nms->output(1),
                                                                          ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0}),
                                                                          multiclass_nms->output(2),
                                                                          ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1}),
                                                                          ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0}));

        auto selected_box_indices = std::make_shared<ov::op::v8::Gather>(valid_selected_indices,
                                                                         ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0}),
                                                                         ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1}));
        selected_box_indices->set_friendly_name(squeeze->get_friendly_name());

        new_ops.insert(new_ops.begin(),
                       {boxes_f32,
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
