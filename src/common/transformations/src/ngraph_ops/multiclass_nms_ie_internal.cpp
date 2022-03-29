// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_ops/multiclass_nms_ie_internal.hpp"

#include "itt.hpp"

using namespace std;
using namespace ngraph;

op::internal::MulticlassNmsIEInternal::MulticlassNmsIEInternal(const Output<Node>& boxes,
                                                               const Output<Node>& scores,
                                                               const ov::op::util::MulticlassNmsBase::Attributes& attrs)
    : opset9::MulticlassNms(boxes, scores, attrs) {
    constructor_validate_and_infer_types();  // FIXME: need?
}

op::internal::MulticlassNmsIEInternal::MulticlassNmsIEInternal(const Output<Node>& boxes,
                                                               const Output<Node>& scores,
                                                               const Output<Node>& roisnum,
                                                               const ov::op::util::MulticlassNmsBase::Attributes& attrs)
    : opset9::MulticlassNms(boxes, scores, roisnum, attrs) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> op::internal::MulticlassNmsIEInternal::clone_with_new_inputs(
    const ngraph::OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(internal_MulticlassNmsIEInternal_clone_with_new_inputs);

    if (new_args.size() == 3) {
        return std::make_shared<MulticlassNms>(new_args.at(0), new_args.at(1), new_args.at(2), m_attrs);
    } else if (new_args.size() == 2) {
        return std::make_shared<MulticlassNms>(new_args.at(0), new_args.at(1), m_attrs);
    }
    throw ngraph::ngraph_error("Unsupported number of inputs: " + std::to_string(new_args.size()));
}

void op::internal::MulticlassNmsIEInternal::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(internal_MulticlassNmsIEInternal_validate_and_infer_types);

    const auto boxes_ps = this->get_input_partial_shape(0);
    const auto scores_ps = this->get_input_partial_shape(1);

    auto first_dim_shape = Dimension::dynamic();

    if (boxes_ps.rank().is_static() && scores_ps.rank().is_static()) {
        const auto num_boxes_boxes = boxes_ps[1];
        if (num_boxes_boxes.is_static() && scores_ps[0].is_static() && scores_ps[1].is_static()) {
            const auto num_boxes = num_boxes_boxes.get_length();
            auto num_classes = scores_ps[1].get_length();
            if (this->m_attrs.background_class >= 0 && this->m_attrs.background_class < num_classes) {
                num_classes = std::max(int64_t{1}, num_classes - 1);
            }
            int64_t max_output_boxes_per_class = 0;
            if (this->m_attrs.nms_top_k >= 0)
                max_output_boxes_per_class = std::min(num_boxes, static_cast<int64_t>(this->m_attrs.nms_top_k));
            else
                max_output_boxes_per_class = num_boxes;

            auto max_output_boxes_per_batch = max_output_boxes_per_class * num_classes;
            if (this->m_keep_top_k >= 0)
                max_output_boxes_per_batch =
                    std::min(max_output_boxes_per_batch, static_cast<int64_t>(this->m_attrs.keep_top_k));

            first_dim_shape = max_output_boxes_per_batch * scores_ps[0].get_length();
        }
    }

    // 'selected_outputs' have the following format:
    //      [number of selected boxes, [class_id, box_score, xmin, ymin, xmax, ymax]]
    this->set_output_type(0, element::f32, {first_dim_shape, 6});
    // 'selected_indices' have the following format:
    //      [number of selected boxes, 1]
    this->set_output_type(1, this->m_attrs.output_type, {first_dim_shape, 1});
    // 'selected_num' have the following format:
    //      [num_batches, ]
    if (boxes_ps.rank().is_static() && boxes_ps.rank().get_length() > 0) {
        this->set_output_type(2, this->m_attrs.output_type, {boxes_ps[0]});
    } else {
        this->set_output_type(2, this->m_attrs.output_type, {Dimension::dynamic()});
    }
}

const ::ngraph::Node::type_info_t& op::internal::MulticlassNmsIEInternal::get_type_info() const {
    return get_type_info_static();
}

const ::ngraph::Node::type_info_t& op::internal::MulticlassNmsIEInternal::get_type_info_static() {
    auto BaseNmsOpTypeInfoPtr = &opset9::MulticlassNms::get_type_info_static();

    static const std::string name = BaseNmsOpTypeInfoPtr->name;

    OPENVINO_SUPPRESS_DEPRECATED_START
    static const ::ngraph::Node::type_info_t type_info_static{name.c_str(),
                                                              BaseNmsOpTypeInfoPtr->version,
                                                              "ie_internal_opset",
                                                              BaseNmsOpTypeInfoPtr};
    OPENVINO_SUPPRESS_DEPRECATED_END
    return type_info_static;
}

#ifndef OPENVINO_STATIC_LIBRARY
const ::ngraph::Node::type_info_t op::internal::MulticlassNmsIEInternal::type_info =
    MulticlassNmsIEInternal::get_type_info_static();
#endif