// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>
#include <algorithm>
#include <string>

#include <transformations_visibility.hpp>

#include "ngraph/op/op.hpp"

namespace ngraph {
namespace op {
namespace internal {

template <typename BaseNmsOp>
class NmsStaticShapeIE : public BaseNmsOp {
public:
    NGRAPH_RTTI_DECLARATION;

    using Attributes = typename BaseNmsOp::Attributes;

    /// \brief Constructs a NmsStaticShapeIE operation
    ///
    /// \param boxes Node producing the box coordinates
    /// \param scores Node producing the box scores
    /// \param attrs Attributes of the operation
    NmsStaticShapeIE(const Output<Node>& boxes,
                     const Output<Node>& scores,
                     const Attributes& attrs) : BaseNmsOp(boxes, scores, attrs) {
        validate_and_infer_types();
    }
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override {
        return std::make_shared<NmsStaticShapeIE>(new_args.at(0), new_args.at(1), this->m_attrs);
    }
};

template <typename BaseNmsOp>
void NmsStaticShapeIE<BaseNmsOp>::validate_and_infer_types() {
    const auto boxes_ps = this->get_input_partial_shape(0);
    const auto scores_ps = this->get_input_partial_shape(1);

    auto first_dim_shape = Dimension::dynamic();

    if (boxes_ps.rank().is_static() && scores_ps.rank().is_static()) {
        const auto num_boxes_boxes = boxes_ps[1];
        if (num_boxes_boxes.is_static() && scores_ps[0].is_static() && scores_ps[1].is_static()) {
            const auto num_boxes = num_boxes_boxes.get_length();
            auto num_classes = scores_ps[1].get_length();
            if (this->m_attrs.background_class >=0 && this->m_attrs.background_class <= num_classes) {
                num_classes = num_classes - 1;
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

template <typename BaseNmsOp>
const ::ngraph::Node::type_info_t& NmsStaticShapeIE<BaseNmsOp>::get_type_info() const { return get_type_info_static(); }

template <typename BaseNmsOp>
const ::ngraph::Node::type_info_t& NmsStaticShapeIE<BaseNmsOp>::get_type_info_static() {
    auto BaseNmsOpTypeInfoPtr = &BaseNmsOp::get_type_info_static();

    // TODO: it should be static const std::string name = std::string("NmsStaticShapeIE_") + BaseNmsOpTypeInfoPtr->name;
    //       but currently it will not pass conversion ot Legacy Opset correctly
    static const std::string name = BaseNmsOpTypeInfoPtr->name;

    static const ::ngraph::Node::type_info_t type_info_static{
        name.c_str(), BaseNmsOpTypeInfoPtr->version, BaseNmsOpTypeInfoPtr};
    return type_info_static;
}

template <typename BaseNmsOp>
const ::ngraph::Node::type_info_t NmsStaticShapeIE<BaseNmsOp>::type_info = NmsStaticShapeIE<BaseNmsOp>::get_type_info_static();

}  // namespace internal
}  // namespace op
}  // namespace ngraph
