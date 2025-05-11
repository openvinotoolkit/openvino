// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "openvino/op/op.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace op {
namespace v8 {

class MatrixNms;

}  // namespace v8
}  // namespace op
}  // namespace ov

namespace ov {
namespace op {
namespace internal {

template <typename BaseNmsOp>
class NmsStaticShapeIE : public BaseNmsOp {
public:
    // TODO: it should be std::string("NmsStaticShapeIE_") + BaseNmsOp::get_type_info_static().name,
    //       but currently it does not pass conversion to Legacy Opset correctly
    OPENVINO_RTTI(BaseNmsOp::get_type_info_static().name, "ie_internal_opset", BaseNmsOp);

    NmsStaticShapeIE() = default;

    using Attributes = typename BaseNmsOp::Attributes;

    /// \brief Constructs a NmsStaticShapeIE operation
    ///
    /// \param boxes Node producing the box coordinates
    /// \param scores Node producing the box scores
    /// \param attrs Attributes of the operation
    NmsStaticShapeIE(const Output<Node>& boxes, const Output<Node>& scores, const Attributes& attrs)
        : BaseNmsOp(boxes, scores, attrs) {
        this->constructor_validate_and_infer_types();
    }
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override {
        return std::make_shared<NmsStaticShapeIE>(new_args.at(0), new_args.at(1), this->m_attrs);
    }

private:
    typedef struct {
    } init_rt_result;

    init_rt_result init_rt_info() {
        BaseNmsOp::get_rt_info()["opset"] = "ie_internal_opset";
        return {};
    }

    init_rt_result init_rt = init_rt_info();
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
            if (this->m_attrs.background_class >= 0 && this->m_attrs.background_class < num_classes) {
                num_classes = std::max(int64_t{1}, num_classes - 1);
            }
            int64_t max_output_boxes_per_class = 0;
            if (this->m_attrs.nms_top_k >= 0)
                max_output_boxes_per_class = std::min(num_boxes, static_cast<int64_t>(this->m_attrs.nms_top_k));
            else
                max_output_boxes_per_class = num_boxes;

            auto max_output_boxes_per_batch = max_output_boxes_per_class * num_classes;
            if (this->m_attrs.keep_top_k >= 0)
                max_output_boxes_per_batch =
                    std::min(max_output_boxes_per_batch, static_cast<int64_t>(this->m_attrs.keep_top_k));

            first_dim_shape = max_output_boxes_per_batch * scores_ps[0].get_length();
        }
    }

    // 'selected_outputs' have the following format:
    //      [number of selected boxes, [class_id, box_score, xmin, ymin, xmax, ymax]]
    this->set_output_type(0, this->get_input_element_type(0), {first_dim_shape, 6});
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

}  // namespace internal
}  // namespace op
}  // namespace ov
