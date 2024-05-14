// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/roi_align_rotated.hpp"

#include "element_visitor.hpp"
#include "itt.hpp"
#include "openvino/reference/roi_align.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace v14 {

namespace helpers {
struct Evaluate : element::NoAction<bool> {
    using element::NoAction<bool>::visit;

    template <element::Type_t ET>
    static result_type visit(const Tensor& input,
                             const Tensor& rois,
                             const Tensor& batch_indices,
                             Tensor& out,
                             int pooled_h,
                             int pooled_w,
                             int sampling_ratio,
                             float spatial_scale,
                             bool clockwise_mode) {
        using T = fundamental_type_for<ET>;
        const auto batch_indices_vec_scaled_up = ov::get_tensor_data_as<int64_t>(batch_indices);
        ov::reference::roi_align<T, ov::reference::roi_policy::ROIAlignRotatedOpDefPolicy>(
            input.data<const T>(),
            rois.data<const T>(),
            batch_indices_vec_scaled_up.data(),
            out.data<T>(),
            input.get_shape(),
            rois.get_shape(),
            batch_indices.get_shape(),
            out.get_shape(),
            pooled_h,
            pooled_w,
            sampling_ratio,
            spatial_scale,
            ov::op::v3::ROIAlign::PoolingMode::AVG,
            ov::op::v9::ROIAlign::AlignedMode::ASYMMETRIC,
            clockwise_mode);
        return true;
    }
};
}  // namespace helpers

ROIAlignRotated::ROIAlignRotated(const Output<Node>& input,
                                 const Output<Node>& rois,
                                 const Output<Node>& batch_indices,
                                 const int pooled_h,
                                 const int pooled_w,
                                 const int sampling_ratio,
                                 const float spatial_scale,
                                 const bool clockwise_mode)
    : ROIAlignBase{input, rois, batch_indices, pooled_h, pooled_w, sampling_ratio, spatial_scale},
      m_clockwise_mode{clockwise_mode} {
    // NOTE: Cannot be called in base class, since then ROIAlignRotated
    // is not fully constructed.
    constructor_validate_and_infer_types();
}

void ROIAlignRotated::validate_and_infer_types() {
    OV_OP_SCOPE(v14_ROIAlignRotated_validate_and_infer_types);
    ROIAlignBase::validate_and_infer_types();
}

bool ROIAlignRotated::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v14_ROIAlignRotated_visit_attributes);
    ROIAlignBase::visit_attributes(visitor);
    visitor.on_attribute("clockwise_mode", m_clockwise_mode);

    return true;
}

std::shared_ptr<Node> ROIAlignRotated::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v14_ROIAlignRotated_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<ROIAlignRotated>(new_args.at(0),
                                             new_args.at(1),
                                             new_args.at(2),
                                             get_pooled_h(),
                                             get_pooled_w(),
                                             get_sampling_ratio(),
                                             get_spatial_scale(),
                                             get_clockwise_mode());
}

bool ROIAlignRotated::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OPENVINO_ASSERT(outputs.size() == 1);
    OPENVINO_ASSERT(inputs.size() == 3);

    using namespace ov::element;
    return IF_TYPE_OF_CONVERT_TENSORS(v14_ROIAlignRotated_evaluate,
                                      this,
                                      outputs,
                                      inputs,
                                      OV_PP_ET_LIST(f32, f64),
                                      helpers::Evaluate,
                                      inputs[0].get_element_type(),
                                      inputs[0],
                                      inputs[1],
                                      inputs[2],
                                      outputs[0],
                                      get_pooled_h(),
                                      get_pooled_w(),
                                      get_sampling_ratio(),
                                      get_spatial_scale(),
                                      get_clockwise_mode());
    return true;
}

bool ROIAlignRotated::has_evaluate() const {
    OV_OP_SCOPE(v14_ROIAlignRotated_has_evaluate);
    switch (get_input_element_type(0)) {
    case element::bf16:
    case element::f16:
    case element::f32:
    case element::f64:
        return true;
    default:
        return false;
    }
}
}  // namespace v14
}  // namespace op
}  // namespace ov
