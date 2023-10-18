// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/roi_align.hpp"

#include "itt.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/roi_align.hpp"
#include "roi_align_shape_inference.hpp"

using namespace std;

namespace ov {
op::v3::ROIAlign::ROIAlign(const Output<Node>& input,
                           const Output<Node>& rois,
                           const Output<Node>& batch_indices,
                           const int pooled_h,
                           const int pooled_w,
                           const int sampling_ratio,
                           const float spatial_scale,
                           const string& mode)
    : Op{{input, rois, batch_indices}},
      m_pooled_h{pooled_h},
      m_pooled_w{pooled_w},
      m_sampling_ratio{sampling_ratio},
      m_spatial_scale{spatial_scale},
      m_mode{EnumNames<ROIAlign::PoolingMode>::as_enum(mode)} {
    constructor_validate_and_infer_types();
}

op::v3::ROIAlign::ROIAlign(const Output<Node>& input,
                           const Output<Node>& rois,
                           const Output<Node>& batch_indices,
                           const int pooled_h,
                           const int pooled_w,
                           const int sampling_ratio,
                           const float spatial_scale,
                           const PoolingMode mode)
    : Op{{input, rois, batch_indices}},
      m_pooled_h{pooled_h},
      m_pooled_w{pooled_w},
      m_sampling_ratio{sampling_ratio},
      m_spatial_scale{spatial_scale},
      m_mode{mode} {
    constructor_validate_and_infer_types();
}

void op::v3::ROIAlign::validate_and_infer_types() {
    OV_OP_SCOPE(v3_ROIAlign_validate_and_infer_types);

    const auto out_et = roi_align::validate::data_and_roi_et(this);
    roi_align::validate::batch_indicies_et(this);

    OPENVINO_SUPPRESS_DEPRECATED_START
    const auto input_shapes = get_node_input_partial_shapes(*this);
    OPENVINO_SUPPRESS_DEPRECATED_END

    auto output_shape = shape_infer(this, input_shapes).front();
    set_output_type(0, out_et, output_shape);

    const auto& input_ps = input_shapes.front();

    // if the channels dimension is not known
    // the first input should be used during the function specialization
    if (input_ps.rank().is_static() && input_ps[1].is_dynamic()) {
        set_input_is_relevant_to_shape(0);
    }
    // if the 'NUM_ROIS' value is not known
    // the last 2 inputs should be used during the function specialization
    if (output_shape[0].is_dynamic()) {
        set_input_is_relevant_to_shape(1);
        set_input_is_relevant_to_shape(2);
    }
}

bool op::v3::ROIAlign::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v3_ROIAlign_visit_attributes);
    visitor.on_attribute("pooled_h", m_pooled_h);
    visitor.on_attribute("pooled_w", m_pooled_w);
    visitor.on_attribute("sampling_ratio", m_sampling_ratio);
    visitor.on_attribute("spatial_scale", m_spatial_scale);
    visitor.on_attribute("mode", m_mode);

    return true;
}

shared_ptr<Node> op::v3::ROIAlign::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v3_ROIAlign_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<ROIAlign>(new_args.at(0),
                                 new_args.at(1),
                                 new_args.at(2),
                                 m_pooled_h,
                                 m_pooled_w,
                                 m_sampling_ratio,
                                 m_spatial_scale,
                                 m_mode);
}

// ------------------------------ V9 ------------------------------

op::v9::ROIAlign::ROIAlign(const Output<Node>& input,
                           const Output<Node>& rois,
                           const Output<Node>& batch_indices,
                           const int pooled_h,
                           const int pooled_w,
                           const int sampling_ratio,
                           const float spatial_scale,
                           const PoolingMode mode,
                           const AlignedMode aligned_mode)
    : Op{{input, rois, batch_indices}},
      m_pooled_h{pooled_h},
      m_pooled_w{pooled_w},
      m_sampling_ratio{sampling_ratio},
      m_spatial_scale{spatial_scale},
      m_mode{mode},
      m_aligned_mode{aligned_mode} {
    constructor_validate_and_infer_types();
}

void op::v9::ROIAlign::validate_and_infer_types() {
    OV_OP_SCOPE(v9_ROIAlign_validate_and_infer_types);

    const auto out_et = roi_align::validate::data_and_roi_et(this);
    roi_align::validate::batch_indicies_et(this);

    OPENVINO_SUPPRESS_DEPRECATED_START
    const auto input_shapes = get_node_input_partial_shapes(*this);
    OPENVINO_SUPPRESS_DEPRECATED_END

    auto output_shape = shape_infer(this, input_shapes).front();
    set_output_type(0, out_et, output_shape);

    const auto& input_ps = input_shapes.front();

    // if the channels dimension is not known
    // the first input should be used during the function specialization
    if (input_ps.rank().is_static() && input_ps[1].is_dynamic()) {
        set_input_is_relevant_to_shape(0);
    }
    // if the 'NUM_ROIS' value is not known
    // the last 2 inputs should be used during the function specialization
    if (output_shape[0].is_dynamic()) {
        set_input_is_relevant_to_shape(1);
        set_input_is_relevant_to_shape(2);
    }
}

bool op::v9::ROIAlign::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v9_ROIAlign_visit_attributes);
    visitor.on_attribute("pooled_h", m_pooled_h);
    visitor.on_attribute("pooled_w", m_pooled_w);
    visitor.on_attribute("sampling_ratio", m_sampling_ratio);
    visitor.on_attribute("spatial_scale", m_spatial_scale);
    visitor.on_attribute("mode", m_mode);
    visitor.on_attribute("aligned_mode", m_aligned_mode);

    return true;
}

shared_ptr<Node> op::v9::ROIAlign::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v9_ROIAlign_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<ROIAlign>(new_args.at(0),
                                 new_args.at(1),
                                 new_args.at(2),
                                 m_pooled_h,
                                 m_pooled_w,
                                 m_sampling_ratio,
                                 m_spatial_scale,
                                 m_mode,
                                 m_aligned_mode);
}

template <>
OPENVINO_API EnumNames<op::v3::ROIAlign::PoolingMode>& EnumNames<op::v3::ROIAlign::PoolingMode>::get() {
    static auto enum_names = EnumNames<op::v3::ROIAlign::PoolingMode>(
        "op::v3::ROIAlign::PoolingMode",
        {{"avg", op::v3::ROIAlign::PoolingMode::AVG}, {"max", op::v3::ROIAlign::PoolingMode::MAX}});
    return enum_names;
}

template <>
OPENVINO_API EnumNames<op::v9::ROIAlign::PoolingMode>& EnumNames<op::v9::ROIAlign::PoolingMode>::get() {
    static auto enum_names = EnumNames<op::v9::ROIAlign::PoolingMode>(
        "op::v9::ROIAlign::PoolingMode",
        {{"avg", op::v9::ROIAlign::PoolingMode::AVG}, {"max", op::v9::ROIAlign::PoolingMode::MAX}});
    return enum_names;
}

template <>
OPENVINO_API EnumNames<op::v9::ROIAlign::AlignedMode>& EnumNames<op::v9::ROIAlign::AlignedMode>::get() {
    static auto enum_names = EnumNames<op::v9::ROIAlign::AlignedMode>(
        "op::v9::ROIAlign::AlignedMode",
        {{"asymmetric", op::v9::ROIAlign::AlignedMode::ASYMMETRIC},
         {"half_pixel_for_nn", op::v9::ROIAlign::AlignedMode::HALF_PIXEL_FOR_NN},
         {"half_pixel", op::v9::ROIAlign::AlignedMode::HALF_PIXEL}});
    return enum_names;
}

std::ostream& operator<<(std::ostream& s, const op::v3::ROIAlign::PoolingMode& type) {
    return s << as_string(type);
}

std::ostream& operator<<(std::ostream& s, const op::v9::ROIAlign::PoolingMode& type) {
    return s << as_string(type);
}

std::ostream& operator<<(std::ostream& s, const op::v9::ROIAlign::AlignedMode& type) {
    return s << as_string(type);
}

namespace op {
namespace roi_align {
namespace {

template <element::Type_t ET>
bool evaluate(const Tensor& feature_maps,
              const Tensor& rois,
              const std::vector<int64_t>& batch_indices_vec_scaled_up,
              const Tensor& out,
              const int pooled_height,
              const int pooled_width,
              const int sampling_ratio,
              const float spatial_scale,
              const v3::ROIAlign::PoolingMode& pooling_mode,
              const Shape& batch_indices_shape,
              const v9::ROIAlign::AlignedMode& aligned_mode = v9::ROIAlign::AlignedMode::ASYMMETRIC) {
    using T = typename element_type_traits<ET>::value_type;
    ov::reference::roi_align<T>(feature_maps.data<T>(),
                                rois.data<T>(),
                                batch_indices_vec_scaled_up.data(),
                                out.data<T>(),
                                feature_maps.get_shape(),
                                rois.get_shape(),
                                batch_indices_shape,
                                out.get_shape(),
                                pooled_height,
                                pooled_width,
                                sampling_ratio,
                                spatial_scale,
                                pooling_mode,
                                aligned_mode);
    return true;
}

bool evaluate(const TensorVector& args,
              const Tensor& out,
              const int pooled_height,
              const int pooled_width,
              const int sampling_ratio,
              const float spatial_scale,
              const v3::ROIAlign::PoolingMode& pooling_mode,
              const v9::ROIAlign::AlignedMode& aligned_mode = v9::ROIAlign::AlignedMode::ASYMMETRIC) {
    const auto& feature_maps = args[0];
    const auto& rois = args[1];
    const auto& batch_indices = args[2];
    const auto batch_indices_vec_scaled_up = get_tensor_data_as<int64_t>(batch_indices, ov::util::Cast<int64_t>());

    bool rc;
    switch (feature_maps.get_element_type()) {
        OPENVINO_TYPE_CASE(evaluate_roi_align,
                           bf16,
                           feature_maps,
                           rois,
                           batch_indices_vec_scaled_up,
                           out,
                           pooled_height,
                           pooled_width,
                           sampling_ratio,
                           spatial_scale,
                           pooling_mode,
                           batch_indices.get_shape(),
                           aligned_mode);
        OPENVINO_TYPE_CASE(evaluate_roi_align,
                           f16,
                           feature_maps,
                           rois,
                           batch_indices_vec_scaled_up,
                           out,
                           pooled_height,
                           pooled_width,
                           sampling_ratio,
                           spatial_scale,
                           pooling_mode,
                           batch_indices.get_shape(),
                           aligned_mode);
        OPENVINO_TYPE_CASE(evaluate_roi_align,
                           f32,
                           feature_maps,
                           rois,
                           batch_indices_vec_scaled_up,
                           out,
                           pooled_height,
                           pooled_width,
                           sampling_ratio,
                           spatial_scale,
                           pooling_mode,
                           batch_indices.get_shape(),
                           aligned_mode);
    default:
        rc = false;
        break;
    }

    return rc;
}
}  // namespace
}  // namespace roi_align

bool v3::ROIAlign::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v3_ROIAlign_evaluate);
    return roi_align::evaluate(inputs, outputs[0], m_pooled_h, m_pooled_w, m_sampling_ratio, m_spatial_scale, m_mode);
}

bool v3::ROIAlign::has_evaluate() const {
    OV_OP_SCOPE(v3_ROIAlign_has_evaluate);
    switch (get_input_element_type(0)) {
    case element::bf16:
    case element::f16:
    case element::f32:
        return true;
    default:
        break;
    }
    return false;
}
}  // namespace op
}  // namespace ov
