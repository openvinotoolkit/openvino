// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/roi_align.hpp"
#include "itt.hpp"

#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/roi_align.hpp"
#include "ngraph/util.hpp" // for host_tensor_2_vector

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::v3::ROIAlign::type_info;

op::v3::ROIAlign::ROIAlign(const Output<Node>& input,
                           const Output<Node>& rois,
                           const Output<Node>& batch_indices,
                           const int pooled_h,
                           const int pooled_w,
                           const int sampling_ratio,
                           const float spatial_scale,
                           const string& mode)
    : Op{{input, rois, batch_indices}}
    , m_pooled_h{pooled_h}
    , m_pooled_w{pooled_w}
    , m_sampling_ratio{sampling_ratio}
    , m_spatial_scale{spatial_scale}
    , m_mode{EnumNames<ROIAlign::PoolingMode>::as_enum(mode)}
{
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
    : Op{{input, rois, batch_indices}}
    , m_pooled_h{pooled_h}
    , m_pooled_w{pooled_w}
    , m_sampling_ratio{sampling_ratio}
    , m_spatial_scale{spatial_scale}
    , m_mode{mode}
{
    constructor_validate_and_infer_types();
}

void op::v3::ROIAlign::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(v3_ROIAlign_validate_and_infer_types);
    NODE_VALIDATION_CHECK(
        this,
        get_input_element_type(0).is_real() && get_input_element_type(1).is_real(),
        "The data type for input and ROIs is expected to be a floating point type. Got: ",
        get_input_element_type(0),
        " and: ",
        get_input_element_type(1));

    NODE_VALIDATION_CHECK(
        this,
        get_input_element_type(0) == get_input_element_type(1),
        "Type of feature maps (inputs) and rois is expected to be the same. Got: ",
        get_input_element_type(0),
        " and: ",
        get_input_element_type(1));

    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(2).is_integral_number(),
                          "The data type for batch indices is expected to be an integer. Got: ",
                          get_input_element_type(2));

    const auto& input_ps = get_input_partial_shape(0);
    NODE_VALIDATION_CHECK(this,
                          input_ps.rank().compatible(4),
                          "Expected a 4D tensor for the input data. Got: ",
                          input_ps);

    const auto& rois_ps = get_input_partial_shape(1);
    NODE_VALIDATION_CHECK(this,
                          rois_ps.rank().compatible(2),
                          "Expected a 2D tensor for the ROIs input. Got: ",
                          rois_ps);

    const auto& batch_indices_ps = get_input_partial_shape(2);
    NODE_VALIDATION_CHECK(this,
                          batch_indices_ps.rank().compatible(1),
                          "Expected a 1D tensor for the batch indices input. Got: ",
                          batch_indices_ps);

    if (rois_ps.rank().is_static())
    {
        const auto rois_second_dim = rois_ps[1];
        NODE_VALIDATION_CHECK(this,
                              rois_second_dim.compatible(4),
                              "The second dimension of ROIs input should contain box coordinates. ",
                              "This dimension is expected to be equal to 4. Got: ",
                              rois_second_dim);

        if (batch_indices_ps.rank().is_static())
        {
            NODE_VALIDATION_CHECK(
                this,
                rois_ps[0].compatible(batch_indices_ps[0]),
                "The first dimension of ROIs input must be equal to the first dimension ",
                "of the batch indices input. Got: ",
                rois_ps[0],
                " and: ",
                batch_indices_ps[0]);
        }
    }

    // the output shape should have the following format [NUM_ROIS, C, pooled_h, pooled_w]
    auto output_shape = PartialShape{{Dimension::dynamic(),
                                      input_ps[1],
                                      Dimension{static_cast<int64_t>(m_pooled_h)},
                                      Dimension{static_cast<int64_t>(m_pooled_w)}}};

    // if either of those 2 dimensions is static its value will be used
    // for the first dimension of the output shape - 'NUM_ROIS'
    if (rois_ps.rank().is_static() && rois_ps[0].is_static())
    {
        output_shape[0] = rois_ps[0];
    }
    else if (batch_indices_ps.rank().is_static() && batch_indices_ps[0].is_static())
    {
        output_shape[0] = batch_indices_ps[0];
    }

    set_output_size(1);
    set_output_type(0, get_input_element_type(0), output_shape);

    // if the channels dimension is not known
    // the first input should be used during the function specialization
    if (input_ps.rank().is_static() && input_ps[1].is_dynamic())
    {
        set_input_is_relevant_to_shape(0);
    }

    // if the 'NUM_ROIS' value is not known
    // the last 2 inputs should be used during the function specialization
    if (output_shape[0].is_dynamic())
    {
        set_input_is_relevant_to_shape(1);
        set_input_is_relevant_to_shape(2);
    }
}

bool op::v3::ROIAlign::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v3_ROIAlign_visit_attributes);
    visitor.on_attribute("pooled_h", m_pooled_h);
    visitor.on_attribute("pooled_w", m_pooled_w);
    visitor.on_attribute("sampling_ratio", m_sampling_ratio);
    visitor.on_attribute("spatial_scale", m_spatial_scale);
    visitor.on_attribute("mode", m_mode);

    return true;
}

shared_ptr<Node> op::v3::ROIAlign::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v3_ROIAlign_clone_with_new_inputs);
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

namespace ngraph
{
    constexpr DiscreteTypeInfo AttributeAdapter<op::v3::ROIAlign::PoolingMode>::type_info;

    template <>
    NGRAPH_API EnumNames<op::v3::ROIAlign::PoolingMode>&
        EnumNames<op::v3::ROIAlign::PoolingMode>::get()
    {
        static auto enum_names =
            EnumNames<op::v3::ROIAlign::PoolingMode>("op::v3::ROIAlign::PoolingMode",
                                                     {{"avg", op::v3::ROIAlign::PoolingMode::AVG},
                                                      {"max", op::v3::ROIAlign::PoolingMode::MAX}});
        return enum_names;
    }

    std::ostream& operator<<(std::ostream& s, const op::v3::ROIAlign::PoolingMode& type)
    {
        return s << as_string(type);
    }
} // namespace ngraph

namespace roi_alinop
{
    template <element::Type_t ET>
    bool evaluate(const HostTensorPtr& feature_maps,
                  const HostTensorPtr& rois,
                  const std::vector<int64_t>& batch_indices_vec_scaled_up,
                  const HostTensorPtr& out,
                  const int pooled_height,
                  const int pooled_width,
                  const int sampling_ratio,
                  const float spatial_scale,
                  const op::v3::ROIAlign::PoolingMode& pooling_mode,
                  const Shape& batch_indices_shape)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::roi_align<T>(feature_maps->get_data_ptr<ET>(),
                                         rois->get_data_ptr<ET>(),
                                         batch_indices_vec_scaled_up.data(),
                                         out->get_data_ptr<ET>(),
                                         feature_maps->get_shape(),
                                         rois->get_shape(),
                                         batch_indices_shape,
                                         out->get_shape(),
                                         pooled_height,
                                         pooled_width,
                                         sampling_ratio,
                                         spatial_scale,
                                         pooling_mode);
        return true;
    }

    bool evaluate_roi_align(const HostTensorVector& args,
                            const HostTensorPtr& out,
                            const int pooled_height,
                            const int pooled_width,
                            const int sampling_ratio,
                            const float spatial_scale,
                            const op::v3::ROIAlign::PoolingMode& pooling_mode)
    {
        auto feature_maps = args[0];
        auto rois = args[1];
        auto batch_indices = args[2];
        std::vector<int64_t> batch_indices_vec_scaled_up =
            host_tensor_2_vector<int64_t>(batch_indices);

        bool rc = true;
        switch (feature_maps->get_element_type())
        {
            NGRAPH_TYPE_CASE(evaluate_roi_align,
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
                             batch_indices->get_shape());
            NGRAPH_TYPE_CASE(evaluate_roi_align,
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
                             batch_indices->get_shape());
            NGRAPH_TYPE_CASE(evaluate_roi_align,
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
                             batch_indices->get_shape());
        default: rc = false; break;
        }

        return rc;
    }
} // namespace roi_alinop

bool op::v3::ROIAlign::evaluate(const HostTensorVector& outputs,
                                const HostTensorVector& inputs) const
{
    NGRAPH_OP_SCOPE(v3_ROIAlign_evaluate);
    return roi_alinop::evaluate_roi_align(
        inputs, outputs[0], m_pooled_h, m_pooled_w, m_sampling_ratio, m_spatial_scale, m_mode);
}

bool op::v3::ROIAlign::has_evaluate() const
{
    NGRAPH_OP_SCOPE(v3_ROIAlign_has_evaluate);
    switch (get_input_element_type(0))
    {
    case ngraph::element::bf16:
    case ngraph::element::f16:
    case ngraph::element::f32: return true;
    default: break;
    }
    return false;
}
