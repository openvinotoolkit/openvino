//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include "roi_align.hpp"

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
    NODE_VALIDATION_CHECK(
        this,
        get_input_element_type(0).is_real() && get_input_element_type(1).is_real(),
        "The data type for input and ROIs is expected to be a floating point type. Got: ",
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
    visitor.on_attribute("pooled_h", m_pooled_h);
    visitor.on_attribute("pooled_w", m_pooled_w);
    visitor.on_attribute("sampling_ratio", m_sampling_ratio);
    visitor.on_attribute("spatial_scale", m_spatial_scale);
    visitor.on_attribute("mode", m_mode);

    return true;
}

shared_ptr<Node> op::v3::ROIAlign::clone_with_new_inputs(const OutputVector& new_args) const
{
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
    EnumNames<op::v3::ROIAlign::PoolingMode>& EnumNames<op::v3::ROIAlign::PoolingMode>::get()
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
}
