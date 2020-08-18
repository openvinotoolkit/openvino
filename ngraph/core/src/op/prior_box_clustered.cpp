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

#include "itt.hpp"

#include "ngraph/op/constant.hpp"
#include "ngraph/op/prior_box_clustered.hpp"

#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/prior_box_clustered.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::PriorBoxClustered::type_info;

op::PriorBoxClustered::PriorBoxClustered(const Output<Node>& layer_shape,
                                         const Output<Node>& image_shape,
                                         const PriorBoxClusteredAttrs& attrs)
    : Op({layer_shape, image_shape})
    , m_attrs(attrs)
{
    constructor_validate_and_infer_types();
}

void op::PriorBoxClustered::validate_and_infer_types()
{
    // shape node should have integer data type. For now we only allow i64
    auto layer_shape_et = get_input_element_type(0);
    NODE_VALIDATION_CHECK(this,
                          layer_shape_et.is_integral_number(),
                          "layer shape input must be an integral number, but is: ",
                          layer_shape_et);

    auto image_shape_et = get_input_element_type(1);
    NODE_VALIDATION_CHECK(this,
                          image_shape_et.is_integral_number(),
                          "image shape input must be an integral number, but is: ",
                          image_shape_et);

    auto layer_shape_rank = get_input_partial_shape(0).rank();
    auto image_shape_rank = get_input_partial_shape(1).rank();
    NODE_VALIDATION_CHECK(this,
                          layer_shape_rank.compatible(image_shape_rank),
                          "layer shape input rank ",
                          layer_shape_rank,
                          " must match image shape input rank ",
                          image_shape_rank);

    NODE_VALIDATION_CHECK(this,
                          m_attrs.widths.size() == m_attrs.heights.size(),
                          "Size of heights vector",
                          m_attrs.widths.size(),
                          " doesn't match size of widths vector ",
                          m_attrs.widths.size());

    set_input_is_relevant_to_shape(0);

    if (auto const_shape = as_type_ptr<op::Constant>(input_value(0).get_node_shared_ptr()))
    {
        NODE_VALIDATION_CHECK(this,
                              shape_size(const_shape->get_shape()) == 2,
                              "Layer shape must have rank 2",
                              const_shape->get_shape());

        auto layer_shape = const_shape->get_shape_val();
        // {Prior boxes, variances-adjusted prior boxes}
        const auto num_priors = m_attrs.widths.size();
        set_output_type(
            0, element::f32, Shape{2, 4 * layer_shape[0] * layer_shape[1] * num_priors});
    }
    else
    {
        set_output_type(0, element::f32, PartialShape::dynamic());
    }
}

shared_ptr<Node> op::PriorBoxClustered::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<PriorBoxClustered>(new_args.at(0), new_args.at(1), m_attrs);
}

bool op::PriorBoxClustered::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("attrs", m_attrs);
    return true;
}

constexpr DiscreteTypeInfo AttributeAdapter<op::PriorBoxClusteredAttrs>::type_info;

AttributeAdapter<op::PriorBoxClusteredAttrs>::AttributeAdapter(op::PriorBoxClusteredAttrs& ref)
    : m_ref(ref)
{
}

bool AttributeAdapter<op::PriorBoxClusteredAttrs>::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("widths", m_ref.widths);
    visitor.on_attribute("heights", m_ref.heights);
    visitor.on_attribute("clip", m_ref.clip);
    visitor.on_attribute("step_widths", m_ref.step_widths);
    visitor.on_attribute("step_heights", m_ref.step_heights);
    visitor.on_attribute("offset", m_ref.offset);
    visitor.on_attribute("variances", m_ref.variances);
    return true;
}

namespace
{
    template <element::Type_t ET>
    bool evaluate(const HostTensorPtr& arg0,
                  const HostTensorPtr& arg1,
                  const HostTensorPtr& out,
                  op::PriorBoxClusteredAttrs attrs)
    {
        runtime::reference::prior_box_clustered(arg0->get_data_ptr<ET>(),
                                                arg1->get_data_ptr<ET>(),
                                                out->get_data_ptr<float>(),
                                                out->get_shape(),
                                                attrs);
        return true;
    }

    bool evaluate_prior_box(const HostTensorPtr& arg0,
                            const HostTensorPtr& arg1,
                            const HostTensorPtr& out,
                            const op::PriorBoxClusteredAttrs& attrs)
    {
        bool rc = true;
        switch (arg0->get_element_type())
        {
            TYPE_CASE(i8)(arg0, arg1, out, attrs);
            break;
            TYPE_CASE(i16)(arg0, arg1, out, attrs);
            break;
            TYPE_CASE(i32)(arg0, arg1, out, attrs);
            break;
            TYPE_CASE(i64)(arg0, arg1, out, attrs);
            break;
            TYPE_CASE(u8)(arg0, arg1, out, attrs);
            break;
            TYPE_CASE(u16)(arg0, arg1, out, attrs);
            break;
            TYPE_CASE(u32)(arg0, arg1, out, attrs);
            break;
            TYPE_CASE(u64)(arg0, arg1, out, attrs);
            break;
        default: rc = false; break;
        }
        return rc;
    }
}

bool op::v0::PriorBoxClustered::evaluate(const HostTensorVector& outputs,
                                         const HostTensorVector& inputs) const
{
    OV_ITT_SCOPED_TASK(itt::domains::nGraphOp, "op::v0::PriorBoxClustered::evaluate");
    return evaluate_prior_box(inputs[0], inputs[1], outputs[0], get_attrs());
}
