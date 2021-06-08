// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/validation_util.hpp>
#include "itt.hpp"

#include "ngraph/op/constant.hpp"
#include "ngraph/op/prior_box.hpp"

#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/prior_box.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::PriorBox::type_info;

op::PriorBox::PriorBox(const Output<Node>& layer_shape,
                       const Output<Node>& image_shape,
                       const PriorBoxAttrs& attrs)
    : Op({layer_shape, image_shape})
    , m_attrs(attrs)
{
    constructor_validate_and_infer_types();
}

void op::PriorBox::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(v0_PriorBox_validate_and_infer_types);
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

    set_input_is_relevant_to_shape(0);

    if (auto const_shape = get_constant_from_source(input_value(0)))
    {
        NODE_VALIDATION_CHECK(this,
                              shape_size(const_shape->get_shape()) == 2,
                              "Layer shape must have rank 2",
                              const_shape->get_shape());

        auto layer_shape = const_shape->get_shape_val();

        set_output_type(0,
                        element::f32,
                        Shape{2,
                              4 * layer_shape[0] * layer_shape[1] *
                                  static_cast<size_t>(number_of_priors(m_attrs))});
    }
    else
    {
        set_output_type(0, element::f32, PartialShape::dynamic());
    }
}

shared_ptr<Node> op::PriorBox::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v0_PriorBox_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<PriorBox>(new_args.at(0), new_args.at(1), m_attrs);
}

int64_t op::PriorBox::number_of_priors(const PriorBoxAttrs& attrs)
{
    // Starting with 0 number of prior and then various conditions on attributes will contribute
    // real number of prior boxes as PriorBox is a fat thing with several modes of
    // operation that will be checked in order in the next statements.
    int64_t num_priors = 0;

    // Total number of boxes around each point; depends on whether flipped boxes are included
    // plus one box 1x1.
    int64_t total_aspect_ratios = normalized_aspect_ratio(attrs.aspect_ratio, attrs.flip).size();

    if (attrs.scale_all_sizes)
        num_priors = total_aspect_ratios * attrs.min_size.size() + attrs.max_size.size();
    else
        num_priors = total_aspect_ratios + attrs.min_size.size() - 1;

    if (!attrs.fixed_size.empty())
        num_priors = total_aspect_ratios * attrs.fixed_size.size();

    for (auto density : attrs.density)
    {
        auto rounded_density = static_cast<int64_t>(density);
        auto density_2d = (rounded_density * rounded_density - 1);
        if (!attrs.fixed_ratio.empty())
            num_priors += attrs.fixed_ratio.size() * density_2d;
        else
            num_priors += total_aspect_ratios * density_2d;
    }
    return num_priors;
}

std::vector<float> op::PriorBox::normalized_aspect_ratio(const std::vector<float>& aspect_ratio,
                                                         bool flip)
{
    std::set<float> unique_ratios;
    for (auto ratio : aspect_ratio)
    {
        unique_ratios.insert(std::round(ratio * 1e6) / 1e6);
        if (flip)
            unique_ratios.insert(std::round(1 / ratio * 1e6) / 1e6);
    }
    unique_ratios.insert(1);
    return std::vector<float>(unique_ratios.begin(), unique_ratios.end());
}

bool op::PriorBox::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v0_PriorBox_visit_attributes);
    visitor.on_attribute("min_size", m_attrs.min_size);
    visitor.on_attribute("max_size", m_attrs.max_size);
    visitor.on_attribute("aspect_ratio", m_attrs.aspect_ratio);
    visitor.on_attribute("density", m_attrs.density);
    visitor.on_attribute("fixed_ratio", m_attrs.fixed_ratio);
    visitor.on_attribute("fixed_size", m_attrs.fixed_size);
    visitor.on_attribute("clip", m_attrs.clip);
    visitor.on_attribute("flip", m_attrs.flip);
    visitor.on_attribute("step", m_attrs.step);
    visitor.on_attribute("offset", m_attrs.offset);
    visitor.on_attribute("variance", m_attrs.variance);
    visitor.on_attribute("scale_all_sizes", m_attrs.scale_all_sizes);
    return true;
}

namespace prior_box
{
    template <element::Type_t ET>
    bool evaluate(const HostTensorPtr& arg0,
                  const HostTensorPtr& arg1,
                  const HostTensorPtr& out,
                  op::PriorBoxAttrs attrs)
    {
        runtime::reference::prior_box(arg0->get_data_ptr<ET>(),
                                      arg1->get_data_ptr<ET>(),
                                      out->get_data_ptr<float>(),
                                      out->get_shape(),
                                      attrs);
        return true;
    }

    bool evaluate_prior_box(const HostTensorPtr& arg0,
                            const HostTensorPtr& arg1,
                            const HostTensorPtr& out,
                            const op::PriorBoxAttrs& attrs)
    {
        bool rc = true;
        switch (arg0->get_element_type())
        {
            NGRAPH_TYPE_CASE(evaluate_prior_box, i8, arg0, arg1, out, attrs);
            NGRAPH_TYPE_CASE(evaluate_prior_box, i16, arg0, arg1, out, attrs);
            NGRAPH_TYPE_CASE(evaluate_prior_box, i32, arg0, arg1, out, attrs);
            NGRAPH_TYPE_CASE(evaluate_prior_box, i64, arg0, arg1, out, attrs);
            NGRAPH_TYPE_CASE(evaluate_prior_box, u8, arg0, arg1, out, attrs);
            NGRAPH_TYPE_CASE(evaluate_prior_box, u16, arg0, arg1, out, attrs);
            NGRAPH_TYPE_CASE(evaluate_prior_box, u32, arg0, arg1, out, attrs);
            NGRAPH_TYPE_CASE(evaluate_prior_box, u64, arg0, arg1, out, attrs);
        default: rc = false; break;
        }
        return rc;
    }
} // namespace prior_box

bool op::v0::PriorBox::evaluate(const HostTensorVector& outputs,
                                const HostTensorVector& inputs) const
{
    NGRAPH_OP_SCOPE(v0_PriorBox_evaluate);
    return prior_box::evaluate_prior_box(inputs[0], inputs[1], outputs[0], get_attrs());
}

bool op::v0::PriorBox::has_evaluate() const
{
    NGRAPH_OP_SCOPE(v0_PriorBox_has_evaluate);
    switch (get_input_element_type(0))
    {
    case ngraph::element::i8:
    case ngraph::element::i16:
    case ngraph::element::i32:
    case ngraph::element::i64:
    case ngraph::element::u8:
    case ngraph::element::u16:
    case ngraph::element::u32:
    case ngraph::element::u64: return true;
    default: break;
    }
    return false;
}
