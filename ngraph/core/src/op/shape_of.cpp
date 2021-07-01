// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <ngraph/validation_util.hpp>
#include <vector>

#include "itt.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/gather.hpp"
#include "ngraph/op/select.hpp"
#include "ngraph/op/shape_of.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/shape_of.hpp"
#include "ngraph/type/element_type_traits.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::v3::ShapeOf::type_info;

op::v3::ShapeOf::ShapeOf(const Output<Node>& arg, element::Type output_type)
    : Op({arg})
    , m_output_type(output_type)
{
    constructor_validate_and_infer_types();
}

void op::v3::ShapeOf::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(v3_ShapeOf_validate_and_infer_types);
    NODE_VALIDATION_CHECK(this,
                          m_output_type == element::i64 || m_output_type == element::i32,
                          "Output type must be i32 or i64");
    set_input_is_relevant_to_value(0, false);
    const auto input_partial_shape = get_input_partial_shape(0);
    set_output_type(0, m_output_type, PartialShape{input_partial_shape.rank()});
}

bool ngraph::op::v3::ShapeOf::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v3_ShapeOf_visit_attributes);
    visitor.on_attribute("output_type", m_output_type);
    return true;
}

shared_ptr<Node> op::v3::ShapeOf::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v3_ShapeOf_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    auto new_shape_of = make_shared<op::v3::ShapeOf>(new_args.at(0), m_output_type);
    new_shape_of->set_is_foldable(m_is_foldable);
    return new_shape_of;
}

namespace shape_of
{
    template <element::Type_t ET>
    inline bool evaluate(const Shape& shape, const HostTensorPtr& output_value)
    {
        runtime::reference::shape_of(shape, output_value->get_data_ptr<ET>());
        return true;
    }

    bool evaluate_shape_of(const HostTensorPtr& output_value, const HostTensorPtr& input_value)
    {
        bool rc = true;
        Shape shape = input_value->get_shape();
        output_value->set_shape(Shape{shape.size()});
        switch (output_value->get_element_type())
        {
            NGRAPH_TYPE_CASE(evaluate_shape_of, i32, shape, output_value);
            NGRAPH_TYPE_CASE(evaluate_shape_of, i64, shape, output_value);
            NGRAPH_TYPE_CASE(evaluate_shape_of, u32, shape, output_value);
            NGRAPH_TYPE_CASE(evaluate_shape_of, u64, shape, output_value);
        default: rc = false; break;
        }
        return rc;
    }

    bool constant_fold_shape_of(Node* shape_of_node,
                                Output<Node>& replacement,
                                const Output<Node>& shape_of_input,
                                bool is_foldable)
    {
        auto partial_shape = shape_of_input.get_partial_shape();
        auto output_type = shape_of_node->get_output_element_type(0);
        if (partial_shape.is_static())
        {
            auto arg_shape = shape_of_input.get_shape();
            auto result_tensor =
                make_shared<HostTensor>(output_type, shape_of_node->get_output_shape(0));
            if (evaluate_shape_of(result_tensor,
                                  make_shared<HostTensor>(output_type, partial_shape)))
            {
                replacement = make_shared<op::Constant>(result_tensor);
                return true;
            }
            return false;
        }
        else if (partial_shape.rank().is_static() && is_foldable)
        {
            auto shape_of = shape_of_node->copy_with_new_inputs({shape_of_input});
            // Ugly
            if (auto ps = as_type_ptr<op::v0::ShapeOf>(shape_of))
            {
                ps->set_is_foldable(false);
            }
            else if (auto ps = as_type_ptr<op::v3::ShapeOf>(shape_of))
            {
                ps->set_is_foldable(false);
            }
            auto dimensions = OutputVector{};
            auto output_dimensions = vector<Dimension>(partial_shape);
            for (size_t i = 0; i < output_dimensions.size(); ++i)
            {
                if (output_dimensions[i].is_static())
                {
                    auto temp = std::make_shared<op::v0::Constant>(
                        output_type,
                        Shape{1},
                        std::vector<int64_t>{output_dimensions[i].get_length()});
                    temp->set_friendly_name("ConstDim/" + temp->get_name());
                    dimensions.emplace_back(temp);
                }
                else
                {
                    auto index = std::make_shared<op::v0::Constant>(
                        output_type, Shape{1}, std::vector<int64_t>{static_cast<int64_t>(i)});
                    auto axis = std::make_shared<op::v0::Constant>(
                        element::i64, Shape{}, std::vector<int64_t>{0});
                    auto temp = make_shared<op::v1::Gather>(shape_of, index, axis);
                    temp->set_friendly_name("DynDim/" + temp->get_name());
                    dimensions.emplace_back(temp);
                }
            }

            replacement = std::make_shared<op::Concat>(dimensions, 0);
            return true;
        }
        return false;
    }

    bool evaluate_bound_shape(const Node* shape_of_node,
                              const HostTensorVector& output_values,
                              bool is_upper)
    {
        NGRAPH_CHECK(shape_of_node, validate_host_tensor_vector(output_values, 1));
        const auto& input_partial_shape = shape_of_node->get_input_partial_shape(0);
        if (input_partial_shape.rank().is_dynamic())
            return false;
        const auto rank = input_partial_shape.rank().get_length();
        auto pshape_low = PartialShape::dynamic(rank), pshape_up = PartialShape::dynamic(rank);
        for (Dimension::value_type i = 0; i < rank; ++i)
        {
            Interval interval = input_partial_shape[i].get_interval();
            pshape_low[i] = interval.get_min_val();
            pshape_up[i] = Dimension(interval.get_max_val()).is_dynamic()
                               ? Dimension(interval.get_max_val() - 1)
                               : interval.get_max_val();
        }
        NGRAPH_CHECK(pshape_up.is_static() && pshape_low.is_static());
        const auto input_et = shape_of_node->get_input_element_type(0);
        const auto output_et = shape_of_node->get_output_element_type(0);
        if (pshape_low.to_shape() == pshape_up.to_shape())
        {
            shape_of_node->evaluate(output_values,
                                    {std::make_shared<HostTensor>(input_et, pshape_low)});
            shape_of_node->get_output_tensor(0).set_lower_value(output_values[0]);
            shape_of_node->get_output_tensor(0).set_upper_value(output_values[0]);
        }
        else
        {
            HostTensorVector upper =
                is_upper ? output_values
                         : HostTensorVector{std::make_shared<HostTensor>(
                               output_et, PartialShape{pshape_up.rank().get_length()})};
            shape_of_node->evaluate(upper, {std::make_shared<HostTensor>(input_et, pshape_up)});
            shape_of_node->get_output_tensor(0).set_upper_value(upper[0]);

            HostTensorVector lower =
                !is_upper ? output_values
                          : HostTensorVector{std::make_shared<HostTensor>(
                                output_et, PartialShape{pshape_low.rank().get_length()})};
            shape_of_node->evaluate(lower, {std::make_shared<HostTensor>(input_et, pshape_low)});
            shape_of_node->get_output_tensor(0).set_lower_value(lower[0]);

            vector<bool> dynamic_mask; // true if dimension is dynamic
            for (const auto& i : input_partial_shape)
                dynamic_mask.push_back(Dimension(i.get_interval().get_max_val()).is_dynamic());
            auto mask_const =
                ngraph::op::Constant::create(element::boolean, {dynamic_mask.size()}, dynamic_mask);
            auto dynamic_min_const = ngraph::op::Constant::create(output_et, {}, {0});
            auto dynamic_max_const = ngraph::op::Constant::create(
                output_et,
                {},
                {output_et == element::i64 ? std::numeric_limits<int64_t>::max()
                                           : std::numeric_limits<int32_t>::max()});

            op::v1::Select().evaluate(lower,
                                      {std::make_shared<HostTensor>(mask_const),
                                       std::make_shared<HostTensor>(dynamic_min_const),
                                       lower[0]});
            op::v1::Select().evaluate(upper,
                                      {std::make_shared<HostTensor>(mask_const),
                                       std::make_shared<HostTensor>(dynamic_max_const),
                                       upper[0]});
        }
        return true;
    }
} // namespace shape_of

bool op::v3::ShapeOf::evaluate(const HostTensorVector& output_values,
                               const HostTensorVector& input_values) const
{
    NGRAPH_OP_SCOPE(v3_ShapeOf_evaluate);
    NGRAPH_CHECK(validate_host_tensor_vector(input_values, 1));
    NGRAPH_CHECK(validate_host_tensor_vector(output_values, 1));
    return shape_of::evaluate_shape_of(output_values[0], input_values[0]);
}

bool op::v3::ShapeOf::has_evaluate() const
{
    NGRAPH_OP_SCOPE(v3_ShapeOf_has_evaluate);
    switch (get_output_element_type(0))
    {
    case ngraph::element::i32:
    case ngraph::element::i64:
    case ngraph::element::u32:
    case ngraph::element::u64: return true;
    default: break;
    }
    return false;
}

bool op::v3::ShapeOf::evaluate_lower(const HostTensorVector& output_values) const
{
    return shape_of::evaluate_bound_shape(this, output_values, false);
}

bool op::v3::ShapeOf::evaluate_upper(const HostTensorVector& output_values) const
{
    return shape_of::evaluate_bound_shape(this, output_values, true);
}

bool op::v3::ShapeOf::constant_fold(OutputVector& output_values, const OutputVector& input_values)
{
    OV_ITT_SCOPED_TASK(itt::domains::nGraph, "op::v3::ShapeOf::constant_fold");
    if (get_rt_info().count("DISABLED_CONSTANT_FOLDING"))
        return false;
    return shape_of::constant_fold_shape_of(this, output_values[0], input_values[0], m_is_foldable);
}

// op::v0::ShapeOf
NGRAPH_RTTI_DEFINITION(op::v0::ShapeOf, "ShapeOf", 0);

op::v0::ShapeOf::ShapeOf(const Output<Node>& arg)
    : Op({arg})
{
    constructor_validate_and_infer_types();
}

void op::v0::ShapeOf::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(v0_ShapeOf_validate_and_infer_types);
    set_input_is_relevant_to_value(0, false);
    set_output_type(0, element::i64, PartialShape{get_input_partial_shape(0).rank()});
}

bool ngraph::op::v0::ShapeOf::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v0_ShapeOf_visit_attributes);
    return true;
}

shared_ptr<Node> op::v0::ShapeOf::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v0_ShapeOf_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    auto new_shape_of = make_shared<op::v0::ShapeOf>(new_args.at(0));
    NGRAPH_CHECK(new_shape_of.get(),
                 new_shape_of != nullptr,
                 "Cannot clone ",
                 description(),
                 " operation with name ",
                 get_friendly_name());
    new_shape_of->set_is_foldable(m_is_foldable);
    return new_shape_of;
}

bool op::v0::ShapeOf::evaluate(const HostTensorVector& output_values,
                               const HostTensorVector& input_values) const
{
    NGRAPH_OP_SCOPE(v0_ShapeOf_evaluate);
    NGRAPH_CHECK(validate_host_tensor_vector(input_values, 1));
    NGRAPH_CHECK(validate_host_tensor_vector(output_values, 1));
    return shape_of::evaluate_shape_of(output_values[0], input_values[0]);
}

bool op::v0::ShapeOf::has_evaluate() const
{
    NGRAPH_OP_SCOPE(v0_ShapeOf_has_evaluate);
    switch (get_output_element_type(0))
    {
    case ngraph::element::i32:
    case ngraph::element::i64:
    case ngraph::element::u32:
    case ngraph::element::u64: return true;
    default: break;
    }
    return false;
}

bool op::v0::ShapeOf::constant_fold(OutputVector& output_values, const OutputVector& input_values)
{
    OV_ITT_SCOPED_TASK(itt::domains::nGraph, "op::v0::ShapeOf::constant_fold");
    if (get_rt_info().count("DISABLED_CONSTANT_FOLDING"))
        return false;
    return shape_of::constant_fold_shape_of(this, output_values[0], input_values[0], m_is_foldable);
}

bool op::v0::ShapeOf::evaluate_lower(const HostTensorVector& output_values) const
{
    return shape_of::evaluate_bound_shape(this, output_values, false);
}

bool op::v0::ShapeOf::evaluate_upper(const HostTensorVector& output_values) const
{
    return shape_of::evaluate_bound_shape(this, output_values, true);
}
