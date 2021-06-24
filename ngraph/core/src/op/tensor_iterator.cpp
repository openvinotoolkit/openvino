// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/tensor_iterator.hpp"
#include "itt.hpp"
#include "ngraph/factory.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/specialize_function.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::v0::TensorIterator::type_info;

op::v0::TensorIterator::TensorIterator(const OutputVector& values)
    : op::util::SubGraphOp(values)
{
}

bool op::v0::TensorIterator::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v0_TensorIterator_visit_attributes);
    visitor.on_attribute("body", m_body);
    visitor.on_attribute("input_descriptions", m_input_descriptions);
    visitor.on_attribute("output_descriptions", m_output_descriptions);

    return true;
}

void op::v0::TensorIterator::revalidate_and_infer_types_for_body_ops()
{
    std::stack<std::shared_ptr<Node>, std::vector<std::shared_ptr<Node>>> nodes_to_do;
    std::unordered_set<std::shared_ptr<Node>> nodes_done;

    for (const auto& r : m_body->get_results())
    {
        nodes_to_do.push(r);
    }
    while (nodes_to_do.size() > 0)
    {
        auto node = nodes_to_do.top();
        if (nodes_done.count(node) == 0)
        {
            NGRAPH_CHECK(as_type_ptr<op::v0::TensorIterator>(node) == nullptr,
                         "No nested TensorIterator");
            bool can_add = true;
            size_t arg_count = node->get_input_size();
            for (size_t i = 0; i < arg_count; ++i)
            {
                auto dep = node->input(arg_count - i - 1)
                               .get_source_output()
                               .get_node()
                               ->shared_from_this();
                if (nodes_done.count(dep) == 0)
                {
                    can_add = false;
                    nodes_to_do.push(dep);
                }
            }
            if (can_add)
            {
                nodes_done.insert(node);
                node->revalidate_and_infer_types();
                nodes_to_do.pop();
            }
        }
        else
        {
            nodes_to_do.pop();
        }
    }
}

void op::v0::TensorIterator::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(v0_TensorIterator_validate_and_infer_types);
    NODE_VALIDATION_CHECK(this,
                          get_input_size() == m_input_descriptions.size(),
                          "Number of inputs must be the same as number of input descriptions");

    std::vector<std::shared_ptr<Node>> ends;

    auto make_positive = [](int64_t value, uint64_t dim_size) -> int64_t {
        if (value < 0)
        {
            value = dim_size + value;
        }
        return value;
    };

    // Input
    for (const auto& input_description : m_input_descriptions)
    {
        auto index = input_description->m_input_index;

        if (auto slice_input_description = as_type_ptr<SliceInputDescription>(input_description))
        {
            auto body_parameter =
                m_body->get_parameters().at(slice_input_description->m_body_parameter_index);
            auto input_partial_shape = inputs().at(index).get_source_output().get_partial_shape();
            if (input_partial_shape.is_static())
            {
                auto input_shape = input_partial_shape.to_shape();
                auto axis = slice_input_description->m_axis;
                auto part_size = slice_input_description->m_part_size;

                auto dim_size = input_shape[axis];
                auto start = make_positive(slice_input_description->m_start, dim_size);
                auto end = make_positive(slice_input_description->m_end, dim_size);

                // +1 because the left and right borders are included [start, end]
                m_num_iterations = (abs(end - start) + 1) / part_size;
                // infer type for m_body_parameter
                Shape out_shape{input_shape};
                out_shape[axis] = part_size;
                body_parameter->set_partial_shape(out_shape);
            }
            else
            {
                body_parameter->set_partial_shape(
                    PartialShape::dynamic(input_partial_shape.rank()));
            }
        }
        else if (auto merged_input_description =
                     as_type_ptr<MergedInputDescription>(input_description))
        {
            auto body_value =
                m_body->get_results().at(merged_input_description->m_body_value_index)->input(0);
            ends.push_back(body_value.get_node()->shared_from_this());

            auto body_parameter =
                m_body->get_parameters().at(merged_input_description->m_body_parameter_index);

            auto body_param_partial_shape = body_parameter->get_partial_shape();
            auto input_partial_shape = inputs().at(index).get_source_output().get_partial_shape();
            body_parameter->set_partial_shape(input_partial_shape);
        }
        else if (auto invariant_input_description =
                     as_type_ptr<InvariantInputDescription>(input_description))
        {
            auto body_parameter =
                m_body->get_parameters().at(invariant_input_description->m_body_parameter_index);

            auto body_param_partial_shape = body_parameter->get_partial_shape();
            auto input_partial_shape = inputs().at(index).get_source_output().get_partial_shape();
            body_parameter->set_partial_shape(input_partial_shape);
        }
    }

    // Body
    revalidate_and_infer_types_for_body_ops();

    // Output
    try_to_set_num_iterations_if_no_slice_inputs();

    for (const auto& output_description : m_output_descriptions)
    {
        auto index = output_description->m_output_index;

        auto body_value =
            m_body->get_results().at(output_description->m_body_value_index)->input_value(0);

        if (auto concat_output_description =
                as_type_ptr<ConcatOutputDescription>(output_description))
        {
            auto body_value_partial_shape = body_value.get_partial_shape();
            set_output_type(index, body_value.get_element_type(), PartialShape::dynamic());
            if (body_value_partial_shape.is_static())
            {
                auto body_value_shape = body_value_partial_shape.to_shape();
                auto part_size = concat_output_description->m_part_size;
                auto axis = concat_output_description->m_axis;

                Shape out_shape{body_value_shape};

                if (body_value_shape.empty())
                {
                    NODE_VALIDATION_CHECK(this,
                                          axis == 0,
                                          "Axis must be equal to 0 if concatenated output "
                                          "tensor slices are scalars. "
                                          "TensorIterator output index: ",
                                          index);
                    out_shape = Shape(1);
                }

                if (m_num_iterations != -1)
                {
                    // for simple RNN case where stride is the same as part_size
                    out_shape[axis] = m_num_iterations * part_size;
                    set_output_type(index, body_value.get_element_type(), out_shape);
                }
            }
            else
            {
                set_output_type(index,
                                body_value.get_element_type(),
                                PartialShape::dynamic(body_value.get_partial_shape().rank()));
            }
        }
        else if (auto body_output_description =
                     as_type_ptr<BodyOutputDescription>(output_description))
        {
            set_output_type(index, body_value.get_element_type(), body_value.get_partial_shape());
        }
    }

    NODE_VALIDATION_CHECK(this,
                          get_output_size() == m_output_descriptions.size(),
                          "Number of outputs must be the same as number of output descriptions");
}

std::shared_ptr<Function> op::v0::TensorIterator::get_function()
{
    return get_body();
}

namespace
{
    template <typename Desc>
    bool has_slice_input_desc(const Desc& desc)
    {
        const auto is_slice_input_desc = +[](typename Desc::const_reference d) {
            return is_type<op::util::SubGraphOp::SliceInputDescription>(d);
        };
        return std::any_of(begin(desc), end(desc), is_slice_input_desc);
    }
} // namespace

void op::v0::TensorIterator::try_to_set_num_iterations_if_no_slice_inputs()
{
    if (m_num_iterations != -1 || has_slice_input_desc(get_input_descriptions()))
    {
        return;
    }

    for (const auto& output_description : m_output_descriptions)
    {
        if (auto concat = as_type_ptr<ConcatOutputDescription>(output_description))
        {
            m_num_iterations = ((std::abs(concat->m_end - concat->m_start)) / concat->m_part_size);
            break;
        }
    }
}

std::shared_ptr<Node>
    op::v0::TensorIterator::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v0_TensorIterator_clone_with_new_inputs);
    auto op = make_shared<op::v0::TensorIterator>(new_args);
    NGRAPH_CHECK(op.get(),
                 op != nullptr,
                 "Cannot clone ",
                 description(),
                 " operation with name ",
                 get_friendly_name());
    op->set_output_size(m_output_descriptions.size());

    std::vector<::ngraph::element::Type> types(m_body->get_parameters().size());
    std::vector<::ngraph::PartialShape> new_shapes(m_body->get_parameters().size());

    for (size_t input_index = 0; input_index < new_args.size(); ++input_index)
    {
        for (auto& input_description : m_input_descriptions)
        {
            if (input_description->m_input_index == input_index)
            {
                types[input_description->m_body_parameter_index] =
                    new_args[input_index].get_element_type();
                new_shapes[input_description->m_body_parameter_index] =
                    new_args[input_index].get_partial_shape();

                if (new_shapes[input_description->m_body_parameter_index].is_static())
                {
                    if (auto slice_in = ::ngraph::as_type_ptr<
                            ngraph::op::v0::TensorIterator::SliceInputDescription>(
                            input_description))
                    {
                        new_shapes[slice_in->m_body_parameter_index][slice_in->m_axis] =
                            slice_in->m_part_size;
                    }
                }
            }
        }
    }

    op->m_num_iterations = m_num_iterations;
    auto func = std::make_shared<Function>(
        m_body->get_results(), m_body->get_sinks(), m_body->get_parameters());
    auto spec_func =
        specialize_function(func, types, new_shapes, std::vector<void*>(new_args.size(), nullptr));
    op->m_body = std::make_shared<Function>(
        spec_func->get_results(), spec_func->get_sinks(), spec_func->get_parameters());

    for (auto& input_description : m_input_descriptions)
    {
        op->m_input_descriptions.push_back(input_description->copy());
    }
    for (auto& output_description : m_output_descriptions)
    {
        op->m_output_descriptions.push_back(output_description->copy());
    }
    op->validate_and_infer_types();
    return op;
}
