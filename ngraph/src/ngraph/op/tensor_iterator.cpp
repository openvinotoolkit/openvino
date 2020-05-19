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

#include "ngraph/op/tensor_iterator.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/pass/get_output_element_elimination.hpp"
#include "ngraph/specialize_function.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::TensorIterator::type_info;

constexpr DiscreteTypeInfo op::TensorIterator::SliceInputDescription::type_info;
constexpr DiscreteTypeInfo op::TensorIterator::MergedInputDescription::type_info;
constexpr DiscreteTypeInfo op::TensorIterator::InvariantInputDescription::type_info;

constexpr DiscreteTypeInfo op::TensorIterator::BodyOutputDescription::type_info;
constexpr DiscreteTypeInfo op::TensorIterator::ConcatOutputDescription::type_info;

constexpr DiscreteTypeInfo op::TensorIterator::BodyLambda::type_info;

op::TensorIterator::TensorIterator(const OutputVector& values)
    : op::util::FusedOp(values)
{
}

op::TensorIterator::InputDescription::InputDescription(uint64_t input_index,
                                                       uint64_t body_parameter_index)
    : m_input_index(input_index)
    , m_body_parameter_index(body_parameter_index)
{
}

op::TensorIterator::SliceInputDescription::SliceInputDescription(uint64_t input_index,
                                                                 uint64_t body_parameter_index,
                                                                 int64_t start,
                                                                 int64_t stride,
                                                                 int64_t part_size,
                                                                 int64_t end,
                                                                 int64_t axis)
    : InputDescription(input_index, body_parameter_index)
    , m_start(start)
    , m_stride(stride)
    , m_part_size(part_size)
    , m_end(end)
    , m_axis(axis)
{
}

shared_ptr<op::TensorIterator::InputDescription>
    op::TensorIterator::SliceInputDescription::copy() const
{
    return make_shared<SliceInputDescription>(
        m_input_index, m_body_parameter_index, m_start, m_stride, m_part_size, m_end, m_axis);
}

op::TensorIterator::MergedInputDescription::MergedInputDescription(uint64_t input_index,
                                                                   uint64_t body_parameter_index,
                                                                   uint64_t body_value_index)
    : InputDescription(input_index, body_parameter_index)
    , m_body_value_index(body_value_index)
{
}

shared_ptr<op::TensorIterator::InputDescription>
    op::TensorIterator::MergedInputDescription::copy() const
{
    return make_shared<MergedInputDescription>(
        m_input_index, m_body_parameter_index, m_body_value_index);
}

op::TensorIterator::InvariantInputDescription::InvariantInputDescription(
    uint64_t input_index, uint64_t body_parameter_index)
    : InputDescription(input_index, body_parameter_index)
{
}

shared_ptr<op::TensorIterator::InputDescription>
    op::TensorIterator::InvariantInputDescription::copy() const
{
    return make_shared<InvariantInputDescription>(m_input_index, m_body_parameter_index);
}

op::TensorIterator::OutputDescription::OutputDescription(uint64_t body_value_index,
                                                         uint64_t output_index)
    : m_body_value_index(body_value_index)
    , m_output_index(output_index)
{
}

op::TensorIterator::ConcatOutputDescription::ConcatOutputDescription(uint64_t body_value_index,
                                                                     uint64_t output_index,
                                                                     int64_t start,
                                                                     int64_t stride,
                                                                     int64_t part_size,
                                                                     int64_t end,
                                                                     int64_t axis)
    : OutputDescription(body_value_index, output_index)
    , m_start(start)
    , m_stride(stride)
    , m_part_size(part_size)
    , m_end(end)
    , m_axis(axis)
{
}

shared_ptr<op::TensorIterator::OutputDescription>
    op::TensorIterator::ConcatOutputDescription::copy() const
{
    return make_shared<ConcatOutputDescription>(
        m_body_value_index, m_output_index, m_start, m_stride, m_part_size, m_end, m_axis);
}

op::TensorIterator::BodyOutputDescription::BodyOutputDescription(uint64_t body_value_index,
                                                                 uint64_t output_index,
                                                                 int64_t iteration)
    : OutputDescription(body_value_index, output_index)
    , m_iteration(iteration)
{
}

shared_ptr<op::TensorIterator::OutputDescription>
    op::TensorIterator::BodyOutputDescription::copy() const
{
    return make_shared<BodyOutputDescription>(m_body_value_index, m_output_index, m_iteration);
}

Input<Node> op::TensorIterator::input_for_value(const Output<Node>& value)
{
    auto input_index = get_input_size();
    set_argument(input_index, value);
    return Input<Node>(this, input_index);
}

void op::TensorIterator::set_sliced_input(const std::shared_ptr<op::Parameter>& body_parameter,
                                          const Output<Node>& value,
                                          int64_t start,
                                          int64_t stride,
                                          int64_t part_size,
                                          int64_t end,
                                          int64_t axis)
{
    m_input_descriptions.push_back(
        make_shared<SliceInputDescription>(input_for_value(value).get_index(),
                                           m_body->get_parameter_index(body_parameter),
                                           start,
                                           stride,
                                           part_size,
                                           end,
                                           axis));
}

void op::TensorIterator::set_merged_input(const std::shared_ptr<Parameter>& body_parameter,
                                          const Output<Node>& initial_value,
                                          const Output<Node>& successive_value)
{
    m_input_descriptions.push_back(
        make_shared<MergedInputDescription>(input_for_value(initial_value).get_index(),
                                            m_body->get_parameter_index(body_parameter),
                                            m_body->get_result_index(successive_value)));
}

void op::TensorIterator::set_invariant_input(const std::shared_ptr<Parameter>& body_parameter,
                                             const Output<Node>& value)
{
    m_input_descriptions.push_back(make_shared<InvariantInputDescription>(
        input_for_value(value).get_index(), m_body->get_parameter_index(body_parameter)));
}

Output<Node> op::TensorIterator::get_iter_value(const Output<Node>& body_value, int64_t iteration)
{
    auto output_index = get_output_size();
    m_output_descriptions.push_back(make_shared<BodyOutputDescription>(
        m_body->get_result_index(body_value), output_index, iteration));
    set_output_size(output_index + 1);
    return Output<Node>(shared_from_this(), output_index);
}

Output<Node> op::TensorIterator::get_concatenated_slices(const Output<Node>& body_value,
                                                         int64_t start,
                                                         int64_t stride,
                                                         int64_t part_size,
                                                         int64_t end,
                                                         int64_t axis)
{
    auto output_index = get_output_size();
    m_output_descriptions.push_back(make_shared<ConcatOutputDescription>(
        m_body->get_result_index(body_value), output_index, start, stride, part_size, end, axis));
    set_output_size(output_index + 1);
    return Output<Node>(shared_from_this(), output_index);
}

NodeVector op::TensorIterator::decompose_op() const
{
    // Stub
    return NodeVector{};
}

void op::TensorIterator::revalidate_and_infer_types_for_body_ops()
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
            NGRAPH_CHECK(as_type_ptr<op::TensorIterator>(node) == nullptr,
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

void op::TensorIterator::validate_and_infer_types()
{
    NODE_VALIDATION_CHECK(this,
                          get_input_size() == m_input_descriptions.size(),
                          "Number of inputs must be the same as number of input descriptions");

    NODE_VALIDATION_CHECK(this,
                          get_output_size() == m_output_descriptions.size(),
                          "Number of outputs must be the same as number of output descriptions");

    std::vector<std::shared_ptr<Node>> ends;

    auto make_positive = [](int64_t value, uint64_t dim_size) -> int64_t {
        if (value < 0)
        {
            value = dim_size + value;
        }
        return value;
    };

    // Input
    uint64_t index_it = 0;
    for (const auto& input_description : m_input_descriptions)
    {
        auto index = input_description->m_input_index;
        NODE_VALIDATION_CHECK(this, index == index_it, "Input_index not in order");
        index_it++;

        if (auto slice_input_description = as_type_ptr<SliceInputDescription>(input_description))
        {
            auto body_parameter =
                m_body->get_parameters().at(slice_input_description->m_body_parameter_index);
            auto body_param_partial_shape = body_parameter->get_partial_shape();
            auto input_partial_shape = inputs().at(index).get_source_output().get_partial_shape();
            if (input_partial_shape.is_static())
            {
                auto input_shape = input_partial_shape.to_shape();
                auto axis = slice_input_description->m_axis;
                auto part_size = slice_input_description->m_part_size;

                auto dim_size = input_shape[axis];
                auto start = make_positive(slice_input_description->m_start, dim_size);
                auto end = make_positive(slice_input_description->m_end, dim_size);

                if (m_num_iterations == -1)
                {
                    // +1 because the left and right borders are included [start, end]
                    m_num_iterations = (abs(end - start) + 1) / part_size;
                }
                else
                {
                    NODE_VALIDATION_CHECK(this,
                                          m_num_iterations == (abs(end - start) + 1) / part_size,
                                          "Number of slices not the same");
                }

                if (body_param_partial_shape.is_static())
                {
                    // validate
                    auto body_param_shape = body_param_partial_shape.to_shape();
                    for (auto i = 0; i < input_shape.size(); i++)
                    {
                        if (i != axis)
                        {
                            NODE_VALIDATION_CHECK(
                                this,
                                input_shape[i] == body_param_shape[i],
                                "Iterator input is not compatible with body param");
                        }
                    }
                }
                else
                {
                    // infer type for m_body_parameter
                    Shape out_shape{input_shape};
                    out_shape[axis] = part_size;
                    body_parameter->set_partial_shape(out_shape);
                }
            }
        }
        else if (auto merged_input_description =
                     as_type_ptr<MergedInputDescription>(input_description))
        {
            auto body_value =
                m_body->get_results().at(merged_input_description->m_body_value_index)->input(0);
            ends.push_back(body_value.get_node()->shared_from_this());

            auto body_value_partial_shape = body_value.get_partial_shape();
            auto body_parameter =
                m_body->get_parameters().at(merged_input_description->m_body_parameter_index);

            auto body_param_partial_shape = body_parameter->get_partial_shape();
            auto input_partial_shape = inputs().at(index).get_source_output().get_partial_shape();
            NODE_VALIDATION_CHECK(this,
                                  body_value_partial_shape.compatible(body_param_partial_shape),
                                  "Iterator successive value is not compatible with body param");
            NODE_VALIDATION_CHECK(this,
                                  input_partial_shape.compatible(body_param_partial_shape),
                                  "Iterator initial value is not compatible with body param");

            if (input_partial_shape.is_static())
            {
                auto input_shape = input_partial_shape.to_shape();
                // infer type for body_parameter
                if (body_param_partial_shape.is_dynamic())
                {
                    body_parameter->set_partial_shape(input_shape);
                }
            }
        }
        else if (auto invariant_input_description =
                     as_type_ptr<InvariantInputDescription>(input_description))
        {
            auto body_parameter =
                m_body->get_parameters().at(invariant_input_description->m_body_parameter_index);

            auto body_param_partial_shape = body_parameter->get_partial_shape();
            auto input_partial_shape = inputs().at(index).get_source_output().get_partial_shape();
            NODE_VALIDATION_CHECK(this,
                                  input_partial_shape.compatible(body_param_partial_shape),
                                  "Iterator initial value is not compatible with body param");

            if (input_partial_shape.is_static())
            {
                auto input_shape = input_partial_shape.to_shape();
                // infer type for m_body_parameter
                if (body_param_partial_shape.is_dynamic())
                {
                    body_parameter->set_partial_shape(input_shape);
                }
            }
        }
    }

    // Body
    revalidate_and_infer_types_for_body_ops();

    // Output
    index_it = 0;
    for (const auto& output_description : m_output_descriptions)
    {
        auto index = output_description->m_output_index;
        NODE_VALIDATION_CHECK(this, index == index_it, "Output_index not in order");
        index_it++;

        auto body_value =
            m_body->get_results().at(output_description->m_body_value_index)->input_value(0);

        if (auto concat_output_description =
                as_type_ptr<ConcatOutputDescription>(output_description))
        {
            auto body_value_partial_shape = body_value.get_partial_shape();
            if (body_value_partial_shape.is_static())
            {
                auto body_value_shape = body_value_partial_shape.to_shape();
                auto part_size = concat_output_description->m_part_size;
                auto axis = concat_output_description->m_axis;

                Shape out_shape{body_value_shape};
                if (m_num_iterations != -1)
                {
                    // for simple RNN case where stride is the same as part_size
                    out_shape[axis] = m_num_iterations * part_size;
                    set_output_type(index, body_value.get_element_type(), out_shape);
                }
            }
        }
        else if (auto body_output_description =
                     as_type_ptr<BodyOutputDescription>(output_description))
        {
            set_output_type(index, body_value.get_element_type(), body_value.get_partial_shape());
        }
    }
}

std::shared_ptr<Node> op::TensorIterator::clone_with_new_inputs(const OutputVector& new_args) const
{
    auto op = make_shared<op::TensorIterator>(new_args);
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
                            ngraph::op::TensorIterator::SliceInputDescription>(input_description))
                    {
                        new_shapes[slice_in->m_body_parameter_index][slice_in->m_axis] =
                            slice_in->m_part_size;
                    }
                }
            }
        }
    }

    op->m_num_iterations = m_num_iterations;
    auto func = std::make_shared<Function>(m_body->get_results(), m_body->get_parameters());
    auto spec_func = specialize_function(
        func, types, new_shapes, std::vector<void*>(new_args.size(), nullptr), false, true);
    op->m_body =
        std::make_shared<BodyLambda>(spec_func->get_results(), spec_func->get_parameters());

    // TODO: remove this code after the fix on the nGraph side (GetOutputElements)
    ::ngraph::pass::GetOutputElementElimination goe_elimination;
    for (const auto& n : spec_func->get_ops())
    {
        goe_elimination.run_on_node(n);
    }

    for (auto& input_description : m_input_descriptions)
    {
        op->m_input_descriptions.push_back(input_description->copy());
    }
    for (auto& output_description : m_output_descriptions)
    {
        op->m_output_descriptions.push_back(output_description->copy());
    }
    return move(op);
}
