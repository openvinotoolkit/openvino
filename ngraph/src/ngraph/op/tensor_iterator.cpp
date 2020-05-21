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
#include "ngraph/factory.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/pass/get_output_element_elimination.hpp"
#include "ngraph/specialize_function.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::v0::TensorIterator::type_info;

constexpr DiscreteTypeInfo op::v0::TensorIterator::SliceInputDescription::type_info;
constexpr DiscreteTypeInfo op::v0::TensorIterator::MergedInputDescription::type_info;
constexpr DiscreteTypeInfo op::v0::TensorIterator::InvariantInputDescription::type_info;

constexpr DiscreteTypeInfo op::v0::TensorIterator::BodyOutputDescription::type_info;
constexpr DiscreteTypeInfo op::v0::TensorIterator::ConcatOutputDescription::type_info;

constexpr DiscreteTypeInfo op::v0::TensorIterator::BodyLambda::type_info;

bool op::v0::TensorIterator::BodyLambda::visit_attributes(AttributeVisitor& visitor)
{
    return true;
}

op::v0::TensorIterator::TensorIterator(const OutputVector& values)
    : op::util::FusedOp(values)
{
}

op::v0::TensorIterator::InputDescription::InputDescription(uint64_t input_index,
                                                           uint64_t body_parameter_index)
    : m_input_index(input_index)
    , m_body_parameter_index(body_parameter_index)
{
}

bool op::v0::TensorIterator::InputDescription::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("input_index", m_input_index);
    visitor.on_attribute("body_parameter_index", m_body_parameter_index);
    return true;
}

op::v0::TensorIterator::SliceInputDescription::SliceInputDescription(uint64_t input_index,
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

shared_ptr<op::v0::TensorIterator::InputDescription>
    op::v0::TensorIterator::SliceInputDescription::copy() const
{
    return make_shared<SliceInputDescription>(
        m_input_index, m_body_parameter_index, m_start, m_stride, m_part_size, m_end, m_axis);
}

bool op::v0::TensorIterator::SliceInputDescription::visit_attributes(AttributeVisitor& visitor)
{
    InputDescription::visit_attributes(visitor);
    visitor.on_attribute("start", m_start);
    visitor.on_attribute("stride", m_stride);
    visitor.on_attribute("part_size", m_part_size);
    visitor.on_attribute("end", m_end);
    visitor.on_attribute("axis", m_axis);
    return true;
}

op::v0::TensorIterator::MergedInputDescription::MergedInputDescription(
    uint64_t input_index, uint64_t body_parameter_index, uint64_t body_value_index)
    : InputDescription(input_index, body_parameter_index)
    , m_body_value_index(body_value_index)
{
}

shared_ptr<op::v0::TensorIterator::InputDescription>
    op::v0::TensorIterator::MergedInputDescription::copy() const
{
    return make_shared<MergedInputDescription>(
        m_input_index, m_body_parameter_index, m_body_value_index);
}

bool op::v0::TensorIterator::MergedInputDescription::visit_attributes(AttributeVisitor& visitor)
{
    InputDescription::visit_attributes(visitor);
    visitor.on_attribute("body_value_index", m_body_value_index);
    return true;
}

op::v0::TensorIterator::InvariantInputDescription::InvariantInputDescription(
    uint64_t input_index, uint64_t body_parameter_index)
    : InputDescription(input_index, body_parameter_index)
{
}

shared_ptr<op::v0::TensorIterator::InputDescription>
    op::v0::TensorIterator::InvariantInputDescription::copy() const
{
    return make_shared<InvariantInputDescription>(m_input_index, m_body_parameter_index);
}

bool op::v0::TensorIterator::InvariantInputDescription::visit_attributes(AttributeVisitor& visitor)
{
    InputDescription::visit_attributes(visitor);
    return true;
}

op::v0::TensorIterator::OutputDescription::OutputDescription(uint64_t body_value_index,
                                                             uint64_t output_index)
    : m_body_value_index(body_value_index)
    , m_output_index(output_index)
{
}

bool op::v0::TensorIterator::OutputDescription::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("body_value_index", m_body_value_index);
    visitor.on_attribute("output_index", m_output_index);
    return true;
}

op::v0::TensorIterator::ConcatOutputDescription::ConcatOutputDescription(uint64_t body_value_index,
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

bool op::v0::TensorIterator::ConcatOutputDescription::visit_attributes(AttributeVisitor& visitor)
{
    OutputDescription::visit_attributes(visitor);
    visitor.on_attribute("start", m_start);
    visitor.on_attribute("stride", m_stride);
    visitor.on_attribute("part_size", m_part_size);
    visitor.on_attribute("end", m_end);
    visitor.on_attribute("axis", m_axis);
    return true;
}

shared_ptr<op::v0::TensorIterator::OutputDescription>
    op::v0::TensorIterator::ConcatOutputDescription::copy() const
{
    return make_shared<ConcatOutputDescription>(
        m_body_value_index, m_output_index, m_start, m_stride, m_part_size, m_end, m_axis);
}

op::v0::TensorIterator::BodyOutputDescription::BodyOutputDescription(uint64_t body_value_index,
                                                                     uint64_t output_index,
                                                                     int64_t iteration)
    : OutputDescription(body_value_index, output_index)
    , m_iteration(iteration)
{
}

shared_ptr<op::v0::TensorIterator::OutputDescription>
    op::v0::TensorIterator::BodyOutputDescription::copy() const
{
    return make_shared<BodyOutputDescription>(m_body_value_index, m_output_index, m_iteration);
}

bool op::v0::TensorIterator::BodyOutputDescription::visit_attributes(AttributeVisitor& visitor)
{
    OutputDescription::visit_attributes(visitor);
    visitor.on_attribute("iteration", m_iteration);
    return true;
}

namespace
{
}

namespace ngraph
{
    template <>
    FactoryRegistry<op::v0::TensorIterator::InputDescription>&
        FactoryRegistry<op::v0::TensorIterator::InputDescription>::get()
    {
        static FactoryRegistry<op::v0::TensorIterator::InputDescription> registry;
        static mutex init_guard;
        if (registry.m_factory_map.size() == 0)
        {
            lock_guard<mutex> guard(init_guard);
            if (registry.m_factory_map.size() == 0)
            {
                registry.register_factory<op::v0::TensorIterator::SliceInputDescription>();
                registry.register_factory<op::v0::TensorIterator::MergedInputDescription>();
                registry.register_factory<op::v0::TensorIterator::InvariantInputDescription>();
            }
        }
        return registry;
    }

    constexpr DiscreteTypeInfo
        AttributeAdapter<std::shared_ptr<op::TensorIterator::InputDescription>>::type_info;

    constexpr DiscreteTypeInfo AttributeAdapter<
        std::vector<std::shared_ptr<op::TensorIterator::InputDescription>>>::type_info;

    AttributeAdapter<std::vector<std::shared_ptr<op::TensorIterator::InputDescription>>>::
        AttributeAdapter(std::vector<std::shared_ptr<op::TensorIterator::InputDescription>>& ref)
        : m_ref(ref)
    {
    }

    bool AttributeAdapter<std::vector<std::shared_ptr<op::TensorIterator::InputDescription>>>::
        visit_attributes(AttributeVisitor& visitor)
    {
        int64_t size = m_ref.size();
        visitor.on_attribute("size", size);
        if (size != m_ref.size())
        {
            m_ref.resize(size);
        }
        ostringstream index;
        for (int64_t i = 0; i < size; i++)
        {
            index.str("");
            index << i;
            visitor.on_attribute(index.str(), m_ref[i]);
        }
        return true;
    }

    template <>
    FactoryRegistry<op::v0::TensorIterator::OutputDescription>&
        FactoryRegistry<op::v0::TensorIterator::OutputDescription>::get()
    {
        static FactoryRegistry<op::v0::TensorIterator::OutputDescription> registry;
        static mutex init_guard;
        // TODO: Add a lock
        if (registry.m_factory_map.size() == 0)
        {
            lock_guard<mutex> guard(init_guard);
            if (registry.m_factory_map.size() == 0)
            {
                registry.register_factory<op::v0::TensorIterator::ConcatOutputDescription>();
                registry.register_factory<op::v0::TensorIterator::BodyOutputDescription>();
            }
        }
        return registry;
    }

    constexpr DiscreteTypeInfo AttributeAdapter<
        std::vector<std::shared_ptr<op::TensorIterator::OutputDescription>>>::type_info;

    constexpr DiscreteTypeInfo
        AttributeAdapter<std::shared_ptr<op::TensorIterator::OutputDescription>>::type_info;

    AttributeAdapter<std::vector<std::shared_ptr<op::TensorIterator::OutputDescription>>>::
        AttributeAdapter(std::vector<std::shared_ptr<op::TensorIterator::OutputDescription>>& ref)
        : m_ref(ref)
    {
    }

    bool AttributeAdapter<std::vector<std::shared_ptr<op::TensorIterator::OutputDescription>>>::
        visit_attributes(AttributeVisitor& visitor)
    {
        int64_t size = m_ref.size();
        visitor.on_attribute("size", size);
        if (size != m_ref.size())
        {
            m_ref.resize(size);
        }
        ostringstream index;
        for (int64_t i = 0; i < size; i++)
        {
            index.str("");
            index << i;
            visitor.on_attribute(index.str(), m_ref[i]);
        }
        return true;
    }
}

bool op::v0::TensorIterator::visit_attributes(AttributeVisitor& visitor)
{
    if (!m_body)
    {
        m_body = make_shared<BodyLambda>();
    }
    shared_ptr<Lambda> lambda = m_body;
    visitor.on_attribute("body", lambda);
    visitor.on_attribute("input_descriptions", m_input_descriptions);
    visitor.on_attribute("output_descriptions", m_output_descriptions);

    return false;
}

Input<Node> op::v0::TensorIterator::input_for_value(const Output<Node>& value)
{
    auto input_index = get_input_size();
    set_argument(input_index, value);
    return Input<Node>(this, input_index);
}

void op::v0::TensorIterator::set_sliced_input(const std::shared_ptr<op::Parameter>& body_parameter,
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

void op::v0::TensorIterator::set_merged_input(const std::shared_ptr<Parameter>& body_parameter,
                                              const Output<Node>& initial_value,
                                              const Output<Node>& successive_value)
{
    m_input_descriptions.push_back(
        make_shared<MergedInputDescription>(input_for_value(initial_value).get_index(),
                                            m_body->get_parameter_index(body_parameter),
                                            m_body->get_result_index(successive_value)));
}

void op::v0::TensorIterator::set_invariant_input(const std::shared_ptr<Parameter>& body_parameter,
                                                 const Output<Node>& value)
{
    m_input_descriptions.push_back(make_shared<InvariantInputDescription>(
        input_for_value(value).get_index(), m_body->get_parameter_index(body_parameter)));
}

Output<Node> op::v0::TensorIterator::get_iter_value(const Output<Node>& body_value,
                                                    int64_t iteration)
{
    auto output_index = get_output_size();
    m_output_descriptions.push_back(make_shared<BodyOutputDescription>(
        m_body->get_result_index(body_value), output_index, iteration));
    set_output_size(output_index + 1);
    return Output<Node>(shared_from_this(), output_index);
}

Output<Node> op::v0::TensorIterator::get_concatenated_slices(const Output<Node>& body_value,
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

NodeVector op::v0::TensorIterator::decompose_op() const
{
    // Stub
    return NodeVector{};
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

std::shared_ptr<Node>
    op::v0::TensorIterator::clone_with_new_inputs(const OutputVector& new_args) const
{
    auto op = make_shared<op::v0::TensorIterator>(new_args);
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

namespace ngraph
{
}
