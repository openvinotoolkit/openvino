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

#include <string>

#include "ngraph/check.hpp"
#include "ngraph/except.hpp"
#include "tensor_iterator_builder.hpp"

util::TensorIteratorBuilder::TensorIteratorBuilder(const ngraph::NodeVector& arguments,
                                                   const py::dict& attributes)
    : m_arguments(arguments)
    , m_attributes(attributes)
{
    get_graph_body();
    // Set-up TI inputs.
    NGRAPH_CHECK(m_attributes.contains("slice_input_desc"),
                 "The required \"slice_input_desc\" attribute is missing. Can't build "
                 "TensorIterator operator.");
    m_slice_input_desc = m_attributes["slice_input_desc"].cast<py::list>();

    if (m_attributes.contains("merged_input_desc"))
    {
        m_merged_input_desc = m_attributes["merged_input_desc"].cast<py::list>();
    }

    if (m_attributes.contains("invariant_input_desc"))
    {
        m_invariant_input_desc = m_attributes["invariant_input_desc"].cast<py::list>();
    }

    if (m_attributes.contains("body_output_desc"))
    {
        py::list body_output_desc = m_attributes["body_output_desc"].cast<py::list>();
        for (py::handle h : body_output_desc)
        {
            py::dict desc = h.cast<py::dict>();
            desc["type"] = "BodyOutputDesc";
            check_attribute(desc, "output_idx", "BodyOutputDesc");
            m_outputs.emplace(desc["output_idx"].cast<int64_t>(), desc);
        }
    }
    if (m_attributes.contains("concat_output_desc"))
    {
        py::list concat_output_desc = m_attributes["concat_output_desc"].cast<py::list>();
        for (py::handle h : concat_output_desc)
        {
            py::dict desc = h.cast<py::dict>();
            desc["type"] = "ConcatOutputDesc";
            check_attribute(desc, "output_idx", "ConcatOutputDesc");
            m_outputs.emplace(desc["output_idx"].cast<int64_t>(), desc);
        }
    }
}

std::shared_ptr<ngraph::op::TensorIterator>
    util::TensorIteratorBuilder::configure(std::shared_ptr<ngraph::op::TensorIterator>&& ti_node)
{
    ti_node->set_body(m_body);
    set_tensor_iterator_sliced_inputs(ti_node);
    set_tensor_iterator_merged_inputs(ti_node);
    set_tensor_iterator_invariant_inputs(ti_node);
    set_tensor_iterator_outputs(ti_node);
    ti_node->constructor_validate_and_infer_types();

    return ti_node;
}

void util::TensorIteratorBuilder::check_attribute(const py::dict& attrs,
                                                  std::string attr_name,
                                                  std::string desc_name) const
{
    NGRAPH_CHECK(attrs.contains(attr_name),
                 "The required \"",
                 attr_name,
                 "\" attribute is missing. Can't build TensorIterator's ",
                 desc_name,
                 ".");
}

void util::TensorIteratorBuilder::get_graph_body()
{
    NGRAPH_CHECK(m_attributes.contains("body"),
                 "The required \"body\" attribute is missing. Can't build TensorIterator "
                 "operator.");

    const py::dict& body_attrs = m_attributes["body"].cast<py::dict>();

    NGRAPH_CHECK(body_attrs.contains("parameters"),
                 "The required body's \"parameters\" "
                 "attribute is missing. Can't build TensorIterator's body.");
    NGRAPH_CHECK(body_attrs.contains("results"),
                 "The required body's \"results\" "
                 "attribute is missing. Can't build TensorIterator's body.");

    m_body_outputs = as_output_vector(body_attrs["results"].cast<ngraph::NodeVector>());
    m_body_parameters = body_attrs["parameters"].cast<ngraph::ParameterVector>();
    m_body =
        std::make_shared<ngraph::op::TensorIterator::BodyLambda>(m_body_outputs, m_body_parameters);
}

void util::TensorIteratorBuilder::set_tensor_iterator_sliced_inputs(
    std::shared_ptr<ngraph::op::TensorIterator>& ti_node) const
{
    for (py::handle h : m_slice_input_desc)
    {
        const py::dict& desc = h.cast<py::dict>();
        check_attribute(desc, "input_idx", "SliceInputDesc");
        check_attribute(desc, "body_parameter_idx", "SliceInputDesc");
        check_attribute(desc, "start", "SliceInputDesc");
        check_attribute(desc, "stride", "SliceInputDesc");
        check_attribute(desc, "part_size", "SliceInputDesc");
        check_attribute(desc, "end", "SliceInputDesc");
        check_attribute(desc, "axis", "SliceInputDesc");

        ti_node->set_sliced_input(m_body_parameters.at(desc["body_parameter_idx"].cast<int64_t>()),
                                  m_arguments.at(desc["input_idx"].cast<int64_t>()),
                                  desc["start"].cast<int64_t>(),
                                  desc["stride"].cast<int64_t>(),
                                  desc["part_size"].cast<int64_t>(),
                                  desc["end"].cast<int64_t>(),
                                  desc["axis"].cast<int64_t>());
    }
}

void util::TensorIteratorBuilder::set_tensor_iterator_merged_inputs(
    std::shared_ptr<ngraph::op::TensorIterator>& ti_node) const
{
    for (py::handle h : m_merged_input_desc)
    {
        const py::dict& desc = h.cast<py::dict>();
        check_attribute(desc, "input_idx", "MergedInputDesc");
        check_attribute(desc, "body_parameter_idx", "MergedInputDesc");
        check_attribute(desc, "body_value_idx", "MergedInputDesc");

        ti_node->set_merged_input(m_body_parameters.at(desc["body_parameter_idx"].cast<int64_t>()),
                                  m_arguments.at(desc["input_idx"].cast<int64_t>()),
                                  m_body_outputs.at(desc["body_value_idx"].cast<int64_t>()));
    }
}

void util::TensorIteratorBuilder::set_tensor_iterator_invariant_inputs(
    std::shared_ptr<ngraph::op::TensorIterator>& ti_node) const
{
    for (py::handle h : m_invariant_input_desc)
    {
        const py::dict& desc = h.cast<py::dict>();
        check_attribute(desc, "input_idx", "InvariantInputDesc");
        check_attribute(desc, "body_parameter_idx", "InvariantInputDesc");

        ti_node->set_invariant_input(
            m_body_parameters.at(desc["body_parameter_idx"].cast<int64_t>()),
            m_arguments.at(desc["input_idx"].cast<int64_t>()));
    }
}

void util::TensorIteratorBuilder::set_tensor_iterator_outputs(
    std::shared_ptr<ngraph::op::TensorIterator>& ti_node) const
{
    for (const auto& elem : m_outputs)
    {
        const py::dict& desc = elem.second.cast<py::dict>();
        if (desc["type"].cast<std::string>() == "BodyOutputDesc")
        {
            set_tensor_iterator_body_output(desc, ti_node);
        }
        else if (desc["type"].cast<std::string>() == "ConcatOutputDesc")
        {
            set_tensor_iterator_concatenated_body_output(desc, ti_node);
        }
        else
        {
            throw ngraph::ngraph_error("Unrecognized TensorIterator output type.");
        }
    }
}

void util::TensorIteratorBuilder::set_tensor_iterator_body_output(
    const py::dict& desc, std::shared_ptr<ngraph::op::TensorIterator>& ti_node) const
{
    check_attribute(desc, "body_value_idx", "BodyOutputDesc");
    check_attribute(desc, "iteration", "BodyOutputDesc");

    NGRAPH_CHECK(desc["output_idx"].cast<size_t>() == ti_node->get_output_size(),
                 "Descriptor output idx value is different from currently configured "
                 "TensorIterator output.");

    ti_node->get_iter_value(m_body_outputs.at(desc["body_value_idx"].cast<int64_t>()),
                            desc["iteration"].cast<int64_t>());
}

void util::TensorIteratorBuilder::set_tensor_iterator_concatenated_body_output(
    const py::dict& desc, std::shared_ptr<ngraph::op::TensorIterator>& ti_node) const
{
    check_attribute(desc, "body_value_idx", "ConcatOutputDesc");
    check_attribute(desc, "start", "ConcatOutputDesc");
    check_attribute(desc, "stride", "ConcatOutputDesc");
    check_attribute(desc, "part_size", "ConcatOutputDesc");
    check_attribute(desc, "end", "ConcatOutputDesc");
    check_attribute(desc, "axis", "ConcatOutputDesc");

    NGRAPH_CHECK(desc["output_idx"].cast<size_t>() == ti_node->get_output_size(),
                 "Descriptor output idx value is different from currently configured "
                 "TensorIterator output.");

    ti_node->get_concatenated_slices(m_body_outputs.at(desc["body_value_idx"].cast<int64_t>()),
                                     desc["start"].cast<int64_t>(),
                                     desc["stride"].cast<int64_t>(),
                                     desc["part_size"].cast<int64_t>(),
                                     desc["end"].cast<int64_t>(),
                                     desc["axis"].cast<int64_t>());
}
