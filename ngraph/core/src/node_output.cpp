//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
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

#include "ngraph/node_output.hpp"
#include "ngraph/log.hpp"
#include "ngraph/node.hpp"

namespace ngraph
{
    Output<Node>::Output(Node* node, size_t index)
        : m_node(node->shared_from_this())
        , m_index(index)
    {
    }

    Output<Node>::Output(const std::shared_ptr<Node>& node, size_t index)
        : m_node(node)
        , m_index(index)
    {
    }

    void Output<Node>::reset()
    {
        m_node.reset();
        m_index = 0;
    }

    Output<Node> Output<Node>::for_node(const std::shared_ptr<Node>& node)
    {
        return Output(node, m_index);
    }
    Node* Output<Node>::get_node() const { return m_node.get(); }
    std::shared_ptr<Node> Output<Node>::get_node_shared_ptr() const { return m_node; }
    size_t Output<Node>::get_index() const { return m_index; }
    descriptor::Tensor& Output<Node>::get_tensor() const
    {
        return m_node->m_outputs.at(m_index).get_tensor();
    }
    std::shared_ptr<descriptor::Tensor> Output<Node>::get_tensor_ptr() const
    {
        return m_node->m_outputs.at(m_index).get_tensor_ptr();
    }
    const element::Type& Output<Node>::get_element_type() const
    {
        return m_node->get_output_element_type(m_index);
    }
    const Shape& Output<Node>::get_shape() const { return m_node->get_output_shape(m_index); }
    const PartialShape& Output<Node>::get_partial_shape() const
    {
        return m_node->get_output_partial_shape(m_index);
    }

    std::set<Input<Node>> Output<Node>::get_target_inputs() const
    {
        std::set<Input<Node>> result;

        for (auto& input : m_node->m_outputs.at(m_index).get_inputs())
        {
            result.emplace(input->get_raw_pointer_node(), input->get_index());
        }

        return result;
    }

    void Output<Node>::remove_target_input(const Input<Node>& target_input) const
    {
        m_node->m_outputs.at(m_index).remove_input(
            &(target_input.get_node()->m_inputs.at(target_input.get_index())));
    }

    void Output<Node>::replace(const Output<Node>& replacement)
    {
        for (auto& input : get_target_inputs())
        {
            input.replace_source_output(replacement);
        }
    }

    bool Output<Node>::operator==(const Output& other) const
    {
        return m_node == other.m_node && m_index == other.m_index;
    }
    bool Output<Node>::operator!=(const Output& other) const { return !(*this == other); }
    bool Output<Node>::operator<(const Output& other) const
    {
        return m_node < other.m_node || (m_node == other.m_node && m_index < other.m_index);
    }
    bool Output<Node>::operator>(const Output& other) const
    {
        return m_node > other.m_node || (m_node == other.m_node && m_index > other.m_index);
    }
    bool Output<Node>::operator<=(const Output& other) const { return !(*this > other); }
    bool Output<Node>::operator>=(const Output& other) const { return !(*this < other); }
    Output<const Node>::Output(const Node* node, size_t index)
        : m_node(node->shared_from_this())
        , m_index(index)
    {
    }

    Output<const Node>::Output(const std::shared_ptr<const Node>& node, size_t index)
        : m_node(node)
        , m_index(index)
    {
    }

    void Output<const Node>::reset()
    {
        m_node.reset();
        m_index = 0;
    }

    Output<const Node> Output<const Node>::for_node(const std::shared_ptr<const Node>& node)
    {
        return Output(node, m_index);
    }

    const Node* Output<const Node>::get_node() const { return m_node.get(); }
    std::shared_ptr<const Node> Output<const Node>::get_node_shared_ptr() const { return m_node; }
    size_t Output<const Node>::get_index() const { return m_index; }
    descriptor::Tensor& Output<const Node>::get_tensor() const
    {
        return m_node->m_outputs.at(m_index).get_tensor();
    }
    std::shared_ptr<descriptor::Tensor> Output<const Node>::get_tensor_ptr() const
    {
        return m_node->m_outputs.at(m_index).get_tensor_ptr();
    }
    const element::Type& Output<const Node>::get_element_type() const
    {
        return m_node->get_output_element_type(m_index);
    }
    const Shape& Output<const Node>::get_shape() const { return m_node->get_output_shape(m_index); }
    const PartialShape& Output<const Node>::get_partial_shape() const
    {
        return m_node->get_output_partial_shape(m_index);
    }

    std::set<Input<Node>> Output<const Node>::get_target_inputs() const
    {
        std::set<Input<Node>> result;

        for (auto& input : m_node->m_outputs.at(m_index).get_inputs())
        {
            result.emplace(input->get_raw_pointer_node(), input->get_index());
        }

        return result;
    }

    bool Output<const Node>::operator==(const Output& other) const
    {
        return m_node == other.m_node && m_index == other.m_index;
    }
    bool Output<const Node>::operator!=(const Output& other) const { return !(*this == other); }
    bool Output<const Node>::operator<(const Output& other) const
    {
        return m_node < other.m_node || (m_node == other.m_node && m_index < other.m_index);
    }
    bool Output<const Node>::operator>(const Output& other) const
    {
        return m_node > other.m_node || (m_node == other.m_node && m_index > other.m_index);
    }
    bool Output<const Node>::operator<=(const Output& other) const { return !(*this > other); }
    bool Output<const Node>::operator>=(const Output& other) const { return !(*this < other); }
    std::ostream& operator<<(std::ostream& out, const Output<Node>& output)
    {
        return output.get_node()->write_description(out, 0)
               << "[" << output.get_index() << "]:" << output.get_element_type()
               << output.get_partial_shape();
    }

    std::ostream& operator<<(std::ostream& out, const Output<const Node>& output)
    {
        return output.get_node()->write_description(out, 0)
               << "[" << output.get_index() << "]:" << output.get_element_type()
               << output.get_partial_shape();
    }
}
