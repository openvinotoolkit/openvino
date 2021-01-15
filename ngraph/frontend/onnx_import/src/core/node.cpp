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

#include <onnx/onnx_pb.h>

#include "core/attribute.hpp"
#include "core/graph.hpp"
#include "core/null_node.hpp"
#include "core/tensor.hpp"
#include "onnx_import/core/node.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        class Node::Impl
        {
        public:
            Impl() = delete;

            Impl(const ONNX_NAMESPACE::NodeProto& node_proto, const Graph& graph)
                : m_node_proto{&node_proto}
                , m_name{node_proto.has_name() ? node_proto.name() : ""}
                , m_domain{get_node_domain(node_proto)}
                , m_graph{&graph}
                , m_attributes{std::begin(node_proto.attribute()), std::end(node_proto.attribute())}
                , m_output_names{std::begin(node_proto.output()), std::end(node_proto.output())}
            {
            }

            const std::vector<Attribute>& attributes() const;
            OutputVector get_ng_nodes(const Node& node) const;
            OutputVector get_ng_inputs() const;

            const std::string& domain() const;
            const std::string& op_type() const;
            const std::string& name() const;

            const std::string& description() const;
            const std::vector<std::reference_wrapper<const std::string>>& get_output_names() const;
            const std::string& output(int index) const;
            std::size_t get_outputs_size() const;

            bool has_attribute(const std::string& name) const;

            template <typename T>
            T get_attribute_value(const std::string& name, T default_value) const;

            template <typename T>
            T get_attribute_value(const std::string& name) const;

            const ONNX_NAMESPACE::NodeProto& node_proto() const;
            const Graph& graph() const;

        private:
            const ONNX_NAMESPACE::NodeProto* m_node_proto;
            std::string m_name;
            std::string m_domain;
            const Graph* m_graph;
            std::vector<Attribute> m_attributes;
            std::vector<std::reference_wrapper<const std::string>> m_output_names;
            mutable std::string m_description;
        };

        const ONNX_NAMESPACE::NodeProto& Node::Impl::node_proto() const { return *m_node_proto; }
        const Graph& Node::Impl::graph() const { return *m_graph; }
        const std::vector<Attribute>& Node::Impl::attributes() const { return m_attributes; }
        const std::string& Node::Impl::domain() const { return m_domain; }
        const std::string& Node::Impl::op_type() const { return m_node_proto->op_type(); }
        const std::string& Node::Impl::name() const { return m_name; }
        const std::vector<std::reference_wrapper<const std::string>>&
            Node::Impl::get_output_names() const
        {
            return m_output_names;
        }

        const std::string& Node::Impl::output(int index) const
        {
            return m_node_proto->output(index);
        }

        std::size_t Node::Impl::get_outputs_size() const { return m_output_names.size(); }
        bool Node::Impl::has_attribute(const std::string& name) const
        {
            auto it = std::find_if(
                std::begin(m_attributes), std::end(m_attributes), [&](const Attribute& attribute) {
                    return attribute.get_name() == name;
                });
            return it != std::end(m_attributes);
        }

        template <typename T>
        T Node::Impl::get_attribute_value(const std::string& name, T default_value) const
        {
            auto it = std::find_if(
                std::begin(m_attributes), std::end(m_attributes), [&](const Attribute& attribute) {
                    return attribute.get_name() == name;
                });
            if (it == std::end(m_attributes))
            {
                return std::forward<T>(default_value);
            }
            return it->template get_value<T>();
        }

        template <typename T>
        T Node::Impl::get_attribute_value(const std::string& name) const
        {
            auto it = std::find_if(
                std::begin(m_attributes), std::end(m_attributes), [&](const Attribute& attribute) {
                    return attribute.get_name() == name;
                });
            if (it == std::end(m_attributes))
            {
                throw error::node::UnknownAttribute{this->name(), name};
            }
            return it->template get_value<T>();
        }

        template <>
        Subgraph Node::Impl::get_attribute_value(const std::string& name) const
        {
            auto it = std::find_if(
                std::begin(m_attributes), std::end(m_attributes), [&](const Attribute& attribute) {
                    return attribute.get_name() == name;
                });
            if (it == std::end(m_attributes))
            {
                throw error::node::UnknownAttribute{this->name(), name};
            }
            return it->get_subgraph(graph());
        }

        OutputVector Node::Impl::get_ng_nodes(const Node& node) const
        {
            return m_graph->make_ng_nodes(node);
        }

        OutputVector Node::Impl::get_ng_inputs() const
        {
            OutputVector result;
            for (const auto& name : m_node_proto->input())
            {
                if (!name.empty())
                {
                    result.push_back(m_graph->get_ng_node_from_cache(name));
                }
                else
                {
                    result.push_back(std::make_shared<NullNode>()->output(0));
                }
            }
            return result;
        }

        const std::string& Node::Impl::description() const
        {
            if (m_description.empty())
            {
                if (!name().empty())
                {
                    m_description = name();
                }
                else
                {
                    for (std::size_t index = 0; index < m_output_names.size(); ++index)
                    {
                        m_description += (index != 0 ? ", " : "") + m_output_names.at(index).get();
                    }
                }
            }
            return m_description;
        }

        Node::Node(const ONNX_NAMESPACE::NodeProto& node_proto, const Graph& graph)
            : m_pimpl{new Impl{node_proto, graph}, [](Impl* impl) { delete impl; }}
        {
        }

        Node::Node(Node&& other) noexcept
            : m_pimpl{std::move(other.m_pimpl)}
        {
        }

        Node::Node(const Node& other)
            : m_pimpl{new Impl{other.m_pimpl->node_proto(), other.m_pimpl->graph()},
                      [](Impl* impl) { delete impl; }}
        {
        }

        OutputVector Node::get_ng_inputs() const { return m_pimpl->get_ng_inputs(); }
        OutputVector Node::get_ng_nodes() const { return m_pimpl->get_ng_nodes(*this); }
        const std::string& Node::domain() const { return m_pimpl->domain(); }
        const std::string& Node::op_type() const { return m_pimpl->op_type(); }
        const std::string& Node::get_description() const { return m_pimpl->description(); }
        const std::string& Node::get_name() const { return m_pimpl->name(); }
        const std::vector<std::reference_wrapper<const std::string>>& Node::get_output_names() const
        {
            return m_pimpl->get_output_names();
        }

        const std::string& Node::output(int index) const { return m_pimpl->output(index); }
        std::size_t Node::get_outputs_size() const { return m_pimpl->get_outputs_size(); }
        bool Node::has_attribute(const std::string& name) const
        {
            return m_pimpl->has_attribute(name);
        }

        template <>
        float Node::get_attribute_value(const std::string& name, float default_value) const
        {
            return m_pimpl->template get_attribute_value<float>(name, default_value);
        }

        template <>
        double Node::get_attribute_value(const std::string& name, double default_value) const
        {
            return m_pimpl->template get_attribute_value<double>(name, default_value);
        }

        template <>
        std::int64_t Node::get_attribute_value(const std::string& name,
                                               std::int64_t default_value) const
        {
            return m_pimpl->template get_attribute_value<std::int64_t>(name, default_value);
        }

        template <>
        std::string Node::get_attribute_value(const std::string& name,
                                              std::string default_value) const
        {
            return m_pimpl->template get_attribute_value<std::string>(name,
                                                                      std::move(default_value));
        }

        template <>
        Tensor Node::get_attribute_value(const std::string& name, Tensor default_value) const
        {
            return m_pimpl->template get_attribute_value<Tensor>(name, std::move(default_value));
        }

        template <>
        Graph Node::get_attribute_value(const std::string& name, Graph default_value) const
        {
            return m_pimpl->template get_attribute_value<Graph>(name, std::move(default_value));
        }

        template <>
        std::vector<float> Node::get_attribute_value(const std::string& name,
                                                     std::vector<float> default_value) const
        {
            return m_pimpl->template get_attribute_value<std::vector<float>>(
                name, std::move(default_value));
        }

        template <>
        std::vector<double> Node::get_attribute_value(const std::string& name,
                                                      std::vector<double> default_value) const
        {
            return m_pimpl->template get_attribute_value<std::vector<double>>(
                name, std::move(default_value));
        }

        template <>
        std::vector<std::int64_t>
            Node::get_attribute_value(const std::string& name,
                                      std::vector<std::int64_t> default_value) const
        {
            return m_pimpl->template get_attribute_value<std::vector<std::int64_t>>(
                name, std::move(default_value));
        }

        template <>
        std::vector<std::size_t>
            Node::get_attribute_value(const std::string& name,
                                      std::vector<std::size_t> default_value) const
        {
            return m_pimpl->template get_attribute_value<std::vector<std::size_t>>(
                name, std::move(default_value));
        }

        template <>
        std::vector<std::string>
            Node::get_attribute_value(const std::string& name,
                                      std::vector<std::string> default_value) const
        {
            return m_pimpl->template get_attribute_value<std::vector<std::string>>(
                name, std::move(default_value));
        }

        template <>
        std::vector<Tensor> Node::get_attribute_value(const std::string& name,
                                                      std::vector<Tensor> default_value) const
        {
            return m_pimpl->template get_attribute_value<std::vector<Tensor>>(
                name, std::move(default_value));
        }

        template <>
        std::vector<Graph> Node::get_attribute_value(const std::string& name,
                                                     std::vector<Graph> default_value) const
        {
            return m_pimpl->template get_attribute_value<std::vector<Graph>>(
                name, std::move(default_value));
        }

        template <>
        float Node::get_attribute_value(const std::string& name) const
        {
            return m_pimpl->template get_attribute_value<float>(name);
        }

        template <>
        double Node::get_attribute_value(const std::string& name) const
        {
            return m_pimpl->template get_attribute_value<double>(name);
        }

        template <>
        std::int64_t Node::get_attribute_value(const std::string& name) const
        {
            return m_pimpl->template get_attribute_value<std::int64_t>(name);
        }

        template <>
        std::size_t Node::get_attribute_value(const std::string& name) const
        {
            return m_pimpl->template get_attribute_value<std::size_t>(name);
        }

        template <>
        std::string Node::get_attribute_value(const std::string& name) const
        {
            return m_pimpl->template get_attribute_value<std::string>(name);
        }

        template <>
        Tensor Node::get_attribute_value(const std::string& name) const
        {
            return m_pimpl->template get_attribute_value<Tensor>(name);
        }

        template <>
        Subgraph Node::get_attribute_value(const std::string& name) const
        {
            return m_pimpl->template get_attribute_value<Subgraph>(name);
        }

        template <>
        std::vector<float> Node::get_attribute_value(const std::string& name) const
        {
            return m_pimpl->template get_attribute_value<std::vector<float>>(name);
        }

        template <>
        std::vector<double> Node::get_attribute_value(const std::string& name) const
        {
            return m_pimpl->template get_attribute_value<std::vector<double>>(name);
        }

        template <>
        std::vector<std::int64_t> Node::get_attribute_value(const std::string& name) const
        {
            return m_pimpl->template get_attribute_value<std::vector<std::int64_t>>(name);
        }

        template <>
        std::vector<std::size_t> Node::get_attribute_value(const std::string& name) const
        {
            return m_pimpl->template get_attribute_value<std::vector<std::size_t>>(name);
        }

        template <>
        std::vector<std::string> Node::get_attribute_value(const std::string& name) const
        {
            return m_pimpl->template get_attribute_value<std::vector<std::string>>(name);
        }

        template <>
        std::vector<Tensor> Node::get_attribute_value(const std::string& name) const
        {
            return m_pimpl->template get_attribute_value<std::vector<Tensor>>(name);
        }

        template <>
        std::vector<Graph> Node::get_attribute_value(const std::string& name) const
        {
            return m_pimpl->template get_attribute_value<std::vector<Graph>>(name);
        }

    } // namespace onnx_import

} // namespace ngraph
