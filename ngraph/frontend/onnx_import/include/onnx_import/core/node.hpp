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

#pragma once

#include <cstddef>
#include <string>

#include "ngraph/except.hpp"
#include "ngraph/node.hpp"
#include "onnx_import/utils/onnx_importer_visibility.hpp"

namespace ONNX_NAMESPACE
{
    // forward declaration
    class NodeProto;
}

namespace ngraph
{
    namespace onnx_import
    {
        namespace error
        {
            namespace node
            {
                struct UnknownAttribute : ngraph_error
                {
                    explicit UnknownAttribute(const std::string& node, const std::string& name)
                        : ngraph_error{"Node (" + node + "): unknown attribute \'" + name + "\'"}
                    {
                    }
                };

            } // namespace node

        } // namespace error

        // forward declaration
        class Graph;

        class ONNX_IMPORTER_API Node
        {
        public:
            Node() = delete;
            Node(const ONNX_NAMESPACE::NodeProto& node_proto, const Graph& graph);

            Node(Node&&) noexcept;
            Node(const Node&);

            Node& operator=(Node&&) noexcept = delete;
            Node& operator=(const Node&) = delete;

            OutputVector get_ng_inputs() const;
            OutputVector get_ng_nodes() const;
            const std::string& domain() const;
            const std::string& op_type() const;
            const std::string& get_name() const;

            /// \brief Describe the ONNX Node to make debugging graphs easier
            /// Function will return the Node's name if it has one, or the names of its outputs.
            /// \return Description of Node
            const std::string& get_description() const;

            const std::vector<std::reference_wrapper<const std::string>>& get_output_names() const;
            const std::string& output(int index) const;
            std::size_t get_outputs_size() const;

            bool has_attribute(const std::string& name) const;

            template <typename T>
            T get_attribute_value(const std::string& name, T default_value) const;

            template <typename T>
            T get_attribute_value(const std::string& name) const;

        private:
            class Impl;
            // In this case we need custom deleter, because Impl is an incomplete
            // type. Node's are elements of std::vector. Without custom deleter
            // compilation fails; the compiler is unable to parameterize an allocator's
            // default deleter due to incomple type.
            std::unique_ptr<Impl, void (*)(Impl*)> m_pimpl;
        };

        inline std::ostream& operator<<(std::ostream& outs, const Node& node)
        {
            return (outs << "<Node(" << node.op_type() << "): " << node.get_description() << ">");
        }

    } // namespace onnx_import

} // namespace ngraph
