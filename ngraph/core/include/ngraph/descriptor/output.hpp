// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "ngraph/descriptor/input.hpp"
#include "ngraph/descriptor/tensor.hpp"
#include "ngraph/node_output.hpp"

namespace ngraph
{
    // The forward declaration of Node is needed here because Node has a deque of
    // Outputs, and Output is an incomplete type at this point. STL containers of
    // incomplete type have undefined behavior according to the C++11 standard, and
    // in practice including node.hpp here was causing compilation errors on some
    // systems (namely macOS).
    class Node;

    class Variant;

    namespace descriptor
    {
        // Describes an output tensor of an op
        class NGRAPH_API Output
        {
        public:
            Output()
                : m_node(nullptr)
                , m_index(0)
                , m_tensor(nullptr)
                , m_inputs()
            {
            }

            /// \param node Node that owns this output.
            /// \param index Position of the output tensor in all output tensors
            /// \param tensor The tensor where the value will be written
            Output(Node* node, size_t index, const std::shared_ptr<Tensor>& tensor);

            std::shared_ptr<Node> get_node() const;
            size_t get_index() const { return m_index; }
            ngraph::Output<Node> get_output() const;
            std::shared_ptr<Tensor> get_tensor_ptr() const { return m_tensor; }
            void set_tensor_ptr(const std::shared_ptr<Tensor>& tensor) { m_tensor = tensor; }
            void add_input(Input* input);
            void remove_input(Input* input);
            const std::vector<Input*>& get_inputs() const { return m_inputs; }
            Tensor& get_tensor() const;

            using RTMap = std::map<std::string, std::shared_ptr<Variant>>;

            RTMap& get_rt_info() { return m_rt_info; }
            const RTMap& get_rt_info() const { return m_rt_info; }
            /// \return the shape of the output
            const Shape& get_shape() const;

            /// \return the partial shape of the output
            const PartialShape& get_partial_shape() const;

            /// \return the element type of the output
            const element::Type& get_element_type() const;

            Output(const Output&) = default;
            Output(Output&&) = default;
            Output& operator=(const Output&) = default;

        protected:
            Node* m_node;
            size_t m_index;
            std::shared_ptr<Tensor> m_tensor;
            RTMap m_rt_info;
            std::vector<Input*> m_inputs;
        };
    } // namespace descriptor
} // namespace ngraph
