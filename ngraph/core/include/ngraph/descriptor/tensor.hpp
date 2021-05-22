// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <unordered_set>

#include "ngraph/partial_shape.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/type/element_type.hpp"

namespace ngraph
{
    class Node;

    namespace runtime
    {
        class HostTensor;
    }
    using HostTensorPtr = std::shared_ptr<runtime::HostTensor>;
    namespace descriptor
    {
        /// \brief Compile-time descriptor of a first-class value that is a tensor.
        class NGRAPH_API Tensor
        {
            Tensor(const Tensor&) = delete;
            Tensor& operator=(const Tensor&) = delete;

        public:
            Tensor(const element::Type& element_type,
                   const PartialShape& pshape,
                   const std::string& name);
            Tensor(const element::Type& element_type,
                   const PartialShape& pshape,
                   Node* node,
                   size_t node_output_number);

            NGRAPH_DEPRECATED("get_name() is deprecated! Please use get_names() instead.")
            const std::string& get_name() const;
            NGRAPH_DEPRECATED("set_name() is deprecated! Please use set_names() instead.")
            void set_name(const std::string& name);

            const std::unordered_set<std::string>& get_names() const;
            void set_names(const std::unordered_set<std::string>& names);
            void set_tensor_type(const element::Type& element_type, const PartialShape& pshape);
            void set_element_type(const element::Type& elemenet_type);
            void set_partial_shape(const PartialShape& partial_shape);

            /// \brief sets lower bound value description
            void set_lower_value(const HostTensorPtr& value);
            /// \brief sets upper bound value description
            void set_upper_value(const HostTensorPtr& value);
            /// \brief unsets bound value descriptions
            void invalidate_values();

            const element::Type& get_element_type() const { return m_element_type; }
            const Shape& get_shape() const;
            const PartialShape& get_partial_shape() const { return m_partial_shape; }
            /// \brief gets lower bound value description
            HostTensorPtr get_lower_value() const { return m_lower_value; }
            /// \brief gets upper bound value description
            HostTensorPtr get_upper_value() const { return m_upper_value; }
            /// \brief checks if lower and upper bound are set and point to the same HostTensor
            bool has_and_set_bound() const
            {
                return m_upper_value != nullptr && m_upper_value == m_lower_value;
            }
            size_t size() const;

        protected:
            element::Type m_element_type;

            // TODO(amprocte): For now we are maintaining both m_shape and m_partial_shape fields,
            //    with m_shape possibly being invalid (get_shape will throw an exception if it
            //    is). This is because get_shape() returns a const reference. I think ideally we
            //    should refactor so that get_shape returns by value.
            Shape m_shape;
            PartialShape m_partial_shape;
            Node* m_node{nullptr};
            HostTensorPtr m_lower_value, m_upper_value;
            size_t m_node_output_number{0};

            std::string m_name;
            std::unordered_set<std::string> m_names;
        };

        NGRAPH_API
        std::ostream& operator<<(std::ostream&, const ngraph::descriptor::Tensor&);
    } // namespace descriptor
} // namespace ngraph
