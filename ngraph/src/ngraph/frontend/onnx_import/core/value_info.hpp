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

#include <onnx/onnx_pb.h>

#include "default_opset.hpp"
#include "ngraph/partial_shape.hpp"
#include "ngraph/type/element_type.hpp"
#include "node.hpp"
#include "tensor.hpp"
#include "utils/common.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace error
        {
            namespace value_info
            {
                struct unspecified_element_type : ngraph_error
                {
                    unspecified_element_type()
                        : ngraph_error{"value info has no element type specified"}
                    {
                    }
                };
            } // namespace value_info
        }     // namespace error

        class ValueInfo
        {
        public:
            ValueInfo(ValueInfo&&) = default;
            ValueInfo(const ValueInfo&) = default;

            ValueInfo() = delete;
            explicit ValueInfo(const ONNX_NAMESPACE::ValueInfoProto& value_info_proto)
                : m_value_info_proto{&value_info_proto}
            {
                if (value_info_proto.type().has_tensor_type())
                {
                    const auto& onnx_tensor = value_info_proto.type().tensor_type();

                    if (onnx_tensor.has_shape())
                    {
                        m_partial_shape = to_ng_shape(onnx_tensor.shape());
                    }
                    else
                    {
                        m_partial_shape = PartialShape::dynamic();
                    }
                }
            }

            ValueInfo& operator=(const ValueInfo&) = delete;
            ValueInfo& operator=(ValueInfo&&) = delete;

            const std::string& get_name() const { return m_value_info_proto->name(); }
            const PartialShape& get_shape() const { return m_partial_shape; }
            const element::Type& get_element_type() const
            {
                if (!m_value_info_proto->type().tensor_type().has_elem_type())
                {
                    throw error::value_info::unspecified_element_type{};
                }
                return common::get_ngraph_element_type(
                    m_value_info_proto->type().tensor_type().elem_type());
            }

            std::shared_ptr<ngraph::Node>
                get_ng_node(ParameterVector& parameters,
                            const std::map<std::string, Tensor>& initializers) const
            {
                const auto it = initializers.find(get_name());
                if (it != std::end(initializers))
                {
                    return get_ng_constant(it->second);
                }
                parameters.push_back(get_ng_parameter());
                return parameters.back();
            }

        protected:
            std::shared_ptr<op::Parameter> get_ng_parameter() const
            {
                auto parameter = std::make_shared<op::Parameter>(get_element_type(), get_shape());
                parameter->set_friendly_name(get_name());
                return parameter;
            }

            std::shared_ptr<op::Constant> get_ng_constant(const Tensor& tensor) const
            {
                return tensor.get_ng_constant();
            }

            PartialShape to_ng_shape(const ONNX_NAMESPACE::TensorShapeProto& onnx_shape) const
            {
                if (onnx_shape.dim_size() == 0)
                {
                    return Shape{}; // empty list of dimensions denotes a scalar
                }

                std::vector<Dimension> dims;
                for (const auto& onnx_dim : onnx_shape.dim())
                {
                    if (onnx_dim.has_dim_value())
                    {
                        dims.emplace_back(onnx_dim.dim_value());
                    }
                    else if (onnx_dim.has_dim_param())
                    {
                        dims.push_back(Dimension::dynamic());
                    }
                }
                return PartialShape{dims};
            }

        private:
            const ONNX_NAMESPACE::ValueInfoProto* m_value_info_proto;
            PartialShape m_partial_shape;
        };

        inline std::ostream& operator<<(std::ostream& outs, const ValueInfo& info)
        {
            return (outs << "<ValueInfo: " << info.get_name() << ">");
        }

    } // namespace onnx_import

} // namespace ngraph
