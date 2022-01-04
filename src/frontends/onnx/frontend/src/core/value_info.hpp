// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <onnx/onnx_pb.h>

#include "core/tensor.hpp"
#include "default_opset.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "onnx_common/utils.hpp"
#include "onnx_import/core/node.hpp"
#include "utils/common.hpp"

namespace ov {
namespace onnx_import {
class ValueInfo {
public:
    ValueInfo(ValueInfo&&) = default;
    ValueInfo(const ValueInfo&) = default;

    ValueInfo() = delete;
    explicit ValueInfo(const ONNX_NAMESPACE::ValueInfoProto& value_info_proto) : m_value_info_proto{&value_info_proto} {
        if (value_info_proto.type().has_tensor_type()) {
            const auto& onnx_tensor = value_info_proto.type().tensor_type();

            if (onnx_tensor.has_shape()) {
                m_partial_shape = onnx_common::to_ng_shape(onnx_tensor.shape());
            }
        }
    }

    ValueInfo& operator=(const ValueInfo&) = delete;
    ValueInfo& operator=(ValueInfo&&) = delete;

    const std::string& get_name() const {
        return m_value_info_proto->name();
    }
    const PartialShape& get_shape() const {
        return m_partial_shape;
    }
    const element::Type& get_element_type() const {
        if (m_value_info_proto->type().tensor_type().has_elem_type()) {
            return common::get_ov_element_type(m_value_info_proto->type().tensor_type().elem_type());
        }
        return element::dynamic;
    }

    std::shared_ptr<ov::Node> get_ng_node(ParameterVector& parameters,
                                              const std::map<std::string, Tensor>& initializers) const {
        const auto it = initializers.find(get_name());
        if (it != std::end(initializers)) {
            return get_ng_constant(it->second);
        }
        parameters.push_back(get_ng_parameter());
        return parameters.back();
    }

protected:
    std::shared_ptr<op::v0::Parameter> get_ng_parameter() const {
        auto parameter = std::make_shared<op::v0::Parameter>(get_element_type(), get_shape());
        parameter->set_friendly_name(get_name());
        parameter->get_output_tensor(0).set_names({get_name()});
        return parameter;
    }

    std::shared_ptr<default_opset::Constant> get_ng_constant(const Tensor& tensor) const {
        return tensor.get_ng_constant();
    }

private:
    const ONNX_NAMESPACE::ValueInfoProto* m_value_info_proto;
    PartialShape m_partial_shape = PartialShape::dynamic();
};

inline std::ostream& operator<<(std::ostream& outs, const ValueInfo& info) {
    return (outs << "<ValueInfo: " << info.get_name() << ">");
}

}  // namespace onnx_import

}  // namespace ov
