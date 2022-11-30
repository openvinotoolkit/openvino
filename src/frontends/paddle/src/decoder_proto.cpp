// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "decoder_proto.hpp"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "framework.pb.h"
#include "openvino/util/log.hpp"

namespace ov {
namespace frontend {
namespace paddle {

using namespace ::paddle::framework;

std::map<proto::VarType_Type, ov::element::Type> TYPE_MAP{
    {proto::VarType_Type::VarType_Type_BOOL, ov::element::boolean},
    {proto::VarType_Type::VarType_Type_INT16, ov::element::i16},
    {proto::VarType_Type::VarType_Type_INT32, ov::element::i32},
    {proto::VarType_Type::VarType_Type_INT64, ov::element::i64},
    {proto::VarType_Type::VarType_Type_FP16, ov::element::f16},
    {proto::VarType_Type::VarType_Type_FP32, ov::element::f32},
    {proto::VarType_Type::VarType_Type_FP64, ov::element::f64},
    {proto::VarType_Type::VarType_Type_UINT8, ov::element::u8},
    {proto::VarType_Type::VarType_Type_INT8, ov::element::i8},
    {proto::VarType_Type::VarType_Type_BF16, ov::element::bf16}};

ov::Any DecoderProto::get_attribute(const std::string& name) const {
    auto attrs = decode_attribute_helper(name);
    if (attrs.empty()) {
        return {};
    }

    switch (attrs[0].type()) {
    case proto::AttrType::INT:
        return attrs[0].i();
    case proto::AttrType::INTS:
        return std::vector<int32_t>(attrs[0].ints().begin(), attrs[0].ints().end());
    case proto::AttrType::FLOAT:
        return attrs[0].f();
    case proto::AttrType::FLOATS:
        return std::vector<float>(attrs[0].floats().begin(), attrs[0].floats().end());
    case proto::AttrType::STRING:
        return attrs[0].s();
    case proto::AttrType::STRINGS:
        return std::vector<std::string>(attrs[0].strings().begin(), attrs[0].strings().end());
    case proto::AttrType::LONG:
        return attrs[0].l();
    case proto::AttrType::LONGS:
        return std::vector<int64_t>(attrs[0].longs().begin(), attrs[0].longs().end());
    case proto::AttrType::BOOLEAN:
        return attrs[0].b();
    case proto::AttrType::BOOLEANS:
        return std::vector<bool>(attrs[0].bools().begin(), attrs[0].bools().end());
    case proto::AttrType::BLOCK:
        return attrs[0].block_idx();
    case proto::AttrType::BLOCKS:
        return std::vector<std::int32_t>(attrs[0].blocks_idx().begin(), attrs[0].blocks_idx().end());
    default:
        FRONT_END_GENERAL_CHECK(false, "Conversion from PaddlePaddle to OpenVINO data type is not supported.");
    }
}

ov::Any DecoderProto::convert_attribute(const Any& data, const std::type_info& type_info) const {
    if (data.is<int32_t>() && type_info == typeid(ov::element::Type)) {
        return TYPE_MAP.at(static_cast<proto::VarType_Type>(data.as<int32_t>()));
    } else if (data.is<std::vector<int32_t>>() && type_info == typeid(std::vector<ov::element::Type>)) {
        const auto& casted = data.as<std::vector<int32_t>>();
        std::vector<ov::element::Type> types(casted.size());
        for (size_t i = 0; i < casted.size(); ++i) {
            types[i] = TYPE_MAP.at(static_cast<proto::VarType_Type>(casted[i]));
        }
        return types;
    }
    // no conversion rules found.
    return data;
}

/// \brief Get the output port names
std::vector<paddle::OutPortName> DecoderProto::get_output_names() const {
    std::vector<std::string> output_names;
    for (const auto& output : m_op_desc.outputs()) {
        output_names.push_back(output.parameter());
    }
    return output_names;
}

const paddle::OutPortName& DecoderProto::get_output_names(size_t idx) const {
    return m_op_desc.outputs()[idx].parameter();
}

/// \brief Get the input port names
std::vector<InPortName> DecoderProto::get_input_names() const {
    std::vector<std::string> input_names;
    for (const auto& input : m_op_desc.inputs()) {
        input_names.push_back(input.parameter());
    }
    return input_names;
}

const InPortName& DecoderProto::get_input_names(size_t idx) const {
    return m_op_desc.inputs()[idx].parameter();
}

/// \brief Get the output tensor names
std::vector<paddle::TensorName> DecoderProto::get_output_var_names(const std::string& var_name) const {
    std::vector<std::string> output_names;
    for (const auto& output : m_op_desc.outputs()) {
        if (output.parameter() == var_name) {
            for (const auto& var : output.arguments()) {
                output_names.push_back(var);
            }
        }
    }
    return output_names;
}

std::vector<paddle::TensorName> DecoderProto::get_output_var_names() const {
    std::vector<std::string> output_names;
    for (const auto& output : m_op_desc.outputs()) {
        for (const auto& var : output.arguments()) {
            output_names.push_back(var);
        }
    }
    return output_names;
}

/// \brief Get the input tensor names
std::vector<paddle::TensorName> DecoderProto::get_input_var_names(const std::string& var_name) const {
    std::vector<std::string> input_names;
    for (const auto& input : m_op_desc.inputs()) {
        if (input.parameter() == var_name) {
            for (const auto& var : input.arguments()) {
                input_names.push_back(var);
            }
        }
    }
    return input_names;
}

std::vector<paddle::TensorName> DecoderProto::get_input_var_names() const {
    std::vector<std::string> input_names;
    for (const auto& input : m_op_desc.inputs()) {
        for (const auto& var : input.arguments()) {
            input_names.push_back(var);
        }
    }
    return input_names;
}

/// \brief Get the output size
size_t DecoderProto::get_output_size(const std::string& port_name) const {
    size_t res = 0;
    for (const auto& output : m_op_desc.outputs()) {
        if (output.parameter() == port_name) {
            res = output.arguments_size();
        }
    }
    return res;
}

size_t DecoderProto::get_output_size() const {
    size_t res = 0;
    for (const auto& output : m_op_desc.outputs()) {
        res += output.arguments().size();
    }
    return res;
}

/// \brief Get the input size
size_t DecoderProto::get_input_size(const std::string& port_name) const {
    size_t res = 0;
    for (const auto& input : m_op_desc.inputs()) {
        if (input.parameter() == port_name) {
            res = input.arguments_size();
        }
    }
    return res;
}

size_t DecoderProto::get_input_size() const {
    size_t res = 0;
    for (const auto& input : m_op_desc.inputs()) {
        res += input.arguments().size();
    }
    return res;
}

// TODO: to reuse get_output_port_infos or abondan this member?
std::map<std::string, std::vector<ov::element::Type>> DecoderProto::get_output_type_map() const {
    std::map<std::string, std::vector<ov::element::Type>> output_types;
    for (const auto& output : m_op_desc.outputs()) {
        for (const auto& var_name : output.arguments()) {
            auto p = m_input_model.get_place_by_tensor_name(var_name);
            const auto& t = std::dynamic_pointer_cast<TensorPlace>(p);
            output_types[output.parameter()].push_back(t->get_element_type());
        }
    }
    return output_types;
}

std::vector<std::pair<ov::element::Type, ov::PartialShape>> DecoderProto::get_output_port_infos(
    const std::string& port_name) const {
    std::vector<std::pair<ov::element::Type, ov::PartialShape>> output_types;
    for (const auto& output : m_op_desc.outputs()) {
        if (output.parameter() == port_name) {
            for (const auto& var_name : output.arguments()) {
                auto p = m_input_model.get_place_by_tensor_name(var_name);
                const auto& t = std::dynamic_pointer_cast<TensorPlace>(p);
                output_types.push_back({t->get_element_type(), t->get_partial_shape()});
            }
        }
    }
    return output_types;
}

// TODO: to reuse get_output_port_infos or abondan this member?
ov::element::Type DecoderProto::get_out_port_type(const std::string& port_name) const {
    std::vector<ov::element::Type> output_types;
    for (const auto& output : m_op_desc.outputs()) {
        if (output.parameter() == port_name) {
            for (const auto& var_name : output.arguments()) {
                auto p = m_input_model.get_place_by_tensor_name(var_name);
                const auto& t = std::dynamic_pointer_cast<TensorPlace>(p);
                output_types.push_back(t->get_element_type());
            }
        }
    }
    FRONT_END_GENERAL_CHECK(!output_types.empty(), "Port has no tensors connected.");
    FRONT_END_GENERAL_CHECK(std::equal(output_types.begin() + 1, output_types.end(), output_types.begin()),
                            "Port has tensors with different types connected.");
    return output_types[0];
}

std::string DecoderProto::get_op_type() const {
    return m_op_desc.type();
}

std::vector<proto::OpDesc_Attr> DecoderProto::decode_attribute_helper(const std::string& name) const {
    std::vector<proto::OpDesc_Attr> attrs;
    for (const auto& attr : m_op_desc.attrs()) {
        if (attr.name() == name)
            attrs.push_back(attr);
    }
    FRONT_END_GENERAL_CHECK(attrs.size() <= 1,
                            "An error occurred while parsing the ",
                            name,
                            " attribute of ",
                            m_op_desc.type(),
                            "node. Unsupported number of attributes. Current number: ",
                            attrs.size(),
                            " Expected number: 0 or 1");
    return attrs;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
VarDecoderProto::VarDecoderProto(const ::paddle::framework::proto::VarDesc& var_desc)
            : m_var_desc(var_desc) {
            OPENVINO_DEBUG << "Paddle VarDesc " << m_var_desc.name() << " has VarType "
                            << ::paddle::framework::proto::VarType_Type_Name(m_var_desc.type().type()) << std::endl;
}

ov::element::Type VarDecoderProto::get_data_type() const {
    const auto& var_type = m_var_desc.type();
    if (var_type.has_lod_tensor()) { // LoDTensorDescs
        const auto& tensor_desc = var_type.lod_tensor().tensor();
        return TYPE_MAP[tensor_desc.data_type()];
    } else if (var_type.has_selected_rows()) { // TensorDesc
        const auto& tensor_desc = var_type.selected_rows();
        return TYPE_MAP[tensor_desc.data_type()];
    } else if (var_type.has_tensor_array()) { // TensorArray
        const auto& tensor_desc = var_type.tensor_array().tensor();
        return TYPE_MAP[tensor_desc.data_type()];
    } else {
        FRONT_END_THROW("Should not get data type for Paddle VarType " + ::paddle::framework::proto::VarType_Type_Name(var_type.type()));
        return ov::element::Type();
    }
}


ov::PartialShape VarDecoderProto::get_tensor_dims() const {
    const auto& var_type = m_var_desc.type();
    if (var_type.has_lod_tensor()) { // LoDTensorDesc
        const auto& tensor_desc = var_type.lod_tensor().tensor();
        return PartialShape(std::vector<Dimension>(tensor_desc.dims().begin(), tensor_desc.dims().end()));
    } else if (var_type.has_selected_rows()) { // TensorDesc
        const auto& tensor_desc = var_type.selected_rows();
        return PartialShape(std::vector<Dimension>(tensor_desc.dims().begin(), tensor_desc.dims().end()));
    } else if (var_type.has_tensor_array()) { // TensorArray
        const auto& tensor_desc = var_type.tensor_array().tensor();
        return PartialShape(std::vector<Dimension>(tensor_desc.dims().begin(), tensor_desc.dims().end()));
    } else {
        FRONT_END_THROW("Should not get data type for Paddle VarType " + ::paddle::framework::proto::VarType_Type_Name(var_type.type()));
        return ov::PartialShape();
    }
}

/// \brief Get the name of the variable
std::string VarDecoderProto::get_name() const {
    return m_var_desc.name();
}

/// \brief check if the variable is persistable
bool VarDecoderProto::is_persistable() const {
    return m_var_desc.persistable();
}

/// \brief check if the variable is LOD_TENSOR
bool VarDecoderProto::is_lod_tensor() const {
    return m_var_desc.type().has_lod_tensor();
}

    /// \brief check if the variable is TENSOR_ARRAY
bool VarDecoderProto::is_tensor_array() const {
    return m_var_desc.type().has_tensor_array();
}
///////////////////////////////////////////////////////////////////////////////////////////////////
namespace {
inline std::map<std::string, OutputVector> map_for_each_input_impl(
    const google::protobuf::RepeatedPtrField<::paddle::framework::proto::OpDesc_Var>& c,
    const std::function<Output<Node>(const std::string&, size_t)>& func) {
    size_t idx = 0;
    std::map<std::string, OutputVector> res;
    for (const auto& port : c) {
        std::vector<Output<Node>> v;
        v.reserve(port.arguments_size());
        for (const auto& inp : port.arguments()) {
            v.push_back(func(inp, idx++));
        }
        res.emplace(std::make_pair(port.parameter(), v));
    }
    return res;
}
}  // namespace

std::map<std::string, OutputVector> DecoderProto::map_for_each_input(
    const std::function<Output<Node>(const std::string&, size_t)>& func) const {
    return map_for_each_input_impl(m_op_desc.inputs(), func);
}

std::map<std::string, OutputVector> DecoderProto::map_for_each_output(
    const std::function<Output<Node>(const std::string&, size_t)>& func) const {
    return map_for_each_input_impl(m_op_desc.outputs(), func);
}

}  // namespace paddle
}  // namespace frontend
}  // namespace ov
