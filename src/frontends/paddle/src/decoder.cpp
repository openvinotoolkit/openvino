// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "decoder.hpp"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "framework.pb.h"

namespace ov {
namespace frontend {
namespace paddle {

using namespace ::paddle::framework;

std::map<::paddle::framework::proto::VarType_Type, ov::element::Type> TYPE_MAP{
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

ov::Any DecoderProto::get_attribute(const std::string& name, const std::type_info& type_info) const {
    auto attrs = decode_attribute_helper(name);
    if (attrs.empty()) {
        return {};
    }

    if (type_info == typeid(std::string)) {
        return attrs[0].s();
    } else if (type_info == typeid(int64_t)) {
        return attrs[0].l();
    } else if (type_info == typeid(std::vector<int64_t>)) {
        return std::vector<int64_t>(attrs[0].longs().begin(), attrs[0].longs().end());
    } else if (type_info == typeid(int32_t)) {
        return attrs[0].i();
    } else if (type_info == typeid(std::vector<int32_t>)) {
        return std::vector<int32_t>(attrs[0].ints().begin(), attrs[0].ints().end());
    } else if (type_info == typeid(float)) {
        return attrs[0].f();
    } else if (type_info == typeid(std::vector<float>)) {
        return std::vector<float>(attrs[0].floats().begin(), attrs[0].floats().end());
    } else if (type_info == typeid(ov::element::Type)) {
        return TYPE_MAP[static_cast<::paddle::framework::proto::VarType_Type>(attrs[0].i())];
    } else if (type_info == typeid(bool)) {
        return attrs[0].b();
    }

    // Type is not supported by decoder
    return {};
}

std::vector<paddle::OutPortName> DecoderProto::get_output_names() const {
    std::vector<std::string> output_names;
    for (const auto& output : op_place->get_desc().outputs()) {
        output_names.push_back(output.parameter());
    }
    return output_names;
}

size_t DecoderProto::get_output_size() const {
    size_t res = 0;
    for (const auto& output : op_place->get_desc().outputs()) {
        res += output.arguments().size();
    }
    return res;
}

std::map<std::string, std::vector<ov::element::Type>> DecoderProto::get_output_type_map() const {
    std::map<std::string, std::vector<ov::element::Type>> output_types;
    for (const auto& out_port_pair : op_place->get_output_ports()) {
        for (const auto& p_place : out_port_pair.second) {
            output_types[out_port_pair.first].push_back(p_place->get_target_tensor_paddle()->get_element_type());
        }
    }
    return output_types;
}

ov::element::Type DecoderProto::get_out_port_type(const std::string& port_name) const {
    std::vector<ov::element::Type> output_types;
    for (const auto& out_port : op_place->get_output_ports().at(port_name)) {
        output_types.push_back(out_port->get_target_tensor_paddle()->get_element_type());
    }
    FRONT_END_GENERAL_CHECK(output_types.size() > 0, "Port has no tensors connected.");
    FRONT_END_GENERAL_CHECK(std::equal(output_types.begin() + 1, output_types.end(), output_types.begin()),
                            "Port has tensors with different types connected.");
    return output_types[0];
}

std::string DecoderProto::get_op_type() const {
    return op_place->get_desc().type();
}

std::vector<proto::OpDesc_Attr> DecoderProto::decode_attribute_helper(const std::string& name) const {
    std::vector<proto::OpDesc_Attr> attrs;
    for (const auto& attr : op_place->get_desc().attrs()) {
        if (attr.name() == name)
            attrs.push_back(attr);
    }
    FRONT_END_GENERAL_CHECK(attrs.size() <= 1,
                            "An error occurred while parsing the ",
                            name,
                            " attribute of ",
                            op_place->get_desc().type(),
                            "node. Unsupported number of attributes. Current number: ",
                            attrs.size(),
                            " Expected number: 0 or 1");
    return attrs;
}

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
    return map_for_each_input_impl(op_place->get_desc().inputs(), func);
}

std::map<std::string, OutputVector> DecoderProto::map_for_each_output(
    const std::function<Output<Node>(const std::string&, size_t)>& func) const {
    return map_for_each_input_impl(op_place->get_desc().outputs(), func);
}

}  // namespace paddle
}  // namespace frontend
}  // namespace ov
