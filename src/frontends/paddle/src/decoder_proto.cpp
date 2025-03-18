// Copyright (C) 2018-2025 Intel Corporation
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

namespace ov {
namespace frontend {
namespace paddle {

using namespace ::paddle::framework;

ov::element::Type get_ov_type(const ::paddle::framework::proto::VarType_Type& type) {
    static const std::map<proto::VarType_Type, ov::element::Type> type_map{
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

    auto it = type_map.find(type);
    OPENVINO_ASSERT(it != type_map.end(), "Cannot convert PDPD type to ov::element::Type");
    return it->second;
}

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
    case proto::AttrType::SCALARS: {
        auto scalars_size = attrs[0].scalars_size();
        if (scalars_size >= 1) {
            if (Scalar_Type_Name(attrs[0].scalars(0).type()) == "LONG") {
                std::vector<int64_t> res;
                res.reserve(scalars_size);
                for (int i = 0; i < scalars_size; ++i) {
                    res.push_back(attrs[0].scalars(i).i());
                }
                return res;
            } else if (Scalar_Type_Name(attrs[0].scalars(0).type()) == "FLOAT64") {
                std::vector<double> res;
                res.reserve(scalars_size);
                for (int i = 0; i < scalars_size; ++i) {
                    res.push_back(attrs[0].scalars(i).r());
                }
                return res;
            } else if (Scalar_Type_Name(attrs[0].scalars(0).type()) == "BOOLEAN") {
                std::vector<bool> res;
                res.reserve(scalars_size);
                for (int i = 0; i < scalars_size; ++i) {
                    res.push_back(attrs[0].scalars(i).b());
                }
                return res;
            }
        } else {
            FRONT_END_GENERAL_CHECK(false,
                                    "Conversion from PaddlePaddle to OpenVINO  is not supported 0 dims in SCALARS.");
            break;
        }
    }
    default:
        FRONT_END_GENERAL_CHECK(false, "Conversion from PaddlePaddle to OpenVINO data type is not supported.");
    }
}

int64_t DecoderProto::get_version() const {
    return get_place()->get_version();
}

ov::Any DecoderProto::convert_attribute(const Any& data, const std::type_info& type_info) const {
    if (data.is<int32_t>() && type_info == typeid(ov::element::Type)) {
        return get_ov_type(static_cast<proto::VarType_Type>(data.as<int32_t>()));
    } else if (data.is<std::vector<int32_t>>() && type_info == typeid(std::vector<ov::element::Type>)) {
        const auto& casted = data.as<std::vector<int32_t>>();
        std::vector<ov::element::Type> types(casted.size());
        for (size_t i = 0; i < casted.size(); ++i) {
            types[i] = get_ov_type(static_cast<proto::VarType_Type>(casted[i]));
        }
        return types;
    }
    // no conversion rules found.
    return data;
}

std::vector<paddle::OutPortName> DecoderProto::get_output_names() const {
    std::vector<std::string> output_names;
    for (const auto& output : get_place()->get_desc().outputs()) {
        output_names.push_back(output.parameter());
    }
    return output_names;
}

std::vector<paddle::TensorName> DecoderProto::get_output_var_names(const std::string& var_name) const {
    std::vector<std::string> output_names;
    for (const auto& output : get_place()->get_desc().outputs()) {
        if (output.parameter() == var_name) {
            for (int idx = 0; idx < output.arguments_size(); ++idx) {
                output_names.push_back(output.arguments()[idx]);
            }
        }
    }
    return output_names;
}

std::vector<paddle::TensorName> DecoderProto::get_input_var_names(const std::string& var_name) const {
    std::vector<std::string> input_names;
    for (const auto& input : get_place()->get_desc().inputs()) {
        if (input.parameter() == var_name) {
            for (int idx = 0; idx < input.arguments_size(); ++idx) {
                input_names.push_back(input.arguments()[idx]);
            }
        }
    }
    return input_names;
}

size_t DecoderProto::get_output_size(const std::string& port_name) const {
    const auto out_port = get_place()->get_output_ports().at(port_name);
    return out_port.size();
}

size_t DecoderProto::get_output_size() const {
    size_t res = 0;
    for (const auto& output : get_place()->get_desc().outputs()) {
        res += output.arguments().size();
    }
    return res;
}

std::map<std::string, std::vector<ov::element::Type>> DecoderProto::get_output_type_map() const {
    std::map<std::string, std::vector<ov::element::Type>> output_types;
    for (const auto& out_port_pair : get_place()->get_output_ports()) {
        for (const auto& p_place : out_port_pair.second) {
            output_types[out_port_pair.first].push_back(p_place->get_target_tensor_paddle()->get_element_type());
        }
    }
    return output_types;
}

std::vector<std::pair<ov::element::Type, ov::PartialShape>> DecoderProto::get_output_port_infos(
    const std::string& port_name) const {
    std::vector<std::pair<ov::element::Type, ov::PartialShape>> output_types;
    for (const auto& out_port : get_place()->get_output_ports().at(port_name)) {
        output_types.push_back({out_port->get_target_tensor_paddle()->get_element_type(),
                                out_port->get_target_tensor_paddle()->get_partial_shape()});
    }
    return output_types;
}

ov::element::Type DecoderProto::get_out_port_type(const std::string& port_name) const {
    std::vector<ov::element::Type> output_types;
    for (const auto& out_port : get_place()->get_output_ports().at(port_name)) {
        output_types.push_back(out_port->get_target_tensor_paddle()->get_element_type());
    }
    FRONT_END_GENERAL_CHECK(!output_types.empty(), "Port has no tensors connected.");
    FRONT_END_GENERAL_CHECK(std::equal(output_types.begin() + 1, output_types.end(), output_types.begin()),
                            "Port has tensors with different types connected.");
    return output_types[0];
}

std::string DecoderProto::get_op_type() const {
    return get_place()->get_desc().type();
}

std::vector<proto::OpDesc_Attr> DecoderProto::decode_attribute_helper(const std::string& name) const {
    std::vector<proto::OpDesc_Attr> attrs;
    for (const auto& attr : get_place()->get_desc().attrs()) {
        if (attr.name() == name)
            attrs.push_back(attr);
    }
    FRONT_END_GENERAL_CHECK(attrs.size() <= 1,
                            "An error occurred while parsing the ",
                            name,
                            " attribute of ",
                            get_place()->get_desc().type(),
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
    return map_for_each_input_impl(get_place()->get_desc().inputs(), func);
}

std::map<std::string, OutputVector> DecoderProto::map_for_each_output(
    const std::function<Output<Node>(const std::string&, size_t)>& func) const {
    return map_for_each_input_impl(get_place()->get_desc().outputs(), func);
}

}  // namespace paddle
}  // namespace frontend
}  // namespace ov
