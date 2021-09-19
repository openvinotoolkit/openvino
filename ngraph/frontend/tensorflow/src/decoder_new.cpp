// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "decoder_new.hpp"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "op_def.pb.h"
#include "node_def.pb.h"

namespace ngraph {
namespace frontend {
std::map<::tensorflow::DataType, ngraph::element::Type> TYPE_MAP{
    {::tensorflow::DataType::DT_BOOL, ngraph::element::boolean},
    {::tensorflow::DataType::DT_INT16, ngraph::element::i16},
    {::tensorflow::DataType::DT_INT32, ngraph::element::i32},
    {::tensorflow::DataType::DT_INT64, ngraph::element::i64},
    {::tensorflow::DataType::DT_HALF, ngraph::element::f16},
    {::tensorflow::DataType::DT_FLOAT, ngraph::element::f32},
    {::tensorflow::DataType::DT_DOUBLE, ngraph::element::f64},
    {::tensorflow::DataType::DT_UINT8, ngraph::element::u8},
    {::tensorflow::DataType::DT_INT8, ngraph::element::i8},
    {::tensorflow::DataType::DT_BFLOAT16, ngraph::element::bf16}};

std::shared_ptr<Variant> DecoderTFProto::get_attribute(const std::string& name,
                                                         const VariantTypeInfo& type_info) const {
    auto attrs = decode_attribute_helper(name);
    if (attrs.empty()) {
        return nullptr;
    }

    if (type_info == VariantWrapper<std::string>::type_info) {
        return std::make_shared<VariantWrapper<std::string>>(attrs[0].s());
    } else if (type_info == VariantWrapper<int64_t>::type_info) {
        return std::make_shared<VariantWrapper<int64_t>>(attrs[0].i());
    } else if (type_info == VariantWrapper<std::vector<int64_t>>::type_info) {
        std::vector<int64_t> longs;
        longs.reserve(attrs[0].list().i_size());
        for (size_t idx = 0; idx < attrs[0].list().i_size(); ++idx) {
            longs.push_back(attrs[0].list().i(idx));
        }
        return std::make_shared<VariantWrapper<std::vector<int64_t>>>(longs);
    } else if (type_info == VariantWrapper<int32_t>::type_info) {
        return std::make_shared<VariantWrapper<int32_t>>(static_cast<int32_t>(attrs[0].i()));
    } else if (type_info == VariantWrapper<std::vector<int32_t>>::type_info) {
        std::vector<int32_t> ints;
        ints.reserve(attrs[0].list().i_size());
        for (size_t idx = 0; idx < attrs[0].list().i_size(); ++idx) {
            ints.push_back(static_cast<int32_t>(attrs[0].list().i(idx)));
        }
        return std::make_shared<VariantWrapper<std::vector<int32_t>>>(ints);
    } else if (type_info == VariantWrapper<float>::type_info) {
        return std::make_shared<VariantWrapper<float>>(attrs[0].f());
    } else if (type_info == VariantWrapper<std::vector<float>>::type_info) {
        std::vector<float> floats;
        floats.reserve(attrs[0].list().i_size());
        for (size_t idx = 0; idx < attrs[0].list().i_size(); ++idx) {
            floats.push_back(attrs[0].list().f(idx));
        }
        return std::make_shared<VariantWrapper<std::vector<float>>>(floats);
    } else if (type_info == VariantWrapper<ngraph::element::Type>::type_info) {
        auto data_type = attrs[0].type();
        return std::make_shared<VariantWrapper<ngraph::element::Type>>(TYPE_MAP[data_type]);
    } else if (type_info == VariantWrapper<bool>::type_info) {
        return std::make_shared<VariantWrapper<bool>>(attrs[0].b());
    }

    // type is not supported by decoder
    return nullptr;
}

size_t DecoderTFProto::get_input_size() const {
    return m_node_def->input_size();
}

void DecoderTFProto::get_input_node(const size_t input_port_idx,
    std::string& producer_name,
    size_t& producer_output_port_index) const {
    std::string producer_port_name = m_node_def->input(input_port_idx);
    // TODO: Implement full logic to detect only the last : as a separator, consult with TF
    auto delim_pos = producer_port_name.find(':');
    if (delim_pos != std::string::npos) {
        producer_name = producer_port_name.substr(0, delim_pos);
        producer_output_port_index = std::stoi(producer_port_name.substr(delim_pos));
        return;
    }
    producer_name = producer_port_name;
    producer_output_port_index = 0;
}

std::vector<tf::OutPortName> DecoderTFProto::get_output_names() const {
    FRONT_END_NOT_IMPLEMENTED("DecoderTFProto::get_output_names");
}

size_t DecoderTFProto::get_output_size() const {
    FRONT_END_NOT_IMPLEMENTED("DecoderTFProto::get_output_size");
}

std::map<size_t, std::vector<ngraph::element::Type>> DecoderTFProto::get_output_type_map() const {
    FRONT_END_NOT_IMPLEMENTED("DecoderTFProto::get_output_type_map");
}

ngraph::element::Type DecoderTFProto::get_out_port_type(const size_t& port_index) const {
    FRONT_END_NOT_IMPLEMENTED("DecoderTFProto::get_out_port_type");
}

std::string DecoderTFProto::get_op_type() const {
    return m_node_def->op();
}

std::string DecoderTFProto::get_op_name() const {
    return m_node_def->name();
}

std::vector<::tensorflow::AttrValue> DecoderTFProto::decode_attribute_helper(const std::string& name) const {
    auto attr_map = m_node_def->attr();
    FRONT_END_GENERAL_CHECK(attr_map.contains(name),
                            "An error occurred while parsing the ",
                            name,
                            " attribute of ",
                            this->get_op_type(),
                            "node");
    auto value = m_node_def->attr().at(name);
    return {value};
}

}  // namespace frontend
}  // namespace ngraph
