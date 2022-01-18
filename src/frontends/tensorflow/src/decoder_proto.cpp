// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "decoder_proto.hpp"

#include "node_context.hpp"

namespace ov {
namespace frontend {
namespace tensorflow {

namespace {
const std::map<::tensorflow::DataType, ov::element::Type>& TYPE_MAP() {
    static const std::map<::tensorflow::DataType, ov::element::Type> type_map{
        {::tensorflow::DataType::DT_BOOL, ov::element::boolean},
        {::tensorflow::DataType::DT_INT16, ov::element::i16},
        {::tensorflow::DataType::DT_INT32, ov::element::i32},
        {::tensorflow::DataType::DT_INT64, ov::element::i64},
        {::tensorflow::DataType::DT_HALF, ov::element::f16},
        {::tensorflow::DataType::DT_FLOAT, ov::element::f32},
        {::tensorflow::DataType::DT_DOUBLE, ov::element::f64},
        {::tensorflow::DataType::DT_UINT8, ov::element::u8},
        {::tensorflow::DataType::DT_INT8, ov::element::i8},
        {::tensorflow::DataType::DT_BFLOAT16, ov::element::bf16}};
    return type_map;
}
}  // namespace

ov::Any DecoderProto::get_attribute(const std::string& name, const std::type_info& type_info) const {
    auto attrs = decode_attribute_helper(name);
    if (attrs.empty()) {
        return {};
    }

    if (type_info == typeid(std::string)) {
        return attrs[0].s();
    } else if (type_info == typeid(int64_t)) {
        return attrs[0].i();
    } else if (type_info == typeid(std::vector<int64_t>)) {
        std::vector<int64_t> longs;
        longs.reserve(attrs[0].list().i_size());
        for (size_t idx = 0; idx < attrs[0].list().i_size(); ++idx) {
            longs.push_back(attrs[0].list().i(idx));
        }
        return longs;
    } else if (type_info == typeid(int32_t)) {
        return static_cast<int32_t>(attrs[0].i());
    } else if (type_info == typeid(std::vector<int32_t>)) {
        std::vector<int32_t> ints;
        ints.reserve(attrs[0].list().i_size());
        for (size_t idx = 0; idx < attrs[0].list().i_size(); ++idx) {
            ints.push_back(static_cast<int32_t>(attrs[0].list().i(idx)));
        }
        return ints;
    } else if (type_info == typeid(float)) {
        return attrs[0].f();
    } else if (type_info == typeid(std::vector<float>)) {
        std::vector<float> floats;
        floats.reserve(attrs[0].list().i_size());
        for (size_t idx = 0; idx < attrs[0].list().i_size(); ++idx) {
            floats.push_back(attrs[0].list().f(idx));
        }
        return floats;
    } else if (type_info == typeid(ov::element::Type)) {
        auto data_type = attrs[0].type();
        return TYPE_MAP().at(data_type);
    } else if (type_info == typeid(bool)) {
        return attrs[0].b();
    } else if (type_info == typeid(::tensorflow::DataType)) {
        return attrs[0].type();
    } else if (type_info == typeid(::tensorflow::TensorProto)) {
        return attrs[0].tensor();
    } else if (type_info == typeid(::ov::PartialShape)) {
        std::vector<ov::Dimension> dims;
        auto tf_shape = attrs[0].shape();
        for (int i = 0; i < tf_shape.dim_size(); i++) {
            dims.push_back(tf_shape.dim(i).size());
        }
        auto pshape = ov::PartialShape(dims);
        return pshape;
    }

    // type is not supported by decoder
    return {};
}

size_t DecoderProto::get_input_size() const {
    return m_node_def->input_size();
}

void DecoderProto::get_input_node(size_t input_port_idx,
                                  std::string& producer_name,
                                  size_t& producer_output_port_index) const {
    // TODO: handle body graph nodes with a couple of columns
    std::string producer_port_name = m_node_def->input(input_port_idx);
    auto delim_pos = producer_port_name.find(':');
    if (delim_pos != std::string::npos) {
        producer_name = producer_port_name.substr(0, delim_pos);
        producer_output_port_index = std::stoi(producer_port_name.substr(delim_pos));
        return;
    }
    producer_name = producer_port_name;
    producer_output_port_index = 0;
}

const std::string& DecoderProto::get_op_type() const {
    return m_node_def->op();
}

const std::string& DecoderProto::get_op_name() const {
    return m_node_def->name();
}

std::vector<::tensorflow::AttrValue> DecoderProto::decode_attribute_helper(const std::string& name) const {
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
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
