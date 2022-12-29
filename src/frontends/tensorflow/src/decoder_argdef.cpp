// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "decoder_argdef.hpp"

#include "op_def.pb.h"
#include "openvino/frontend/tensorflow/node_context.hpp"
#include "openvino/frontend/tensorflow/special_types.hpp"
#include "types.pb.h"

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

size_t DecoderArgDef::get_input_size() const {
    FRONT_END_GENERAL_CHECK(m_op_type == "input_arg" || m_op_type == "output_arg",
                            "[TensorFlow Frontend] Internal error: Incorrect use of DecoderArgDef class.");
    if (m_op_type == "input_arg") {
        return 0;
    } else {
        return 1;
    }
}

const std::string& DecoderArgDef::get_op_type() const {
    FRONT_END_GENERAL_CHECK(m_op_type == "input_arg" || m_op_type == "output_arg",
                            "[TensorFlow Frontend] Internal error: Incorrect use of DecoderArgDef class.");
    return m_op_type;
}

const std::string& DecoderArgDef::get_op_name() const {
    return m_arg_def->name();
}

void DecoderArgDef::get_input_node(size_t input_port_idx,
                                   std::string& producer_name,
                                   size_t& producer_output_port_index) const {
    // Body graph nodes may have two colons `:`, for example,
    // producer_name:z:2 means that producer operation name is `producer_name`
    // and output port is 2
    FRONT_END_GENERAL_CHECK(m_op_type == "output_arg",
                            "[TensorFlow Frontend] Internal error: get_input_node is supported only for output_arg.");
    auto first_colon = m_producer_name.find_first_of(":");
    auto last_colon = m_producer_name.find_last_of(":");
    if (first_colon != std::string::npos && last_colon != std::string::npos) {
        producer_name = m_producer_name.substr(0, first_colon);
        auto port_id = m_producer_name.substr(last_colon + 1);
        FRONT_END_GENERAL_CHECK(!port_id.empty() && std::all_of(port_id.begin(), port_id.end(), ::isdigit),
                                "Port id is not specified or not a number. Value: ",
                                port_id);
        producer_output_port_index = std::stoi(port_id);
        return;
    }
    producer_name = m_producer_name;
    producer_output_port_index = 0;
}

ov::Any DecoderArgDef::get_attribute(const std::string& name) const {
    FRONT_END_GENERAL_CHECK(name == "type",
                            "[TensorFlow Frontend] Internal error: DecoderArgDef supports only `type` attribute.");
    if (TYPE_MAP().count(m_arg_def->type())) {
        return TYPE_MAP().at(m_arg_def->type());
    } else {
        // for all unsupported types return undefined type
        return ov::element::undefined;
    }
}

}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
