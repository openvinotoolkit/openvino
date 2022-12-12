// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "decoder_flatbuffer.h"
#include "schema_generated.h"
//#include "attr_value.pb.h"
//#include "node_def.pb.h"
#include "openvino/frontend/tensorflow/node_context.hpp"
#include "openvino/frontend/tensorflow/special_types.hpp"
//#include "types.pb.h"

namespace ov {
namespace frontend {
namespace tensorflow_lite {

ov::Any DecoderFlatBuffer::get_attribute(const std::string& name) const {

}

size_t DecoderFlatBuffer::get_input_size() const {
    return m_node_def->inputs()->size();
}

void DecoderFlatBuffer::get_input_node(size_t input_port_idx,
                                      std::string& producer_name,
                                      size_t& producer_output_port_index) const {

}

const std::string& DecoderFlatBuffer::get_op_type() const {
    return m_type;
}

const std::string& DecoderFlatBuffer::get_op_name() const {
    return m_name;
}


}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
