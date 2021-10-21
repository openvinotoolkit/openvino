// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/ngraph.hpp>
#include <string>
#include <tensorflow_frontend/decoder.hpp>
#include <tensorflow_frontend/frontend.hpp>
#include <tensorflow_frontend/place.hpp>
#include <vector>

#include "attr_value.pb.h"
#include "node_def.pb.h"
#include "types.pb.h"

namespace ov {
namespace frontend {
namespace tf {

class DecoderTFProto : public DecoderBase {
public:
    explicit DecoderTFProto(const ::tensorflow::NodeDef* node_def) : m_node_def(node_def) {}

    std::shared_ptr<ov::Variant> get_attribute(const std::string& name,
                                               const VariantTypeInfo& type_info) const override;

    size_t get_input_size() const override;

    void get_input_node(size_t input_port_idx,
                        std::string& producer_name,
                        size_t& producer_output_port_index) const override;

    const std::string& get_op_type() const override;

    const std::string& get_op_name() const override;

private:
    std::vector<::tensorflow::AttrValue> decode_attribute_helper(const std::string& name) const;
    const ::tensorflow::NodeDef* m_node_def;
};
}  // namespace tf
}  // namespace frontend
}  // namespace ov
