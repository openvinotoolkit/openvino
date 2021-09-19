// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <chrono>
#include <fstream>
#include <map>
#include <memory>
#include <ngraph/ngraph.hpp>
#include <tensorflow_frontend/frontend.hpp>
#include <tensorflow_frontend/place.hpp>
#include <string>
#include <utility>
#include <vector>

#include "attr_value.pb.h"
#include "types.pb.h"
#include "node_context_new.hpp"
#include "node_def.pb.h"

namespace ngraph {
namespace frontend {
extern std::map<::tensorflow::DataType, ngraph::element::Type> TYPE_MAP;

class DecoderTFProto : public ::ngraph::frontend::DecoderBase {
public:
    explicit DecoderTFProto(const ::tensorflow::NodeDef* node_def) : m_node_def(node_def) {}

    std::shared_ptr<Variant> get_attribute(const std::string& name, const VariantTypeInfo& type_info) const override;

    virtual size_t get_input_size() const override;

    virtual void get_input_node(const size_t input_port_idx,
                                std::string& producer_name,
                                size_t& producer_output_port_index) const override;

    std::vector<tf::OutPortName> get_output_names() const override;

    size_t get_output_size() const override;

    ngraph::element::Type get_out_port_type(const size_t& port_index) const override;

    std::string get_op_type() const override;

    std::string get_op_name() const override;

    std::map<size_t, std::vector<ngraph::element::Type>> get_output_type_map() const;

private:
    std::vector<::tensorflow::AttrValue> decode_attribute_helper(const std::string& name) const;
    const ::tensorflow::NodeDef* m_node_def;
};

}  // namespace frontend
}  // namespace ngraph
