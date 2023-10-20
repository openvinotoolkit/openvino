// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>

#include "openvino/core/type/element_type.hpp"
#include "openvino/frontend/tensorflow/decoder.hpp"
#include "types.pb.h"

namespace ov_tensorflow {
class GraphDef;
class FunctionDef;
class NodeDef;
class AttrValue;
}  // namespace ov_tensorflow

namespace ov {
namespace frontend {
namespace tensorflow {

void parse_producer_name(const std::string& producer_port_name,
                         std::string& producer_name,
                         std::string& producer_output_port_name,
                         size_t& producer_output_port_index);

class DecoderProto : public ov::frontend::tensorflow::DecoderBase {
public:
    explicit DecoderProto(const ::ov_tensorflow::NodeDef* node_def,
                          const std::shared_ptr<::ov_tensorflow::GraphDef>& graph_def)
        : m_node_def(node_def),
          m_graph_def(graph_def),
          m_func_def(nullptr) {}

    explicit DecoderProto(const ::ov_tensorflow::NodeDef* node_def,
                          const std::shared_ptr<::ov_tensorflow::GraphDef>& graph_def,
                          const std::shared_ptr<::ov_tensorflow::FunctionDef>& func_def)
        : m_node_def(node_def),
          m_graph_def(graph_def),
          m_func_def(func_def) {}

    ov::Any get_attribute(const std::string& name) const override;

    size_t get_input_size() const override;

    void get_input_node(size_t input_port_idx,
                        std::string& producer_name,
                        std::string& producer_output_port_name,
                        size_t& producer_output_port_index) const override;

    const std::string& get_op_type() const override;

    const std::string& get_op_name() const override;

private:
    std::vector<::ov_tensorflow::AttrValue> decode_attribute_helper(const std::string& name) const;
    const ::ov_tensorflow::NodeDef* m_node_def;
    // For existence of NodeDef object corresponding to the main graph node,
    // GraphDef object must live in the memory
    const std::shared_ptr<::ov_tensorflow::GraphDef> m_graph_def;
    // For existence of NodeDef object corresponding to the body graph node,
    // both GraphDef and FunctionDef objects must be alive in the memory
    const std::shared_ptr<::ov_tensorflow::FunctionDef> m_func_def;
};
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
