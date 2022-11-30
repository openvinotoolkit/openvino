// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <chrono>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "openvino/core/any.hpp"
#include "openvino/frontend/paddle/frontend.hpp"
#include "openvino/frontend/paddle/node_context.hpp"
#include "place.hpp"

namespace paddle {
namespace framework {
namespace proto {
class OpDesc;
class VarDesc;
class OpDesc_Attr;
}  // namespace proto
}  // namespace framework
}  // namespace paddle

namespace ov {
namespace frontend {
namespace paddle {
class DecoderProto : public paddle::DecoderBase {
public:
    explicit DecoderProto(const ov::frontend::InputModel& input_model, const ::paddle::framework::proto::OpDesc& op_desc)
             : m_op_desc(op_desc),
               m_input_model(input_model) {}

    ov::Any get_attribute(const std::string& name) const override;

    ov::Any convert_attribute(const ov::Any& data, const std::type_info& type_info) const override;

    std::string get_op_type() const override;

    /// \brief Get the output port names
    std::vector<OutPortName> get_output_names() const override;
    const OutPortName& get_output_names(size_t idx) const override;

    /// \brief Get the input port names
    std::vector<InPortName> get_input_names() const override;
    const InPortName& get_input_names(size_t idx) const override;

    /// \brief Get the output tensor names
    std::vector<TensorName> get_output_var_names(const std::string& port_name) const override;
    std::vector<TensorName> get_output_var_names() const override;

    /// \brief Get the input tensor names
    std::vector<TensorName> get_input_var_names(const std::string& port_name) const override;
    std::vector<TensorName> get_input_var_names() const override;

    /// \brief Get the output size
    size_t get_output_size() const override;
    size_t get_output_size(const std::string& port_name) const override;

    /// \brief Get the input size
    size_t get_input_size() const override;
    size_t get_input_size(const std::string& port_name) const override;

    ov::element::Type get_out_port_type(const std::string& port_name) const override;

    std::map<std::string, std::vector<ov::element::Type>> get_output_type_map() const;
    std::vector<std::pair<ov::element::Type, ov::PartialShape>> get_output_port_infos(
        const std::string& port_name) const override;

    std::map<std::string, OutputVector> map_for_each_input(
        const std::function<Output<Node>(const std::string&, size_t)>& func) const;

    std::map<std::string, OutputVector> map_for_each_output(
        const std::function<Output<Node>(const std::string&, size_t)>& func) const;

private:
    std::vector<::paddle::framework::proto::OpDesc_Attr> decode_attribute_helper(const std::string& name) const;
    const ::paddle::framework::proto::OpDesc& m_op_desc;
    const ov::frontend::InputModel& m_input_model;  // Need InputModel because OpDesc uses var names to retrieve VarDesc.
};

class VarDecoderProto : public paddle::VarDecoderBase {
public:
    explicit VarDecoderProto(const ::paddle::framework::proto::VarDesc& var_desc);

    /// \brief Get the name of the variable
    std::string get_name() const;

    /// \brief Get the tensor data type
    ov::element::Type get_data_type() const;
    
    /// \brief Get the tensor data type
    ov::PartialShape get_tensor_dims() const;

    /// \brief check if the variable is persistable
    bool is_persistable() const;

    /// \brief check if the variable is LOD_TENSOR
    bool is_lod_tensor() const;

     /// \brief check if the variable is TENSOR_ARRAY
    bool is_tensor_array() const;

private:
    const ::paddle::framework::proto::VarDesc& m_var_desc;
};
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
