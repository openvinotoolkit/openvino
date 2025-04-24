// Copyright (C) 2018-2025 Intel Corporation
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

#include "framework.pb.h"
#include "openvino/core/any.hpp"
#include "openvino/frontend/paddle/frontend.hpp"
#include "openvino/frontend/paddle/node_context.hpp"
#include "place.hpp"

namespace ov {
namespace frontend {
namespace paddle {

ov::element::Type get_ov_type(const ::paddle::framework::proto::VarType_Type& type);

class DecoderProto : public paddle::DecoderBase {
public:
    explicit DecoderProto(const std::shared_ptr<OpPlace>& op) : op_place(op) {}

    ov::Any get_attribute(const std::string& name) const override;

    std::vector<TensorName> get_output_var_names(const std::string& var_name) const override;
    std::vector<TensorName> get_input_var_names(const std::string& var_name) const override;

    ov::Any convert_attribute(const ov::Any& data, const std::type_info& type_info) const override;

    std::vector<paddle::OutPortName> get_output_names() const override;

    size_t get_output_size() const override;
    size_t get_output_size(const std::string& port_name) const override;

    ov::element::Type get_out_port_type(const std::string& port_name) const override;

    std::string get_op_type() const override;

    std::map<std::string, std::vector<ov::element::Type>> get_output_type_map() const;
    std::vector<std::pair<ov::element::Type, ov::PartialShape>> get_output_port_infos(
        const std::string& port_name) const override;

    std::map<std::string, OutputVector> map_for_each_input(
        const std::function<Output<Node>(const std::string&, size_t)>& func) const;

    std::map<std::string, OutputVector> map_for_each_output(
        const std::function<Output<Node>(const std::string&, size_t)>& func) const;

    int64_t get_version() const override;

private:
    std::vector<::paddle::framework::proto::OpDesc_Attr> decode_attribute_helper(const std::string& name) const;
    std::weak_ptr<OpPlace> op_place;

    const std::shared_ptr<OpPlace> get_place() const {
        auto place = op_place.lock();
        if (!place)
            FRONT_END_THROW("This proto decoder contains empty op place.");
        return place;
    }
};

}  // namespace paddle
}  // namespace frontend
}  // namespace ov
