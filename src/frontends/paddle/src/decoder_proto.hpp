// Copyright (C) 2018-2022 Intel Corporation
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
extern std::map<::paddle::framework::proto::VarType_Type, ov::element::Type> TYPE_MAP;

class DecoderProto : public paddle::DecoderBase {
public:
    explicit DecoderProto(const std::shared_ptr<OpPlace>& op) : op_place(op) {}

    ov::Any get_attribute(const std::string& name) const override;

    ov::Any convert_attribute(const ov::Any& data, const std::type_info& type_info) const override;

    std::vector<paddle::OutPortName> get_output_names() const override;

    size_t get_output_size() const override;

    ov::element::Type get_out_port_type(const std::string& port_name) const override;

    std::string get_op_type() const override;

    std::map<std::string, std::vector<ov::element::Type>> get_output_type_map() const;

    std::map<std::string, OutputVector> map_for_each_input(
        const std::function<Output<Node>(const std::string&, size_t)>& func) const;

    std::map<std::string, OutputVector> map_for_each_output(
        const std::function<Output<Node>(const std::string&, size_t)>& func) const;

private:
    std::vector<::paddle::framework::proto::OpDesc_Attr> decode_attribute_helper(const std::string& name) const;
    std::shared_ptr<OpPlace> op_place;
};

}  // namespace paddle
}  // namespace frontend
}  // namespace ov
