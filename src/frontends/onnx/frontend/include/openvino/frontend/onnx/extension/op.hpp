// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "openvino/frontend/extension/op.hpp"
#include "openvino/frontend/onnx/extension/conversion.hpp"

namespace ov {
namespace frontend {
namespace onnx {

// One-to-one operation mapping for OVOpType != void which means OV type is specified by OVOpType
// See a specialization for OVOptype = void
template <typename OVOpType = void>
class ONNX_FRONTEND_API OpExtension : public ConversionExtension {
public:
    // All attributes come from OVOpType definition, op type in FW and OV match, available for OVOpType != void only
    // Attributes mapping can be modified with optional parameters
    OpExtension(const std::map<std::string, std::string>& attr_names_map = {},
                    const std::map<std::string, ov::Any>& attr_values_map = {})
        : OpExtension(OVOpType::get_type_info_static().name, "", attr_names_map, attr_values_map) {}

    // Maps op with a given type in FW and OV type given in template parameter
    OpExtension(const std::string& fw_type_name,
                    const std::map<std::string, std::string>& attr_names_map = {},
                    const std::map<std::string, ov::Any>& attr_values_map = {})
                    : OpExtension(fw_type_name, "", attr_names_map, attr_values_map) {}

    // Maps op with a given type and domain in FW and OV type given in template parameter
    OpExtension(const std::string& fw_type_name,
                    const std::string& fw_domain,
                    const std::map<std::string, std::string>& attr_names_map = {},
                    const std::map<std::string, ov::Any>& attr_values_map = {})
                    : ConversionExtension(fw_type_name,
                                        fw_domain,
                                        OpConversionFunction(
                                            []() {
                                                return std::make_shared<OVOpType>();
                                            },
                                            attr_names_map,
                                            attr_values_map)) {}
};

template<>
class ONNX_FRONTEND_API OpExtension<void> : public ConversionExtension {
    public:
    // Default ctor is not available, you need to specify OV type with another ctor
    OpExtension() = delete;

    // Maps op with a given type in FW and matching OV type given in template parameter
    explicit OpExtension(const std::string& fw_ov_type_name,
                             const std::map<std::string, std::string>& attr_names_map = {},
                             const std::map<std::string, ov::Any>& attr_values_map = {})
        : OpExtension(fw_ov_type_name, fw_ov_type_name, attr_names_map, attr_values_map) {}

    // Maps op with a given type in FW and specified OV type given in template parameter
    OpExtension(const std::string& ov_type_name,
                    const std::string& fw_type_name,
                    const std::map<std::string, std::string>& attr_names_map = {},
                    const std::map<std::string, ov::Any>& attr_values_map = {})
         : OpExtension(ov_type_name, fw_type_name, "", attr_names_map, attr_values_map) {}

    // Maps op with a given type in FW and specified OV type given in template parameter
    OpExtension(const std::string& ov_type_name,
                    const std::string& fw_type_name,
                    const std::string& fw_domain_name,
                    const std::map<std::string, std::string>& attr_names_map = {},
                    const std::map<std::string, ov::Any>& attr_values_map = {})
         : ConversionExtension(fw_type_name,
                               fw_domain_name,
                               OpConversionFunction(
                                [=]() -> std::shared_ptr<ov::Node> {
                                                                auto split = [](const std::string& s, const std::string& delimiter) {
                                                                    size_t pos_start = 0, pos_end, delim_len = delimiter.length();
                                                                    std::string token;
                                                                    std::vector<std::string> res;

                                                                    while ((pos_end = s.find(delimiter, pos_start)) != std::string::npos) {
                                                                        token = s.substr(pos_start, pos_end - pos_start);
                                                                        pos_start = pos_end + delim_len;
                                                                        res.push_back(token);
                                                                    }

                                                                    res.push_back(s.substr(pos_start));
                                                                    return res;
                                                                };

                                                                // Expected formats:
                                                                // opsetN::OpName
                                                                // opsetN.OpName
                                                                // OpName
                                                                std::string opset_name;
                                                                std::string op_name;
                                                                auto cnt_colons = std::count(ov_type_name.begin(), ov_type_name.end(), ':');
                                                                auto cnt_dots = std::count(ov_type_name.begin(), ov_type_name.end(), '.');
                                                                if (cnt_colons == 2 && cnt_dots == 0) {
                                                                    auto divided = split(ov_type_name, "::");
                                                                    if (divided.size() != 2) {
                                                                        FRONT_END_GENERAL_CHECK(
                                                                            false,
                                                                            "Invalid OpenVINO operation format, one of the next is expected:"
                                                                            "opsetN::OpName or opsetN.OpName or OpName. Provided operation format: ",
                                                                            ov_type_name);
                                                                    }
                                                                    opset_name = divided[0];
                                                                    op_name = divided[1];
                                                                } else if (cnt_colons == 0 && cnt_dots == 1) {
                                                                    auto divided = split(ov_type_name, ".");
                                                                    if (divided.size() != 2) {
                                                                        FRONT_END_GENERAL_CHECK(
                                                                            false,
                                                                            "Invalid OpenVINO operation format, one of the next is expected:"
                                                                            "opsetN::OpName or opsetN.OpName or OpName. Provided operation format: ",
                                                                            ov_type_name);
                                                                    }
                                                                    opset_name = divided[0];
                                                                    op_name = divided[1];
                                                                } else if (cnt_colons == 0 && cnt_dots == 0) {
                                                                    opset_name = "latest";
                                                                    op_name = ov_type_name;
                                                                } else {
                                                                    FRONT_END_GENERAL_CHECK(
                                                                        false,
                                                                        "Invalid OpenVINO operation format, one of the next is expected: \n"
                                                                        "opsetN::OpName or opsetN.OpName or OpName. Provided operation format: ",
                                                                        ov_type_name);
                                                                }

                                                                const auto& opset = get_opset_by_name(opset_name);
                                                                if (!opset.contains_type(op_name)) {
                                                                    FRONT_END_GENERAL_CHECK(false,
                                                                                            "OpenVINO opset doesn't contain operation with "
                                                                                            "name ",
                                                                                            op_name);
                                                                }

                                                                return std::shared_ptr<ngraph::Node>(opset.create(op_name));
                                                            },
                                                            attr_names_map,
                                                            attr_values_map)) {}
};

}  // namespace onnx
}  // namespace frontend
}  // namespace ov
