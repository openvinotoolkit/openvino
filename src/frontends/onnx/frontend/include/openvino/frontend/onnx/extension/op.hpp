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
                                [ov_type_name](){return create_ov_node_by_name(ov_type_name);},
                                                            attr_names_map,
                                                            attr_values_map)) {}
};

}  // namespace onnx
}  // namespace frontend
}  // namespace ov
