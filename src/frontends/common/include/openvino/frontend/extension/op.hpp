// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <fstream>
#include <iostream>
#include <type_traits>

#include "openvino/core/extension.hpp"
#include "openvino/frontend/extension/conversion.hpp"
#include "openvino/frontend/node_context.hpp"
#include "openvino/frontend/visibility.hpp"
#include "openvino/opsets/opset.hpp"

namespace ov {
namespace frontend {

/// \brief The helper function to return an instance of OpSet class initialized with
/// operations from provided opset by name.
/// \param opset_name Opset name (opsetN) to initialize OpSet class.
inline const ov::OpSet& get_opset_by_name(const std::string& opset_name) {
    const auto& opsets = ov::get_available_opsets();
    if (opsets.find(opset_name) != opsets.end())
        return opsets.at(opset_name)();
    if (opset_name.empty() || opset_name == "latest") {
        return ov::get_opset11();
    } else {
        FRONT_END_GENERAL_CHECK(false, "Unsupported opset name: ", opset_name);
    }
}

/// \brief The helper function to create an instance of ov::Node class initialized by provided type name.
/// Expected formats:
/// - opsetN::OpName
/// - opsetN.OpName
/// - OpName
/// \param ov_type_name Type name of created ov::Node.
inline std::shared_ptr<ov::Node> create_ov_node_by_name(const std::string& ov_type_name) {
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

    std::string opset_name;
    std::string op_name;
    auto cnt_colons = std::count(ov_type_name.begin(), ov_type_name.end(), ':');
    auto cnt_dots = std::count(ov_type_name.begin(), ov_type_name.end(), '.');
    if (cnt_colons == 2 && cnt_dots == 0) {
        auto divided = split(ov_type_name, "::");
        if (divided.size() != 2) {
            FRONT_END_GENERAL_CHECK(false,
                                    "Invalid OpenVINO operation format, one of the next is expected:"
                                    "opsetN::OpName or opsetN.OpName or OpName. Provided operation format: ",
                                    ov_type_name);
        }
        opset_name = divided[0];
        op_name = divided[1];
    } else if (cnt_colons == 0 && cnt_dots == 1) {
        auto divided = split(ov_type_name, ".");
        if (divided.size() != 2) {
            FRONT_END_GENERAL_CHECK(false,
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
        FRONT_END_GENERAL_CHECK(false,
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
}

// One-to-one operation mapping for OVOpType != void which means OV type is specified by OVOpType
// See a specialization for OVOptype = void
template <typename BaseConversionType, typename OVOpType = void>
class OpExtensionBase : public BaseConversionType {
public:
    // All attributes come from OVOpType definition, op type in FW and OV match, available for OVOpType != void only
    // Attributes mapping can be modified with optional parameters
    OpExtensionBase(const std::map<std::string, std::string>& attr_names_map = {},
                    const std::map<std::string, ov::Any>& attr_values_map = {})
        : OpExtensionBase(OVOpType::get_type_info_static().name, attr_names_map, attr_values_map) {}

    // Maps op with a given type in FW and OV type given in template parameter
    OpExtensionBase(const std::string& fw_type_name,
                    const std::map<std::string, std::string>& attr_names_map = {},
                    const std::map<std::string, ov::Any>& attr_values_map = {});
};

template <typename BaseConversionType>
class OpExtensionBase<BaseConversionType, void> : public BaseConversionType {
public:
    // Default ctor is not available, you need to specify OV type with another ctor
    OpExtensionBase() = delete;

    // Maps op with a given type in FW and matching OV type given in template parameter
    explicit OpExtensionBase(const std::string& fw_ov_type_name,
                             const std::map<std::string, std::string>& attr_names_map = {},
                             const std::map<std::string, ov::Any>& attr_values_map = {})
        : OpExtensionBase(fw_ov_type_name, fw_ov_type_name, attr_names_map, attr_values_map) {}

    // Maps op with a given type in FW and specified OV type given in template parameter
    OpExtensionBase(const std::string& ov_type_name,
                    const std::string& fw_type_name,
                    const std::map<std::string, std::string>& attr_names_map = {},
                    const std::map<std::string, ov::Any>& attr_values_map = {});
};

class FWVisitor : public ov::AttributeVisitor {
public:
    explicit FWVisitor(const NodeContext& context,
                       const std::map<std::string, std::string>& attr_names_map = {},
                       const std::map<std::string, ov::Any>& attr_values_map = {})
        : m_context(context),
          m_attr_names_map(attr_names_map),
          m_attr_values_map(attr_values_map) {}

    void on_adapter(const std::string& name, ValueAccessor<void>& adapter) override {
        auto p_value = m_attr_values_map.find(name);

        if (p_value != m_attr_values_map.end()) {
            adapter.set_as_any(p_value->second);
        } else {
            auto p_name = m_attr_names_map.find(name);
            const std::string& target_name = p_name != m_attr_names_map.end() ? p_name->second : name;
            try {
                adapter.set_as_any(m_context.get_attribute_as_any(target_name));
            } catch (::ov::AssertFailure& ex) {
                OPENVINO_ASSERT(false,
                                ex.what(),
                                "\nValue for attribute \"",
                                target_name,
                                "\" is not set or mapping between "
                                "framework and openvino node attributes is incorrect.");
            }
        }
    }

private:
    const NodeContext& m_context;
    const std::map<std::string, std::string>& m_attr_names_map;
    const std::map<std::string, ov::Any>& m_attr_values_map;
};

class OpConversionFunction {
public:
    explicit OpConversionFunction(const std::function<std::shared_ptr<ov::Node>()>& op_creator,
                                  const std::map<std::string, std::string>& attr_names_map = {},
                                  const std::map<std::string, ov::Any>& attr_values_map = {})
        : m_op_creator(op_creator),
          m_attr_names_map(attr_names_map),
          m_attr_values_map(attr_values_map) {}

    ov::OutputVector operator()(const NodeContext& context) {
        auto node = m_op_creator();

        std::vector<Output<Node>> inputs;
        for (size_t i = 0; i < context.get_input_size(); ++i) {
            inputs.push_back(context.get_input(static_cast<int>(i)));
        }
        node->set_arguments(inputs);
        FWVisitor fw_visitor(context, m_attr_names_map, m_attr_values_map);
        node->visit_attributes(fw_visitor);
        node->validate_and_infer_types();
        return node->outputs();
    }

private:
    std::function<std::shared_ptr<ov::Node>()> m_op_creator;
    std::map<std::string, std::string> m_attr_names_map;
    std::map<std::string, ov::Any> m_attr_values_map;
};

template <typename BaseConversionType>
OpExtensionBase<BaseConversionType, void>::OpExtensionBase(const std::string& ov_type_name,
                                                           const std::string& fw_type_name,
                                                           const std::map<std::string, std::string>& attr_names_map,
                                                           const std::map<std::string, ov::Any>& attr_values_map)
    : BaseConversionType(fw_type_name,
                         OpConversionFunction(
                             [ov_type_name]() {
                                 return create_ov_node_by_name(ov_type_name);
                             },
                             attr_names_map,
                             attr_values_map)) {}

template <typename BaseConversionType, typename OVOpType>
OpExtensionBase<BaseConversionType, OVOpType>::OpExtensionBase(const std::string& fw_type_name,
                                                               const std::map<std::string, std::string>& attr_names_map,
                                                               const std::map<std::string, ov::Any>& attr_values_map)
    : BaseConversionType(fw_type_name,
                         OpConversionFunction(
                             []() {
                                 return std::make_shared<OVOpType>();
                             },
                             attr_names_map,
                             attr_values_map)) {}

template <typename OVOpType = void>
using OpExtension = ov::frontend::OpExtensionBase<ov::frontend::ConversionExtension, OVOpType>;

// Per each FRAMEWORK this macro can be used once in one operation class definition
// It defines a member inline function that creates required extension.
#define OPENVINO_FRAMEWORK_MAP(FRAMEWORK, ...)                                                           \
    template <typename T>                                                                                \
    struct __openvino_framework_map_helper_##FRAMEWORK {                                                 \
        static auto get() -> std::shared_ptr<ov::frontend::FRAMEWORK::OpExtension<T>> {                  \
            auto make_spec_tuple = [](const std::string& s = "",                                         \
                                      const std::map<std::string, std::string>& attr_mp = {},            \
                                      const std::map<std::string, ov::Any>& val_mp = {}) {               \
                return std::make_tuple(s, attr_mp, val_mp);                                              \
            };                                                                                           \
            auto params = make_spec_tuple(__VA_ARGS__);                                                  \
            const auto& name = std::get<0>(params);                                                      \
            const auto& attr_mp = std::get<1>(params);                                                   \
            const auto& val_mp = std::get<2>(params);                                                    \
            if (!name.empty())                                                                           \
                return std::make_shared<ov::frontend::FRAMEWORK::OpExtension<T>>(name, attr_mp, val_mp); \
            return std::make_shared<ov::frontend::FRAMEWORK::OpExtension<T>>(attr_mp, val_mp);           \
        }                                                                                                \
    };

}  // namespace frontend
}  // namespace ov
