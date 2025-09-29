// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <istream>
#include <pugixml.hpp>
#include <string>
#include <variant>

#include "../xml_util/include/openvino/xml_util/xml_deserialize_util.hpp"
#include "openvino/core/model.hpp"
#include "openvino/runtime/aligned_buffer.hpp"
#include "openvino/util/xml_parse_utils.hpp"
#include "utils/codec_xor.hpp"

namespace ov::intel_cpu {

template <class T>
bool getParameters(const pugi::xml_node& node, const std::string& name, std::vector<T>& value) {
    ov::util::str_to_container(ov::util::pugixml::get_str_attr(node, name.c_str()), value);
    return true;
}

class XmlDeserializer : public ov::util::XmlDeserializer {
public:
    explicit XmlDeserializer(const pugi::xml_node& node,
                             const std::shared_ptr<ov::AlignedBuffer>& weights,
                             const std::shared_ptr<ov::AlignedBuffer>& origin_weights,
                             const std::unordered_map<std::string, ov::OpSet>& opsets,
                             const std::unordered_map<ov::DiscreteTypeInfo, ov::BaseOpExtension::Ptr>& extensions,
                             std::unordered_map<std::string, std::shared_ptr<ov::op::util::Variable>>& variables,
                             size_t version)
        : ov::util::XmlDeserializer(node, weights, opsets, extensions, variables, version),
          m_origin_weights{origin_weights} {}

    explicit XmlDeserializer(const pugi::xml_node& node,
                             const std::shared_ptr<ov::AlignedBuffer>& weights,
                             const std::unordered_map<std::string, ov::OpSet>& opsets,
                             const std::unordered_map<ov::DiscreteTypeInfo, ov::BaseOpExtension::Ptr>& extensions,
                             std::unordered_map<std::string, std::shared_ptr<ov::op::util::Variable>>& variables,
                             size_t version)
        : XmlDeserializer(node, weights, nullptr, opsets, extensions, variables, version) {}

protected:
    ov::Any parse_weightless_cache_attribute(const pugi::xml_node& node) const override;

    void set_constant_num_buffer(ov::AttributeAdapter<std::shared_ptr<ov::AlignedBuffer>>& adapter) override;

private:
    std::unique_ptr<ov::util::XmlDeserializer> make_visitor(
        const pugi::xml_node& node,
        const std::shared_ptr<ov::AlignedBuffer>& weights,
        const std::unordered_map<std::string, ov::OpSet>& opsets,
        const std::unordered_map<ov::DiscreteTypeInfo, ov::BaseOpExtension::Ptr>& extensions,
        std::unordered_map<std::string, std::shared_ptr<ov::op::util::Variable>>& variables,
        size_t version) const override {
        return std::make_unique<XmlDeserializer>(node,
                                                 weights,
                                                 m_origin_weights,
                                                 opsets,
                                                 extensions,
                                                 variables,
                                                 version);
    }

    std::shared_ptr<ov::AlignedBuffer> m_origin_weights;
};

class ModelDeserializer {
public:
    using ModelBuilder = std::function<std::shared_ptr<ov::Model>(const std::shared_ptr<ov::AlignedBuffer>&,
                                                                  const std::shared_ptr<ov::AlignedBuffer>&,
                                                                  const std::shared_ptr<ov::AlignedBuffer>&)>;

    ModelDeserializer(std::shared_ptr<ov::AlignedBuffer>& model_buffer,
                      ModelBuilder fn,
                      const CacheDecrypt& decrypt_fn,
                      bool decript_from_string,
                      std::string origin_weights_path = "");

    ModelDeserializer(std::istream& model_stream,
                      ModelBuilder fn,
                      const CacheDecrypt& decrypt_fn,
                      bool decript_from_string,
                      std::string origin_weights_path = "");

    virtual ~ModelDeserializer() = default;

    void operator>>(std::shared_ptr<ov::Model>& model);

protected:
    static void set_info(pugi::xml_node& root, std::shared_ptr<ov::Model>& model);

    void process_model(std::shared_ptr<ov::Model>& model, const std::shared_ptr<ov::AlignedBuffer>& model_buffer);
    void process_model(std::shared_ptr<ov::Model>& model, std::reference_wrapper<std::istream> model_stream);

    std::variant<std::shared_ptr<ov::AlignedBuffer>, std::reference_wrapper<std::istream>> m_model;
    ModelBuilder m_model_builder;
    CacheDecrypt m_cache_decrypt;
    bool m_decript_from_string;
    std::string m_origin_weights_path;
};

}  //  namespace ov::intel_cpu
