// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ostream>
#include <pugixml.hpp>
#include <string>

#include "openvino/core/model.hpp"
#include "openvino/pass/serialize.hpp"

namespace ov::intel_cpu {

class ModelSerializer : private ov::pass::StreamSerialize {
public:
    using CacheEncrypt = std::function<std::string(const std::string&)>;

    explicit ModelSerializer(std::ostream& ostream, const CacheEncrypt& encrypt_fn = {}, bool weightless_mode = false);

    void operator<<(const std::shared_ptr<ov::Model>& model);

private:
    bool use_absolute_offset() override;

    std::unique_ptr<util::XmlSerializer> make_serializer(pugi::xml_node& data,
                                                         const std::string& node_type_name,
                                                         util::ConstantWriter& constant_write_handler,
                                                         int64_t version,
                                                         bool deterministic,
                                                         bool compress_to_fp16,
                                                         ov::element::Type output_element_type,
                                                         bool data_is_temporary) const override;

    bool m_weightless_mode;
};

/**
 * @brief Serializes @p model into two streams instead of the single legacy one: the constants go to @p weights as they
 * are produced, the IR XML (encrypted when @p encrypt_fn is set) to @p xml. Offsets in the XML are relative to the
 * start of @p weights.
 *
 * @param model Model to serialize, left unchanged.
 * @param weights Receives the constants.
 * @param xml Receives the IR XML.
 * @param encrypt_fn Optional encryption of the XML text.
 * @param weightless_mode Skip the constants that can be read from the original weights file on import.
 */
void serialize_model(std::shared_ptr<ov::Model> model,
                     std::ostream& weights,
                     std::ostream& xml,
                     const ModelSerializer::CacheEncrypt& encrypt_fn,
                     bool weightless_mode);

static constexpr uint64_t runtime_requirements_magic = 0x4F564350555F5252ULL;  // "OVCPU_RR" in ASCII
static constexpr uint32_t runtime_requirements_version = 1;
static constexpr uint64_t runtime_requirements_max_size = 4096;
std::string build_runtime_requirements();
bool is_runtime_requirements_compatible(const std::string& requirements);

}  // namespace ov::intel_cpu
