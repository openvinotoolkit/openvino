// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ostream>
#include <pugixml.hpp>
#include <string>

#include "openvino/core/model.hpp"
#include "openvino/core/type/element_type.hpp"
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

static constexpr uint64_t runtime_requirements_magic = 0x4F564350555F5252ULL;  // "OVCPU_RR" in ASCII
static constexpr uint32_t runtime_requirements_version = 2;
static constexpr uint64_t runtime_requirements_max_size = 4096;
// inference_precision controls what precision flags are stored:
//   f32      -> bf16=0;f16=0 (no special hardware needed, compatible everywhere)
//   bf16     -> bf16=1;f16=0
//   f16      -> bf16=0;f16=1
//   dynamic  -> hardware capabilities (conservative: store what the machine can use)
std::string build_runtime_requirements(ov::element::Type inference_precision = ov::element::dynamic);
bool is_runtime_requirements_compatible(const std::string& requirements);

}  // namespace ov::intel_cpu
