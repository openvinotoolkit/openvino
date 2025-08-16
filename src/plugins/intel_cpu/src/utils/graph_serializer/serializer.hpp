// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <istream>
#include <memory>
#include <ostream>
#include <pugixml.hpp>
#include <string>
#include <variant>

#include "openvino/core/model.hpp"
#include "openvino/pass/serialize.hpp"
#include "openvino/runtime/aligned_buffer.hpp"
#include "utils/codec_xor.hpp"

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

}  // namespace ov::intel_cpu
