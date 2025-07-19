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

    ModelSerializer(std::ostream& ostream, const CacheEncrypt& encrypt_fn = {});

    void operator<<(const std::shared_ptr<ov::Model>& model);

private:
    bool use_absolute_offset() override;
};

class ModelDeserializer {
public:
    using ModelBuilder = std::function<std::shared_ptr<ov::Model>(const std::shared_ptr<ov::AlignedBuffer>&,
                                                                  const std::shared_ptr<ov::AlignedBuffer>&)>;

    ModelDeserializer(std::shared_ptr<ov::AlignedBuffer>& model_buffer,
                      ModelBuilder fn,
                      const CacheDecrypt& decrypt_fn,
                      bool decript_from_string);

    ModelDeserializer(std::istream& model_stream,
                      ModelBuilder fn,
                      const CacheDecrypt& decrypt_fn,
                      bool decript_from_string);

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
};

}  // namespace ov::intel_cpu
