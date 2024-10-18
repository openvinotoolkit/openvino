// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <pugixml.hpp>

#include "openvino/core/model.hpp"
#include "openvino/pass/serialize.hpp"
#include "openvino/runtime/aligned_buffer.hpp"
#include "openvino/util/mmap_object.hpp"
#include "utils/codec_xor.hpp"

namespace ov {
namespace intel_cpu {

class ModelSerializer {
public:
    typedef std::function<std::string(const std::string&)> CacheEncrypt;

    ModelSerializer(std::ostream& ostream, CacheEncrypt encrypt_fn = {});

    void operator<<(const std::shared_ptr<ov::Model>& model);

private:
    std::ostream& m_ostream;
    CacheEncrypt m_cache_encrypt;
};

class ModelDeserializer {
public:
    typedef std::function<std::shared_ptr<ov::Model>(const std::shared_ptr<ov::AlignedBuffer>&, const std::shared_ptr<ov::AlignedBuffer>&)> ModelBuilder;

    ModelDeserializer(std::istream& model, ModelBuilder fn, const CacheDecrypt& encrypt_fn, bool decript_from_string);

    virtual ~ModelDeserializer() = default;

    void operator>>(std::shared_ptr<ov::Model>& model);

protected:
    static void set_info(pugi::xml_node& root, std::shared_ptr<ov::Model>& model);

    void process_mmap(std::shared_ptr<ov::Model>& model, const std::shared_ptr<ov::AlignedBuffer>& memory);

    void process_stream(std::shared_ptr<ov::Model>& model);

    std::istream& m_istream;
    ModelBuilder m_model_builder;
    CacheDecrypt m_cache_decrypt;
    bool m_decript_from_string;
};

}   // namespace intel_cpu
}   // namespace ov
