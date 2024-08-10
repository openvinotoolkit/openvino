// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <pugixml.hpp>

#include "openvino/core/model.hpp"
#include "openvino/pass/serialize.hpp"
#include "openvino/runtime/aligned_buffer.hpp"
#include "openvino/util/mmap_object.hpp"

namespace ov {
namespace intel_cpu {

class ModelSerializer {
public:
    ModelSerializer(std::ostream& ostream);

    void operator<<(const std::shared_ptr<ov::Model>& model);

private:
    std::ostream& m_ostream;
};

class ModelDeserializer {
public:
    typedef std::function<std::shared_ptr<ov::Model>(const std::shared_ptr<ov::AlignedBuffer>&, const std::shared_ptr<ov::AlignedBuffer>&)> ModelBuilder;

    ModelDeserializer(std::istream& model, ModelBuilder fn);

    virtual ~ModelDeserializer() = default;

    void operator>>(std::shared_ptr<ov::Model>& model);

protected:
    static void set_info(pugi::xml_node& root, std::shared_ptr<ov::Model>& model);

    inline void process_mmap(std::shared_ptr<ov::Model>& model, const std::shared_ptr<ov::MappedMemory>& mmemory);

    inline void process_stream(std::shared_ptr<ov::Model>& model, const std::istream& mmemory);

    std::istream& m_istream;
    ModelBuilder m_model_builder;
};

}   // namespace intel_cpu
}   // namespace ov
