// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "serialize_base.hpp"

#include "openvino/util/mmap_object.hpp"

namespace ov {
namespace intel_cpu {

class ModelMmapDeserializer : public ModelDeserializerBase {
public:
    ModelMmapDeserializer(const std::shared_ptr<ov::MappedMemory>& buffer, model_builder fn);

    void parse(std::shared_ptr<ov::Model>& model) override;

private:
    std::shared_ptr<ov::MappedMemory> m_model_buffer;
};

}   // namespace intel_cpu
}   // namespace ov
