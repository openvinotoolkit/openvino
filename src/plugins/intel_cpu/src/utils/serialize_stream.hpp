// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "serialize_base.hpp"

namespace ov {
namespace intel_cpu {

class ModelStreamDeserializer : public ModelDeserializerBase {
public:
    ModelStreamDeserializer(std::istream& istream, model_builder fn);

    void parse(std::shared_ptr<ov::Model>& model) override;

private:
    std::istream* m_istream;
};

}   // namespace intel_cpu
}   // namespace ov
