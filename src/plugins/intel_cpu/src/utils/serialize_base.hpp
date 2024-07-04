// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <pugixml.hpp>

#include "openvino/core/model.hpp"

namespace ov {
namespace intel_cpu {

class ModelSerializer {
public:
    ModelSerializer(std::ostream& ostream);

    void operator<<(const std::shared_ptr<ov::Model>& model);

private:
    std::ostream& m_ostream;
};

class ModelDeserializerBase {
public:
    virtual ~ModelDeserializerBase() = default;

    void operator>>(std::shared_ptr<ov::Model>& model);

    virtual void parse(std::shared_ptr<ov::Model>& model) = 0;

protected:
    ModelDeserializerBase() {}

    static void set_info(pugi::xml_node& root, std::shared_ptr<ov::Model>& model);
};

}   // namespace intel_cpu
}   // namespace ov
