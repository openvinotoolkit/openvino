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

    typedef std::function<std::shared_ptr<ov::Model>(const std::string&, const ov::Tensor&)> model_builder;

    void operator>>(std::shared_ptr<ov::Model>& model);

    virtual void parse(std::shared_ptr<ov::Model>& model) = 0;

protected:
    ModelDeserializerBase(model_builder fn) : m_model_builder(fn) {}

    static void set_info(pugi::xml_node& root, std::shared_ptr<ov::Model>& model);

    model_builder m_model_builder;
};

}   // namespace intel_cpu
}   // namespace ov
