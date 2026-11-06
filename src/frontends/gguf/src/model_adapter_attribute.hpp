// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/runtime_attribute.hpp"
#include "openvino/frontend/gguf/model_adapter.hpp"

namespace ov::frontend::gguf {

class ModelAdapterAttribute : public ov::RuntimeAttribute {
public:
    OPENVINO_RTTI("gguf.model_adapter", "0", ov::RuntimeAttribute);

    explicit ModelAdapterAttribute(std::shared_ptr<ModelAdapter> adapter) : adapter(std::move(adapter)) {}

    bool visit_attributes(ov::AttributeVisitor&) override {
        return false;
    }

    bool is_deterministic() const override {
        return false;
    }

    std::shared_ptr<ModelAdapter> adapter;
};

inline std::shared_ptr<ModelAdapter> get_model_adapter(const std::shared_ptr<ov::Node>& node) {
    const auto found = node->get_rt_info().find(ModelAdapterAttribute::get_type_info_static());
    return found == node->get_rt_info().end() ? nullptr : found->second.as<ModelAdapterAttribute>().adapter;
}

inline void remove_model_adapters(const std::shared_ptr<ov::Model>& model) {
    for (const auto& node : model->get_ops())
        node->get_rt_info().erase(ModelAdapterAttribute::get_type_info_static());
}

}  // namespace ov::frontend::gguf
