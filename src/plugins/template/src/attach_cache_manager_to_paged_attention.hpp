// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Template-plugin-specific pass: attaches a shared PagedCacheManager
// handle to every PagedAttentionExtension node in the model via rt_info.

#pragma once

#include "openvino/core/model.hpp"
#include "openvino/op/paged_attention.hpp"
#include "openvino/pass/pass.hpp"
#include "openvino/reference/utils/paged_cache_manager_helper.hpp"

namespace ov {
namespace pass {

class AttachCacheManagerToPagedAttention : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("AttachCacheManagerToPagedAttention");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override {
        using namespace ov::reference::paged_attention_cache;

        CacheManagerHandle shared_handle;
        ov::element::Type cache_dtype;
        bool changed = false;

        for (const auto& node : model->get_ordered_ops()) {
            auto pa = std::dynamic_pointer_cast<ov::op::PagedAttentionExtension>(node);
            if (!pa)
                continue;
            if (get_cache_manager(pa.get()))
                continue;

            if (!shared_handle) {
                cache_dtype = pa->get_input_element_type(0);
                shared_handle = make_cache_handle(cache_dtype);
            }

            OPENVINO_ASSERT(pa->get_input_element_type(0) == cache_dtype,
                            "AttachCacheManagerToPagedAttention: incompatible cache data types");

            set_cache_manager(pa.get(), shared_handle);
            changed = true;
        }
        return changed;
    }
};

}  // namespace pass
}  // namespace ov
