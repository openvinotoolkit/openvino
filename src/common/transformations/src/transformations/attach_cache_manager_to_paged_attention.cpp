// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/attach_cache_manager_to_paged_attention.hpp"

#include <algorithm>
#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/core/descriptor_tensor.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/paged_attention.hpp"
#include "openvino/op/util/multi_subgraph_base.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/util/common_util.hpp"

bool ov::pass::AttachCacheManagerToPagedAttention::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_FUNCTION_SCOPE(AttachCacheManagerToPagedAttention);

    ov::op::PagedAttentionExtension::PagedCacheManagerHandle shared_cache_manager;
    ov::element::Type cache_manager_dtype;
    bool changed = false;

    for (const auto& node : model->get_ordered_ops()) {
        auto pa = std::dynamic_pointer_cast<ov::op::PagedAttentionExtension>(node);
        if (!pa) {
            continue;
        }

        if (pa->get_cache_manager()) {
            continue;
        }

        if (!shared_cache_manager) {
            cache_manager_dtype = pa->get_input_element_type(0);
            shared_cache_manager = ov::op::make_paged_cache_handle(cache_manager_dtype);
        }

        // All PA nodes in the same model must share the same compute dtype
        if (pa->get_input_element_type(0) != cache_manager_dtype) {
            OPENVINO_THROW("AttachCacheManagerToPagedAttention: multiple PagedAttention nodes with incompatible cache "
                           "data types were found, which is not supported");
        }

        pa->set_cache_manager(shared_cache_manager);
        changed = true;
    }

    return changed;
}
