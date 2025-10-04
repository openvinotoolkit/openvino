// SPDX-License-Identifier: Apache-2.0
#include "transformations/attach_cache_manager_to_paged_attention.hpp"

#include <algorithm>
#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/core/descriptor_tensor.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/paged_cache_manager.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/paged_attention.hpp"
#include "openvino/op/util/multi_subgraph_base.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/util/common_util.hpp"

namespace ov {
namespace pass {

bool AttachCacheManagerToPagedAttention::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_MODEL_SCOPE(AttachCacheManagerToPagedAttention);

    std::shared_ptr<ov::internal::PagedCacheManager> shared_cache_manager;

    bool graph_modified = false;

    for (const auto& node : model->get_ordered_ops()) {
        auto pa = std::dynamic_pointer_cast<ov::op::PagedAttentionExtension>(node);
        if (!pa) {
            continue;
        }

        // If this PA already has a CacheManager bound, skip it
        if (pa->get_cache_manager()) {
            continue;
        }

        // Initialize the shared CacheManager from the first encountered PA
        if (!shared_cache_manager) {
            shared_cache_manager = std::make_shared<ov::internal::PagedCacheManager>(pa->get_input_element_type(0));
        }

        // Compatibility check: ensure every PAs dtype matches the dtype of cache.
        if (pa->get_input_element_type(0) != shared_cache_manager->get_element_type()) {
            throw std::runtime_error(
                "AttachCacheManagerToPagedAttention: multiple PagedAttention nodes with incompatible cache "
                "data types were found, which is not supported");
        }

        // Attach the shared CacheManager to this PagedAttention.
        pa->set_cache_manager(shared_cache_manager);

        graph_modified = true;
    }

    return graph_modified;
}

}  // namespace pass
}  // namespace ov
