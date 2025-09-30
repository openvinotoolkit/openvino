// SPDX-License-Identifier: Apache-2.0
#include "transformations/attach_cache_manager_to_paged_attention.hpp"

#include <algorithm>
#include <memory>
#include <vector>

#include "openvino/core/model.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/paged_attention.hpp"

#include "itt.hpp"
#include "openvino/core/descriptor_tensor.hpp"
#include "openvino/op/util/multi_subgraph_base.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/util/common_util.hpp"



#include "openvino/core/cache_manager.hpp"  // your class in ov::internal

namespace ov {
namespace pass {

namespace {

// Helper: return bytes per element for a given element type (throws on dynamic).
static inline size_t get_element_size_in_bytes(const ov::element::Type& et) {
    const auto s = et.size();
    if (s == 0) {
        throw std::runtime_error("PagedAttention CacheManager: element type has unknown byte size (dynamic?)");
    }
    return static_cast<size_t>(s);
}

// Helper: read a static 4D shape from an input port. Returns empty vector if not static/4D.
static inline std::vector<size_t> try_get_static_4d_shape(const ov::Output<ov::Node>& port) {
    const auto& pshape = port.get_partial_shape();
    if (!pshape.rank().is_static() || pshape.rank().get_length() != 4 || !pshape.is_static()) {
        return {};
    }
    const auto shape = pshape.to_shape();
    return {shape.begin(), shape.end()};
}

// Decide CacheManager sizing from the first PagedAttention’s cache shapes.
// - page == one block (across all heads)
// - page_bytes = num_heads * block_size * max(key_head_size, value_head_size) * elem_size
// - total_bytes = page_bytes * num_blocks
struct CacheSizing {
    ov::element::Type elem_type{};
    size_t page_bytes{0};
    size_t total_bytes{0};
    size_t alignment_bytes{64};
    // For sanity checks (not used by CacheManager directly):
    size_t num_blocks{0}, num_heads{0}, block_size{0}, key_head_size{0}, value_head_size{0};
};

static inline CacheSizing derive_cache_sizing_from_node(const std::shared_ptr<ov::Node>& n) {
    // Indexing follows the typical PA signature:
    // 0: Q, 1: K, 2: V, 3: key_cache, 4: value_cache, ...
    auto elem_type = n->get_input_element_type(0);  // Q dtype; K/V expected the same
    const size_t elem_size = get_element_size_in_bytes(elem_type);

    const auto kc4 = try_get_static_4d_shape(n->input_value(3));  // [num_blocks, num_heads, block_size, k_head_size]
    const auto vc4 = try_get_static_4d_shape(n->input_value(4));  // [num_blocks, num_heads, block_size, v_head_size]

    if (kc4.empty() || vc4.empty()) {
        throw std::runtime_error("PagedAttention CacheManager: key/value cache shapes must be static 4D");
    }
    if (kc4[0] != vc4[0] || kc4[1] != vc4[1] || kc4[2] != vc4[2]) {
        throw std::runtime_error("PagedAttention CacheManager: key/value cache shapes mismatch");
    }

    const size_t num_blocks      = kc4[0];
    const size_t num_heads       = kc4[1];
    const size_t block_size      = kc4[2];
    const size_t key_head_size   = kc4[3];
    const size_t value_head_size = vc4[3];

    const size_t head_width_max  = std::max(key_head_size, value_head_size);

    const size_t page_bytes  = num_heads * block_size * head_width_max * elem_size;
    const size_t total_bytes = page_bytes * num_blocks;

    CacheSizing s;
    s.elem_type        = elem_type;
    s.page_bytes       = page_bytes;
    s.total_bytes      = total_bytes;
    s.num_blocks       = num_blocks;
    s.num_heads        = num_heads;
    s.block_size       = block_size;
    s.key_head_size    = key_head_size;
    s.value_head_size  = value_head_size;
    return s;
}

} // namespace

bool AttachCacheManagerToPagedAttention::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_MODEL_SCOPE(AttachCacheManagerToPagedAttention);

    std::shared_ptr<ov::internal::CacheManager> shared_cache_manager;

    CacheSizing sizing{};
    bool sizing_initialized = false;
    bool graph_modified = false;

    for (const auto& node : model->get_ordered_ops()) {
        // We expect PagedAttention to live in ov::internal
        auto pa = std::dynamic_pointer_cast<ov::op::PagedAttentionExtension>(node);
        if (!pa) {
            continue;
        }

        // If this PA already has a CacheManager bound, skip it.
        if (pa->get_cache_manager()) {
            graph_modified = true; // we still consider the pass touched the graph logically
            continue;
        }

        // Initialize the shared CacheManager from the first encountered PA.
        if (!sizing_initialized) {
            sizing = derive_cache_sizing_from_node(pa);
            shared_cache_manager = std::make_shared<ov::internal::CacheManager>(
                sizing.elem_type,
                /*total_bytes=*/sizing.total_bytes,
                /*page_bytes=*/sizing.page_bytes,
                /*alignment_bytes=*/64);
            sizing_initialized = true;
        }

        // Optional compatibility check (defensive): ensure other PAs match the first one.
        {
            const auto s_other = derive_cache_sizing_from_node(pa);
            const bool compatible =
                (s_other.elem_type == sizing.elem_type) &&
                (s_other.page_bytes == sizing.page_bytes) &&
                (s_other.total_bytes == sizing.total_bytes);
            if (!compatible) {
                throw std::runtime_error(
                    "AttachCacheManagerToPagedAttention: multiple PagedAttention nodes with incompatible cache "
                    "layouts were found. Either harmonize their cache shapes or extend the pass to allocate distinct "
                    "CacheManagers per group.");
            }
        }

        // Attach the shared CacheManager to this PagedAttention.
        pa->set_cache_manager(shared_cache_manager);

        // (Optional) If your PA has an API to register itself and store a handle, you can do:
        // auto handle = shared_cache_manager->register_operator(pa->output(0));
        // pa->set_cache_handle(handle);

        graph_modified = true;
    }

    return graph_modified;
}

}  // namespace pass
}  // namespace ov
