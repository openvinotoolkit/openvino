// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <memory>
#include "openvino/pass/pass.hpp"
#include "transformations_visibility.hpp"
#include "openvino/pass/graph_rewrite.hpp"

// Forward decls to keep the header light.
namespace ov { class Model; }

namespace ov {
namespace pass {

/**
 * @brief Model pass that finds ov::internal::PagedAttention nodes, constructs a single
 *        ov::internal::CacheManager (sized from the first PA’s cache shapes and dtype),
 *        and attaches it to every PagedAttention via set_cache_manager(...).
 *
 * The pass is idempotent: if a PA already has a CacheManager, it leaves it in place.
 */
class AttachCacheManagerToPagedAttention : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("AttachCacheManagerToPagedAttention");

    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace pass
}  // namespace ov
