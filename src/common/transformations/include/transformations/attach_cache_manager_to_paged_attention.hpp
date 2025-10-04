// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <memory>

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {
/**
 * @brief Model pass that finds ov::internal::PagedAttention nodes, constructs a single
 *        ov::internal::CacheManager (assumes same data type for all PagedAttention nodes),
 *        and attaches it to every PagedAttention via set_cache_manager(...).
 */
class AttachCacheManagerToPagedAttention : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("AttachCacheManagerToPagedAttention");

    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};
}  // namespace pass
}  // namespace ov
