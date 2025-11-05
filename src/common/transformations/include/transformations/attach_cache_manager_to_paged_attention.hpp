// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API AttachCacheManagerToPagedAttention;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief Model pass that finds ov::util::PagedAttention nodes, constructs a single
 *        ov::util::PagedCacheManager (assumes same data type for all PagedAttention nodes),
 *        and attaches it to every PagedAttention via set_cache_manager(...).
 */
class ov::pass::AttachCacheManagerToPagedAttention : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("AttachCacheManagerToPagedAttention");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};
