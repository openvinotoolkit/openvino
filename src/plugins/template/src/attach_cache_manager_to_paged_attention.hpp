// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Attaches a shared PagedCacheManager to every PA node via rt_info.

#pragma once

#include "openvino/core/model.hpp"
#include "openvino/op/paged_attention.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/reference/utils/paged_cache_manager_helper.hpp"

namespace ov {
namespace pass {

class AttachCacheManagerToPagedAttention : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("AttachCacheManagerToPagedAttention");
    AttachCacheManagerToPagedAttention() {
        using namespace ov::reference::paged_attention_cache;

        auto pa_pattern = pattern::wrap_type<ov::op::PagedAttentionExtension>();

        auto shared_handle = std::make_shared<CacheManagerHandle>();
        auto shared_dtype = std::make_shared<ov::element::Type>();

        ov::matcher_pass_callback callback = [shared_handle, shared_dtype](pattern::Matcher& m) -> bool {
            auto pa = std::dynamic_pointer_cast<ov::op::PagedAttentionExtension>(m.get_match_root());
            if (!pa || get_cache_manager(pa.get()) != nullptr)
                return false;

            if (!*shared_handle) {
                *shared_dtype = pa->get_input_element_type(3);
                *shared_handle = make_cache_handle(*shared_dtype);
            }

            OPENVINO_ASSERT(pa->get_input_element_type(3) == *shared_dtype,
                            "AttachCacheManagerToPagedAttention: incompatible cache data types");

            set_cache_manager(pa.get(), *shared_handle);
            return true;
        };

        auto m = std::make_shared<pattern::Matcher>(pa_pattern, "AttachCacheManagerToPagedAttention");
        register_matcher(m, callback);
    }
};

}  // namespace pass
}  // namespace ov
