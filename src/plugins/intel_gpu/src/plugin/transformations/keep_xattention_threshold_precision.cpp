// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "keep_xattention_threshold_precision.hpp"

#include "intel_gpu/primitives/paged_attention.hpp"
#include "openvino/op/paged_attention.hpp"
#include "openvino/op/util/precision_sensitive_attribute.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace ov::intel_gpu {

KeepXAttentionThresholdPrecision::KeepXAttentionThresholdPrecision() {
    auto pa_m = ov::pass::pattern::wrap_type<ov::op::PagedAttentionExtension>();

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        auto pa = ov::as_type_ptr<ov::op::PagedAttentionExtension>(m.get_match_root());
        if (!pa)
            return false;

        if (transformation_callback(pa))
            return false;

        constexpr size_t thr_idx = cldnn::paged_attention::PagedAttentionInputIdx::XATTENTION_THRESHOLD;
        ov::mark_as_precision_sensitive(pa->input(thr_idx));
        return true;
    };

    auto matcher = std::make_shared<ov::pass::pattern::Matcher>(pa_m, "KeepXAttentionThresholdPrecision");
    register_matcher(matcher, callback);
}

}  // namespace ov::intel_gpu

