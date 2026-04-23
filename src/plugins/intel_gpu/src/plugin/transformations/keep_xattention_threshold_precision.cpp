// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "keep_xattention_threshold_precision.hpp"

#include "intel_gpu/primitives/paged_attention.hpp"

#include "openvino/op/paged_attention.hpp"
#include "openvino/op/util/precision_sensitive_attribute.hpp"

namespace ov::intel_gpu {

KeepXAttentionThresholdPrecision::KeepXAttentionThresholdPrecision() {
    auto pa_m = ov::pass::pattern::wrap_type<ov::op::PagedAttentionExtension>();
    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        auto pa = ov::as_type_ptr<ov::op::PagedAttentionExtension>(m.get_match_root());
        if (transformation_callback(pa))
            return false;
        const size_t thr_idx = cldnn::paged_attention::PagedAttentionInputIdx::XATTENTION_THRESHOLD;
        ov::mark_as_precision_sensitive(pa->input(thr_idx));
        return true;
    }
}
    bool changed = false;

    for (const auto& node : model->get_ops()) {
        auto pa = ov::as_type_ptr<ov::op::PagedAttentionExtension>(node);
        if (!pa)
            continue;

        const size_t thr_idx = cldnn::paged_attention::PagedAttentionInputIdx::XATTENTION_THRESHOLD;
        if (thr_idx >= pa->get_input_size())
            continue;

        const auto et = pa->get_input_element_type(thr_idx);
        if (!et.is_real())
            continue;

        ov::mark_as_precision_sensitive(pa->input(thr_idx));
        changed = true;
    }

    return changed;
}

}  // namespace ov::intel_gpu
