// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "keep_gqa_kv_scale_precision.hpp"

#include "openvino/op/group_query_attention.hpp"
#include "openvino/op/util/precision_sensitive_attribute.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace ov::intel_gpu {

KeepGQAKVScalePrecision::KeepGQAKVScalePrecision() {
    auto gqa_m = ov::pass::pattern::wrap_type<ov::op::internal::GroupQueryAttention>();

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        auto gqa = ov::as_type_ptr<ov::op::internal::GroupQueryAttention>(m.get_match_root());
        if (!gqa)
            return false;

        if (transformation_callback(gqa))
            return false;

        // Only a quantized KV cache has scales at inputs 12 (k_scale) / 13 (v_scale).
        if (!gqa->is_kv_quantized())
            return false;

        constexpr size_t k_scale_idx = 12;
        constexpr size_t v_scale_idx = 13;
        ov::mark_as_precision_sensitive(gqa->input(k_scale_idx));
        ov::mark_as_precision_sensitive(gqa->input(v_scale_idx));
        return true;
    };

    auto matcher = std::make_shared<ov::pass::pattern::Matcher>(gqa_m, "KeepGQAKVScalePrecision");
    register_matcher(matcher, callback);
}

}  // namespace ov::intel_gpu
