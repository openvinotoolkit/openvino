// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "keep_xattention_threshold_precision.hpp"

#include "intel_gpu/primitives/paged_attention.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/paged_attention.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/util/precision_sensitive_attribute.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/pp.hpp"
#include "transformations/rt_info/disable_precision_conversion.hpp"

namespace ov::intel_gpu {

KeepXAttentionThresholdPrecision::KeepXAttentionThresholdPrecision(bool prevent_bf16) {
    auto pa_m = ov::pass::pattern::wrap_type<ov::op::PagedAttentionExtension>();

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        auto pa = ov::as_type_ptr<ov::op::PagedAttentionExtension>(m.get_match_root());
        if (!pa) {
            return false;
        }

        if (transformation_callback(pa)) {
            return false;
        }

        using InputIdx = cldnn::paged_attention::PagedAttentionInputIdx;
        constexpr auto thr_idx = InputIdx::XATTENTION_THRESHOLD;
        ov::mark_as_precision_sensitive(pa->input(thr_idx));

        if (prevent_bf16) {
            for (const auto idx : {InputIdx::ROTATION_TRIG_LUT, thr_idx}) {
                const auto source = pa->input_value(idx);
                const auto type = source.get_element_type();
                if (type.is_dynamic()) {
                    continue;
                }

                const auto producer = source.get_node_shared_ptr();
                if (ov::is_type<ov::op::v0::Constant>(producer) || ov::is_type<ov::op::v0::Parameter>(producer)) {
                    ov::disable_conversion(producer, type, ov::element::bf16);
                } else if (!ov::is_conversion_disabled(producer, type, ov::element::bf16) ||
                           !ov::is_type<ov::op::v0::Convert>(producer)) {
                    auto convert = std::make_shared<ov::op::v0::Convert>(source, type);
                    ov::copy_runtime_info(producer, convert);
                    ov::disable_conversion(convert, type, ov::element::bf16);
                    pa->input(idx).replace_source_output(convert);
                }
            }
        }
        return true;
    };

    auto matcher = std::make_shared<ov::pass::pattern::Matcher>(pa_m, "KeepXAttentionThresholdPrecision");
    register_matcher(matcher, callback);
}

}  // namespace ov::intel_gpu
