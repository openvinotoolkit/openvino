// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "disable_fp16_comp_for_all_rms.hpp"

#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "ov_ops/rms.hpp"
#include "transformations/utils/utils.hpp"
#include "transformations/rt_info/disable_precision_conversion.hpp"

namespace ov::intel_gpu {

DisableFP16CompForAllRMS::DisableFP16CompForAllRMS() {
    auto rms_pattern = ov::pass::pattern::wrap_type<ov::op::internal::RMS>();

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        auto rms = m.get_match_root();
        if (transformation_callback(rms))
            return false;

        if (ov::is_conversion_disabled(rms, ov::element::f16))
            return false;

        ov::disable_conversion(rms, ov::element::f16);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(rms_pattern, "DisableFP16CompForAllRMS");
    this->register_matcher(m, callback);
}

}  // namespace ov::intel_gpu
