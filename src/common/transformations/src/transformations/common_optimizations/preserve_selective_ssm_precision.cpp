// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/preserve_selective_ssm_precision.hpp"

#include <cstddef>
#include <memory>

#include "itt.hpp"
#include "openvino/op/paged_selective_ssm.hpp"
#include "openvino/op/selective_ssm.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/disable_precision_conversion.hpp"

namespace ov::pass {

PreserveSelectiveSSMPrecision::PreserveSelectiveSSMPrecision() {
    MATCHER_SCOPE(PreserveSelectiveSSMPrecision);
    const auto selective_ssm =
        pattern::wrap_type<ov::op::internal::SelectiveSSM, ov::op::internal::PagedSelectiveSSM>();
    const matcher_pass_callback callback = [](pattern::Matcher& matcher) {
        const auto& node = matcher.get_match_root();
        // Both operations require one common data type across their data inputs. PagedSelectiveSSM additionally
        // updates its state table in place and requires one exact-width type across all metadata inputs.
        ov::disable_conversion(node, ov::element::dynamic, ov::element::dynamic);
        for (size_t input = 0; input < node->get_input_size(); ++input) {
            ov::disable_conversion(node->get_input_node_shared_ptr(input), ov::element::dynamic, ov::element::dynamic);
        }
        return false;
    };

    register_matcher(std::make_shared<pattern::Matcher>(selective_ssm, matcher_name), callback);
}

}  // namespace ov::pass
