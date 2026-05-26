// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "disable_fp16_comp_cumsum_sin_gen.hpp"

#include "openvino/op/cum_sum.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/sin.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/disable_fp16_compression.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::intel_gpu {

// See header for the F0 oscillator pattern and motivation.
DisableFP16CompCumSumSinGen::DisableFP16CompCumSumSinGen() {
    using namespace ov::pass::pattern;

    auto cumsum_m = wrap_type<ov::op::v0::CumSum>({any_input(), any_input()});
    auto mul1_m = wrap_type<ov::op::v1::Multiply>({cumsum_m, any_input()});
    auto transpose2_m = wrap_type<ov::op::v1::Transpose>({mul1_m, any_input()});
    auto mul2_m = wrap_type<ov::op::v1::Multiply>({transpose2_m, any_input()});
    auto interpolate_m =
        wrap_type<ov::op::v4::Interpolate, ov::op::v11::Interpolate>({mul2_m, any_input(), any_input()});
    auto transpose3_m = wrap_type<ov::op::v1::Transpose>({interpolate_m, any_input()});
    auto sin_m = wrap_type<ov::op::v0::Sin>({transpose3_m});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto sin_node = pattern_map.at(sin_m).get_node_shared_ptr();
        if (transformation_callback(sin_node))
            return false;

        auto cumsum_node = pattern_map.at(cumsum_m).get_node_shared_ptr();

        // Also tag the producer feeding CumSum's first input.
        auto cumsum_input = cumsum_node->input_value(0).get_node_shared_ptr();
        if (cumsum_input)
            ov::disable_fp16_compression(cumsum_input);

        for (const auto& key : {cumsum_m, mul1_m, transpose2_m, mul2_m, interpolate_m, transpose3_m, sin_m}) {
            ov::disable_fp16_compression(pattern_map.at(key).get_node_shared_ptr());
        }
        return true;
    };

    auto m = std::make_shared<Matcher>(sin_m, "DisableFP16CompCumSumSinGen");
    this->register_matcher(m, callback);
}

}  // namespace ov::intel_gpu
