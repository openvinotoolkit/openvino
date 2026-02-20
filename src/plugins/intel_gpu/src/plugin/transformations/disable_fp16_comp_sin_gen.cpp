// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "disable_fp16_comp_sin_gen.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/sin.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"
#include "openvino/core/graph_util.hpp"
#include "transformations/symbolic_transformations/symbolic_optimizations.hpp"

namespace ov::intel_gpu {
DisableFP16ComSinGenPatternForHiFiGAN::DisableFP16ComSinGenPatternForHiFiGAN() {
    using namespace ov::pass::pattern;
    using ov::op::v0::Sin;
    using ov::op::v1::Multiply;
    using ov::op::v1::Transpose;

    // SineGen of HiFiGAN(https://github.com/FunAudioLLM/CosyVoice/blob/1dcc59676fe3fa863f983ab7820e481560c73be7/cosyvoice/hifigan/generator.py#L157-L189)
    // could make inf in fp16 because of large input value multiplication (e.g. hop_length=480 makes multiply x480)
    // So keep fp32 from Multiply x480 to Sin to avoid inf in fp16
    auto multiply = wrap_type<Multiply>();
    // This pass is called after ConvertToInterpolateV4 passes. So consider only v4 here.
    auto interpolate_v0 = wrap_type<ov::op::v0::Interpolate>({multiply, any_input()});
    auto interpolate_v4 = wrap_type<ov::op::v4::Interpolate>({multiply, any_input(), any_input()});
    auto interpolate_v4_with_axes = wrap_type<ov::op::v4::Interpolate>({multiply, any_input(), any_input(), any_input()});
    auto interpolate_v11 = wrap_type<ov::op::v11::Interpolate>({multiply, any_input()});
    auto interpolate_v11_with_axes = wrap_type<ov::op::v11::Interpolate>({multiply, any_input(), any_input()});

    auto interpolate = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{interpolate_v0, interpolate_v4, interpolate_v4_with_axes, interpolate_v11, interpolate_v11_with_axes});
    auto transpose = wrap_type<Transpose>({interpolate, any_input()});
    auto sin = wrap_type<Sin>({transpose});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto sin_node = pattern_map.at(sin).get_node_shared_ptr();
        auto transpose_node = pattern_map.at(transpose).get_node_shared_ptr();
        auto interpolate_node = pattern_map.at(interpolate).get_node_shared_ptr();
        auto multiply_node = pattern_map.at(multiply).get_node_shared_ptr();

        if (transformation_callback(sin_node)) return false;

        for (const auto& node : {multiply_node, interpolate_node, transpose_node, sin_node}) {
            ov::disable_fp16_compression(node);
        }
        
        return true;
    };

    auto m = std::make_shared<Matcher>(sin, "DisableFP16ComSinGenPatternForHiFiGAN");
    this->register_matcher(m, callback);
}
}  // namespace ov::intel_gpu
