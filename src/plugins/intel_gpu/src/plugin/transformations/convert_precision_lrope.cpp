// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_precision_lrope.hpp"

#include "intel_gpu/op/gemm.hpp"

#include "openvino/op/broadcast.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/sin.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::intel_gpu {

ConvertPrecisionLRoPE::ConvertPrecisionLRoPE() {
    using namespace ov::pass::pattern;

    auto data_const_m = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto concat_list_m = wrap_type<ov::op::v0::Concat>({any_input(), any_input(), any_input()}, type_matches(element::i32));
    auto broadcast_m = wrap_type<ov::op::v3::Broadcast>({data_const_m, concat_list_m}, type_matches(element::f16));
    auto reshape_m = wrap_type<ov::op::v1::Reshape>({any_input(), wrap_type<ov::op::v0::Constant>()}, type_matches(element::i32));
    auto convert_m = wrap_type<ov::op::v0::Convert>({reshape_m}, type_matches(element::f16));
    auto gemm_m = wrap_type<op::Gemm>({broadcast_m, convert_m});
    auto transpose_m = wrap_type<ov::op::v1::Reshape>({gemm_m, wrap_type<ov::op::v0::Constant>()}, type_matches(element::f16));
    auto concat_m = wrap_type<ov::op::v0::Concat>({transpose_m, transpose_m}, type_matches(element::f16));

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        auto concat = ov::as_type_ptr<ov::op::v0::Concat>(m.get_match_root());
        if (!concat || transformation_callback(concat))
            return false;

        const auto& pattern_map = m.get_pattern_value_map();
        const auto& data_const = pattern_map.at(data_const_m).get_node_shared_ptr();
        auto broadcast = ov::as_type_ptr<ov::op::v3::Broadcast>(pattern_map.at(broadcast_m).get_node_shared_ptr());
        auto convert =  ov::as_type_ptr<ov::op::v0::Convert>(pattern_map.at(convert_m).get_node_shared_ptr());
        auto data_const_convert = std::make_shared<ov::op::v0::Convert>(data_const, element::f32);
        auto gemm =  ov::as_type_ptr<op::Gemm>(pattern_map.at(gemm_m).get_node_shared_ptr());
        auto transpose =  ov::as_type_ptr<ov::op::v1::Reshape>(pattern_map.at(transpose_m).get_node_shared_ptr());

        data_const_convert->set_friendly_name(data_const->get_friendly_name() + "/Convert");
        broadcast->input(0).replace_source_output(data_const_convert);
        broadcast->set_output_type(0, element::f32, broadcast->get_output_partial_shape(0));
        convert->set_destination_type(element::f32);
        gemm->set_output_type(0, element::f32, gemm->get_output_partial_shape(0));
        transpose->set_output_type(0, element::f32, transpose->get_output_partial_shape(0));
        concat->set_output_type(0, element::f32, concat->get_output_partial_shape(0));

        for (auto user : concat->get_users()) {
            if (auto cos = ov::as_type_ptr<ov::op::v0::Cos>(user)) {
                auto target_inputs = cos->get_output_target_inputs(0);
                auto cos_convert = std::make_shared<ov::op::v0::Convert>(cos, element::f16);
                cos_convert->set_friendly_name(cos->get_friendly_name() + "/Convert");
                ov::copy_runtime_info(cos, cos_convert);
                for (auto& in : target_inputs) {
                    in.replace_source_output(cos_convert);
                }
            }
            if (auto sin = ov::as_type_ptr<ov::op::v0::Sin>(user)) {
                auto target_inputs = sin->get_output_target_inputs(0);
                auto sin_convert = std::make_shared<ov::op::v0::Convert>(sin, element::f16);
                sin_convert->set_friendly_name(sin->get_friendly_name() + "/Convert");
                ov::copy_runtime_info(sin, sin_convert);
                for (auto& in : target_inputs) {
                    in.replace_source_output(sin_convert);
                }
            }
        }

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(concat_m, "ConvertPrecisionLRoPE");
    this->register_matcher(m, callback);
}

}  // namespace ov::intel_gpu
