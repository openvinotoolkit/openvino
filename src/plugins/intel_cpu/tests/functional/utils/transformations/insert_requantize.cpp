// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "insert_requantize.hpp"

#include "openvino/core/type.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/shuffle_channels.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/pp.hpp"
#include "utils/quantization_utils.hpp"

namespace CPUTestUtils {

using namespace ov;

InsertRequantize::InsertRequantize(size_t input_id, const QuantizationData& qinfo) {
    auto result_m = ov::pass::pattern::wrap_type<ov::op::v0::Result>(ov::pass::pattern::any_input());

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto result = ov::as_type_ptr<ov::op::v0::Result>(pattern_map.at(result_m).get_node_shared_ptr());
        auto input = result->get_input_source_output(0);

        if (!result || transformation_callback(result)) {
            return false;
        }

        // insert FakeQuantize and ShuffleChannels before Result node
        auto fq = std::make_shared<ov::op::v0::FakeQuantize>(
            input,
            ov::op::v0::Constant::create(ov::element::f32, Shape{}, {qinfo.il}),
            ov::op::v0::Constant::create(ov::element::f32, Shape{}, {qinfo.ih}),
            ov::op::v0::Constant::create(ov::element::f32, Shape{}, {qinfo.ol}),
            ov::op::v0::Constant::create(ov::element::f32, Shape{}, {qinfo.oh}),
            qinfo.levels);

        auto shuffle_channels = std::make_shared<ov::op::v0::ShuffleChannels>(fq);
        fq->set_friendly_name(shuffle_channels->get_friendly_name() + "/FQ");

        result->input(input_id).replace_source_output(shuffle_channels->output(0));

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(result_m, "InsertRequantize");
    this->register_matcher(m, callback);
}

}  // namespace CPUTestUtils
