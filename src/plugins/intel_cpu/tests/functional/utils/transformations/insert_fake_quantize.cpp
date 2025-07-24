// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "insert_fake_quantize.hpp"

#include "openvino/core/graph_util.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/type.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/pp.hpp"
#include "utils/quantization_utils.hpp"

namespace CPUTestUtils {

using namespace ov;

InsertFakeQuantize::InsertFakeQuantize(size_t input_id, const QuantizationData& qinfo) {
    // more operation can be covered if necessery
    auto ops_m = ov::pass::pattern::wrap_type<ov::op::v1::Convolution, ov::op::v0::MatMul>(
        {ov::pass::pattern::any_input(), ov::pass::pattern::any_input()});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto op = pattern_map.at(ops_m).get_node_shared_ptr();
        auto input = op->get_input_source_output(input_id);

        if (!op || transformation_callback(op)) {
            return false;
        }

        auto fq = std::make_shared<ov::op::v0::FakeQuantize>(
            input,
            ov::op::v0::Constant::create(ov::element::f32, Shape{}, {qinfo.il}),
            ov::op::v0::Constant::create(ov::element::f32, Shape{}, {qinfo.ih}),
            ov::op::v0::Constant::create(ov::element::f32, Shape{}, {qinfo.ol}),
            ov::op::v0::Constant::create(ov::element::f32, Shape{}, {qinfo.oh}),
            qinfo.levels);

        fq->set_friendly_name(input.get_node_shared_ptr()->get_friendly_name() + "/FQ");

        OutputVector input_values = op->input_values();
        input_values[input_id] = fq->output(0);
        auto clone = op->clone_with_new_inputs(input_values);

        clone->set_friendly_name(op->get_friendly_name());
        ov::copy_runtime_info(op, clone);
        ov::replace_node(op, clone);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(ops_m, "InsertFakeQuantize");
    this->register_matcher(m, callback);
}

}  // namespace CPUTestUtils
