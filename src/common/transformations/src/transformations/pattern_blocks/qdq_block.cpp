// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/pattern_blocks/qdq_block.hpp"

#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace pattern = ov::pass::pattern;

ov::pass::pattern::op::QDQBlock::QDQBlock(ov::pass::pattern::op::Predicate data_pred,
                                          ov::pass::pattern::op::Predicate q_convert_pred,
                                          ov::pass::pattern::op::Predicate dq_convert_pred)
    : Block({}, {}, "QDQBlock") {
    auto data_pattern = pattern::any_input(data_pred);
    auto input_low_pattern = pattern::any_input();
    auto input_high_pattern = pattern::any_input();
    auto output_low_pattern = pattern::wrap_type<v0::Constant>();
    auto output_high_pattern = pattern::wrap_type<v0::Constant>();
    auto fq_pattern = pattern::wrap_type<v0::FakeQuantize>(
        {data_pattern, input_low_pattern, input_high_pattern, output_low_pattern, output_high_pattern});

    auto q_convert_pattern = pattern::wrap_type<v0::Convert>({fq_pattern}, q_convert_pred);

    auto dq_convert_pattern = pattern::wrap_type<v0::Convert>({q_convert_pattern}, dq_convert_pred);

    auto zero_point_pattern = pattern::any_input();
    auto sub_pattern =
        pattern::optional<v1::Subtract>({dq_convert_pattern, zero_point_pattern}, pattern::consumers_count(1));

    auto scale_pattern = pattern::any_input();
    auto mul_pattern = pattern::wrap_type<v1::Multiply>({sub_pattern, scale_pattern});

    m_inputs = {data_pattern};
    m_outputs = {mul_pattern};

    REGISTER_ANCHORS(this,
                     data_pattern,
                     input_low_pattern,
                     input_high_pattern,
                     output_low_pattern,
                     output_high_pattern,
                     fq_pattern,
                     q_convert_pattern,
                     dq_convert_pattern,
                     sub_pattern,
                     zero_point_pattern,
                     scale_pattern,
                     mul_pattern);
}
