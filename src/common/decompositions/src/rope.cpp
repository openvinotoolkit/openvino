// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/decompositions/rope.hpp"

#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/variadic_split.hpp"

namespace ov {
namespace decompositions {

ov::Output<ov::Node> rope(ov::pass::NodeRegistry& reg,
                          const ov::Output<ov::Node>& x,
                          const ov::Output<ov::Node>& cos,
                          const ov::Output<ov::Node>& sin,
                          int64_t half_head_size) {
    // Split x along the last axis into two equal halves of size `half_head_size`.
    auto split_axis = reg.make<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, std::vector<int64_t>{-1});
    auto split_lengths = reg.make<ov::op::v0::Constant>(ov::element::i64,
                                                        ov::Shape{2},
                                                        std::vector<int64_t>{half_head_size, half_head_size});
    auto halves = reg.make<ov::op::v1::VariadicSplit>(x, split_axis, split_lengths)->outputs();
    const auto& first_half = halves[0];
    const auto& second_half = halves[1];

    // Core RoPE formula:
    //   first_  = first_half  * cos - second_half * sin
    //   second_ = second_half * cos + first_half  * sin
    // The "minus" is expressed as Multiply(-1) + Add (not Subtract) so the
    // RoPEFusion pattern matcher recognises it.
    auto first_half_mul_cos = reg.make<ov::op::v1::Multiply>(first_half, cos);
    auto second_half_mul_sin = reg.make<ov::op::v1::Multiply>(second_half, sin);

    ov::Output<ov::Node> neg_one;
    {
        const auto& et = x.get_element_type();
        if (et.is_static() && et != ov::element::dynamic) {
            neg_one = reg.make<ov::op::v0::Constant>(et, ov::Shape{}, std::vector<float>{-1.0f});
        } else {
            auto c = reg.make<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<float>{-1.0f});
            neg_one = reg.make<ov::op::v1::ConvertLike>(c, x);
        }
    }
    auto neg_second_half_mul_sin = reg.make<ov::op::v1::Multiply>(second_half_mul_sin, neg_one);
    auto first_part = reg.make<ov::op::v1::Add>(first_half_mul_cos, neg_second_half_mul_sin);

    auto second_half_mul_cos = reg.make<ov::op::v1::Multiply>(second_half, cos);
    auto first_half_mul_sin = reg.make<ov::op::v1::Multiply>(first_half, sin);
    auto second_part = reg.make<ov::op::v1::Add>(second_half_mul_cos, first_half_mul_sin);

    return reg.make<ov::op::v0::Concat>(ov::NodeVector{first_part, second_part}, /*axis=*/-1);
}

}  // namespace decompositions
}  // namespace ov
