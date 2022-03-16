// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <pad_shape_inference.hpp>

#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;

TEST(StaticShapeInferenceTest, Padv1) {
    const auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());

    const auto pads_begin = ov::op::v0::Constant::create(element::i64, ov::Shape{4}, {3, 2, 1, 0});
    const auto pads_end = ov::op::v0::Constant::create(element::i64, ov::Shape{4}, {0, 1, 2, 3});
    const auto pad_val = ov::op::v0::Constant::create(element::f32, ov::Shape{}, {2112});

    const auto pad = std::make_shared<ov::op::v1::Pad>(data, pads_begin, pads_end, pad_val, op::PadMode::CONSTANT);

    check_static_shape(pad.get(),
                       {StaticShape{3, 6, 5, 5},
                        StaticShape{4},
                        StaticShape{4},
                        StaticShape()},
                       {StaticShape({6, 9, 8, 8})});
}
