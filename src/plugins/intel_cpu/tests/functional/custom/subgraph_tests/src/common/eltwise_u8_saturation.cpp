// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <memory>

#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/split.hpp"
#include "openvino/openvino.hpp"

TEST(smoke_CPUEltwiseU8Saturation, ConcatSplitChainMultiply) {
    auto v0 = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, ov::Shape{1, 1});
    auto v3 = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, ov::Shape{59, 3});
    auto one = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{1}, {1});
    auto v2 = std::make_shared<ov::op::v1::Multiply>(one, v0);
    auto v8 = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{v2, v2}, 0);
    auto v11 = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{v8, v8}, 0);
    auto axis0 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
    auto split12 = std::make_shared<ov::op::v1::Split>(v11, axis0, 2);
    auto split13 = std::make_shared<ov::op::v1::Split>(split12->output(0), axis0, 2);
    auto v13 = split13->output(0);
    auto v19 = std::make_shared<ov::op::v1::Multiply>(v13, v13);
    auto v20 = std::make_shared<ov::op::v1::Multiply>(v3, v13);
    auto v21 = std::make_shared<ov::op::v1::Add>(v20, v19);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{v21}, ov::ParameterVector{v0, v3});

    ov::Tensor t0(ov::element::u8, ov::Shape{1, 1});
    t0.data<uint8_t>()[0] = 255;
    ov::Tensor t3(ov::element::u8, ov::Shape{59, 3});
    std::fill_n(t3.data<uint8_t>(), t3.get_size(), uint8_t{0});

    ov::Core core;
    auto compiled = core.compile_model(model, "CPU");
    auto request = compiled.create_infer_request();
    request.set_input_tensor(0, t0);
    request.set_input_tensor(1, t3);
    request.infer();

    const auto out = request.get_output_tensor(0);
    ASSERT_EQ(out.get_element_type(), ov::element::u8);
    ASSERT_EQ(out.get_shape(), (ov::Shape{59, 3}));
    const auto* result = out.data<const uint8_t>();
    for (size_t i = 0; i < out.get_size(); ++i) {
        ASSERT_EQ(result[i], 255) << "element " << i;
    }
}
