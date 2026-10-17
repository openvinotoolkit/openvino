// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cstdint>
#include <tuple>
#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "gtest/gtest.h"
#include "openvino/op/clamp.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/openvino.hpp"
#include "openvino/runtime/properties.hpp"

namespace {
using Bounds = std::tuple<double, double>;
class ClampI64BoundsGPU : public testing::TestWithParam<std::tuple<Bounds, bool>> {};

TEST_P(ClampI64BoundsGPU, PreservesNarrowedValues) {
    const auto bounds = std::get<0>(GetParam());
    const auto lower = std::get<0>(bounds);
    const auto upper = std::get<1>(bounds);
    const bool convert_input = std::get<1>(GetParam());
    // Values fit i32 and are exactly representable in f32. Bounds below i32
    // must leave these values intact, including INT32_MIN itself.
    const std::vector<int64_t> pattern{-2147483648LL, -2147483392LL, -1024, -1, 0, 146, 360, 1000};
    const ov::Shape shape{1, 64};
    const auto type = convert_input ? ov::element::f32 : ov::element::i64;
    auto input = std::make_shared<ov::op::v0::Parameter>(type, shape);
    ov::Output<ov::Node> data = input;
    if (convert_input)
        data = std::make_shared<ov::op::v0::Convert>(data, ov::element::i64);
    auto clamp = std::make_shared<ov::op::v0::Clamp>(data, lower, upper);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{clamp}, ov::ParameterVector{input});
    ov::Core core;
    auto compiled = core.compile_model(model, ov::test::utils::DEVICE_GPU, {ov::hint::inference_precision(ov::element::f32)});
    auto request = compiled.create_infer_request();
    ov::Tensor tensor(type, shape);
    for (size_t repeat = 0; repeat < 2; ++repeat) {
        for (size_t i = 0; i < tensor.get_size(); ++i) {
            const auto value = pattern[(i + repeat) % pattern.size()];
            if (convert_input)
                tensor.data<float>()[i] = static_cast<float>(value);
            else
                tensor.data<int64_t>()[i] = value;
        }
        request.set_input_tensor(tensor);
        request.infer();
        const auto output = request.get_output_tensor();
        ASSERT_EQ(output.get_element_type(), ov::element::i64);
        ASSERT_EQ(output.get_shape(), shape);
        for (size_t i = 0; i < output.get_size(); ++i) {
            const auto value = pattern[(i + repeat) % pattern.size()];
            const auto expected = static_cast<int64_t>(std::max(lower, std::min(upper, static_cast<double>(value))));
            ASSERT_EQ(output.data<const int64_t>()[i], expected) << "index=" << i << " repeat=" << repeat;
        }
    }
}

INSTANTIATE_TEST_SUITE_P(smoke,
                         ClampI64BoundsGPU,
                         testing::Combine(testing::Values(Bounds{-9223372036854775808.0, 360.0},
                                                          Bounds{-1099511627776.0, 360.0},
                                                          Bounds{-2147483904.0, 360.0},
                                                          Bounds{-2147483648.0, 360.0},
                                                          Bounds{-1000.0, 360.0},
                                                          Bounds{0.0, 360.0},
                                                          Bounds{-2147483648.0, 9223372036854775808.0}),
                                          testing::Bool()));
}  // namespace
