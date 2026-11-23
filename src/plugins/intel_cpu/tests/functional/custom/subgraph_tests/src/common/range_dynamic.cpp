// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "openvino/core/model.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/range.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov::test {
namespace {

struct RangeCase {
    double start;
    double stop;
    double step;
};

// All inputs are Parameters, so the CPU node computes the output length at runtime.
// f32 inference precision, so the fractional values are not rounded to f16 (the ARM default).
ov::InferRequest make_range_request(const element::Type& type) {
    ParameterVector params;
    for (const char* name : {"start", "stop", "step"}) {
        params.push_back(std::make_shared<op::v0::Parameter>(type, Shape{}));
        params.back()->set_friendly_name(name);
    }
    auto range = std::make_shared<op::v4::Range>(params[0], params[1], params[2], type);
    auto model = std::make_shared<Model>(OutputVector{range}, params);
    auto compiled = ov::Core().compile_model(model, "CPU", ov::hint::inference_precision(element::f32));
    return compiled.create_infer_request();
}

template <typename T>
std::vector<T> run_range(ov::InferRequest& request, const element::Type& type, T start, T stop, T step) {
    const T values[] = {start, stop, step};
    for (size_t i = 0; i < 3; ++i) {
        Tensor input(type, Shape{});
        *input.data<T>() = values[i];
        request.set_input_tensor(i, input);
    }
    request.infer();
    const auto output = request.get_output_tensor();
    if (output.get_size() == 0) {
        return {};
    }
    const auto* data = output.data<T>();
    return {data, data + output.get_size()};
}

// Range-4: max(ceil((stop - start) / step), 0) elements, start + i * step.
template <typename T>
std::vector<T> reference_range(T start, T stop, T step) {
    const auto span = static_cast<double>(stop) - static_cast<double>(start);
    const double count = std::ceil(span / static_cast<double>(step));
    std::vector<T> result(count > 0 ? static_cast<size_t>(count) : 0);
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] = static_cast<T>(start + static_cast<T>(i) * step);
    }
    return result;
}

template <typename T>
void check_cases(const element::Type& type, const std::vector<RangeCase>& cases) {
    auto request = make_range_request(type);
    for (const auto& c : cases) {
        const auto start = static_cast<T>(c.start);
        const auto stop = static_cast<T>(c.stop);
        const auto step = static_cast<T>(c.step);
        SCOPED_TRACE(testing::Message() << type << " start=" << c.start << " stop=" << c.stop << " step=" << c.step);
        const auto actual = run_range<T>(request, type, start, stop, step);
        const auto expected = reference_range<T>(start, stop, step);
        ASSERT_EQ(actual.size(), expected.size());
        for (size_t i = 0; i < expected.size(); ++i) {
            EXPECT_NEAR(static_cast<double>(actual[i]), static_cast<double>(expected[i]), 1e-5) << "at index " << i;
        }
    }
}

// The same infer request runs every case in order, so the output also goes from empty back to non-empty.
const std::vector<RangeCase> common_cases = {
    {0, 5, 1},
    {5, 0, -1},
    {2, -3, -2},
    // step points away from stop, or start == stop: the output is empty
    {5, 0, 1},
    {0, 5, -1},
    {-1, -4, 2},
    {3, 3, 1},
    {3, 3, -1},
    {0, 5, 1},
};

TEST(smoke_RangeDynamicCPUTest, StepSignF32) {
    auto cases = common_cases;
    cases.push_back({1, 2, 0.3});
    cases.push_back({2, 1, 0.3});
    check_cases<float>(element::f32, cases);
}

TEST(smoke_RangeDynamicCPUTest, StepSignI32) {
    check_cases<int32_t>(element::i32, common_cases);
}

TEST(smoke_RangeDynamicCPUTest, ZeroStepThrows) {
    auto f32_request = make_range_request(element::f32);
    EXPECT_THROW(run_range<float>(f32_request, element::f32, 0.0F, 5.0F, 0.0F), ov::Exception);
    auto i32_request = make_range_request(element::i32);
    EXPECT_THROW(run_range<int32_t>(i32_request, element::i32, 0, 5, 0), ov::Exception);
}

}  // namespace
}  // namespace ov::test
