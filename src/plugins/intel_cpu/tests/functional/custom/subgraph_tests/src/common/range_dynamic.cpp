// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>

#include "openvino/core/model.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/range.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov::test {
namespace {

// All inputs are Parameters, so the CPU node computes the output length at runtime.
// The values themselves are covered by the RangeLayerTest instances; this only checks that step 0 is rejected.
ov::InferRequest make_range_request(const element::Type& type) {
    ParameterVector params;
    for (const char* name : {"start", "stop", "step"}) {
        params.push_back(std::make_shared<op::v0::Parameter>(type, Shape{}));
        params.back()->set_friendly_name(name);
    }
    auto range = std::make_shared<op::v4::Range>(params[0], params[1], params[2], type);
    auto model = std::make_shared<Model>(OutputVector{range}, params);
    return ov::Core().compile_model(model, "CPU").create_infer_request();
}

template <typename T>
void run_range(ov::InferRequest& request, const element::Type& type, T start, T stop, T step) {
    const T values[] = {start, stop, step};
    for (size_t i = 0; i < 3; ++i) {
        Tensor input(type, Shape{});
        *input.data<T>() = values[i];
        request.set_input_tensor(i, input);
    }
    request.infer();
}

TEST(smoke_RangeDynamicCPUTest, ZeroStepThrows) {
    auto f32_request = make_range_request(element::f32);
    EXPECT_THROW(run_range<float>(f32_request, element::f32, 0.0F, 5.0F, 0.0F), ov::Exception);
    auto i32_request = make_range_request(element::i32);
    EXPECT_THROW(run_range<int32_t>(i32_request, element::i32, 0, 5, 0), ov::Exception);
}

}  // namespace
}  // namespace ov::test
