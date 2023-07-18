// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <gtest/gtest.h>

#include <transformations/common_optimizations/shared_ops_optimization.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/slice.hpp"

using namespace ov;
using namespace ov::op;

TEST_F(TransformationTestsF, SharedSlice) {
    auto make_slice = [](const Output<Node>& out,
                         const int64_t& start,
                         const int64_t& stop,
                         const int64_t& step,
                         const int64_t& axis) {
        return std::make_shared<v8::Slice>(out,
                                           v0::Constant::create(element::i64, Shape{1}, {start}),
                                           v0::Constant::create(element::i64, Shape{1}, {stop}),
                                           v0::Constant::create(element::i64, Shape{1}, {step}),
                                           v0::Constant::create(element::i64, Shape{1}, {axis}));
    };

    {
        auto data = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});

        auto slice_0 = make_slice(data, 1, 2, 3, 3);
        auto slice_1 = make_slice(data, 1, 2, 3, 3);
        auto slice_2 = make_slice(data, 1, 3, 3, 3);
        auto slice_3 = make_slice(data, 1, 2, 3, 3);
        auto slice_4 = make_slice(data, 1, 2, 3, 3);

        auto concat = std::make_shared<v0::Concat>(OutputVector{slice_0, slice_1, slice_2, slice_3, slice_4}, 0);

        auto result = std::make_shared<v0::Result>(concat);
        model = std::make_shared<ov::Model>(ResultVector{result}, ParameterVector{data});
        manager.register_pass<ov::pass::SharedOpOptimization>();
    }
    {
        auto data = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});

        auto slice_0 = make_slice(data, 1, 2, 3, 3);
        auto slice_2 = make_slice(data, 1, 3, 3, 3);

        auto concat = std::make_shared<v0::Concat>(OutputVector{slice_0, slice_0, slice_2, slice_0, slice_0}, 0);

        auto result = std::make_shared<v0::Result>(concat);
        model_ref = std::make_shared<ov::Model>(ResultVector{result}, ParameterVector{data});
    }
}