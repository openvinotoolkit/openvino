// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/runtime/core.hpp"
#include "transformations/common_optimizations/strided_slice_reshape_concat_fusion.hpp"

namespace {

std::shared_ptr<ov::Model> build_original_model() {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 16});
    ov::OutputVector concat_inputs;

    for (const int64_t start : {0, 2, 4}) {
        auto begin = ov::op::v0::Constant::create<int64_t>(ov::element::i64, ov::Shape{2}, {0, start});
        auto end = ov::op::v0::Constant::create<int64_t>(ov::element::i64, ov::Shape{2}, {1, start + 4});
        auto strides = ov::op::v0::Constant::create<int64_t>(ov::element::i64, ov::Shape{2}, {1, 1});
        auto strided_slice =
            std::make_shared<ov::op::v1::StridedSlice>(input,
                                                       begin,
                                                       end,
                                                       strides,
                                                       std::vector<int64_t>{0, 0},
                                                       std::vector<int64_t>{0, 0});
        auto shape = ov::op::v0::Constant::create<int64_t>(ov::element::i64, ov::Shape{3}, {1, 1, 4});
        auto reshape = std::make_shared<ov::op::v1::Reshape>(strided_slice, shape, false);
        concat_inputs.push_back(reshape);
    }

    auto concat = std::make_shared<ov::op::v0::Concat>(concat_inputs, 1);
    return std::make_shared<ov::Model>(ov::OutputVector{concat}, ov::ParameterVector{input});
}

}  // namespace

TEST(StridedSliceReshapeConcatFusionCPU, CompareWithAndWithoutPass) {
    auto model_without_pass = build_original_model();
    auto model_with_pass = model_without_pass->clone();

    ov::pass::Manager manager;
    manager.register_pass<ov::pass::StridedSliceReshapeConcatFusion>();
    manager.run_passes(model_with_pass);

    const auto input_tensor =
        ov::test::utils::create_and_fill_tensor(ov::element::f32,
                                                ov::Shape{1, 16},
                                                ov::test::utils::InputGenerateData{-3.0, 7, 1, 1});

    try {
        ov::Core core;
        auto compiled_without_pass = core.compile_model(model_without_pass, "CPU");
        auto compiled_with_pass = core.compile_model(model_with_pass, "CPU");

        auto req_without_pass = compiled_without_pass.create_infer_request();
        auto req_with_pass = compiled_with_pass.create_infer_request();

        req_without_pass.set_tensor(0, input_tensor);
        req_with_pass.set_tensor(0, input_tensor);

        req_without_pass.infer();
        req_with_pass.infer();

        ov::test::utils::compare(req_without_pass.get_output_tensor(0), req_with_pass.get_output_tensor(0), 0.0, 0.0);
    } catch (const std::exception& ex) {
        GTEST_SKIP() << "CPU plugin is not available in this test runtime: " << ex.what();
    }
}
