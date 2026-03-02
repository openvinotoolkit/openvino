// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <tuple>

#include "base_reference_test.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/rgb_to_nv12.hpp"
#include "openvino/op/bgr_to_nv12.hpp"

using namespace ov;
using namespace reference_tests;

class ReferenceConvertColorToNV12LayerTest : public testing::Test, public CommonReferenceTest {
public:
    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        abs_threshold = 2.f;  // allow Y, U, V absolute deviation to 2 (of max 255)
        threshold = 1.f;      // Ignore relative comparison (100%)
    }

public:
    template <typename T>
    static std::shared_ptr<Model> CreateFunction(const reference_tests::Tensor& input) {
        const auto in = std::make_shared<op::v0::Parameter>(input.type, input.shape);
        auto conv = std::make_shared<T>(in);
        auto res = std::make_shared<op::v0::Result>(conv);
        return std::make_shared<Model>(ResultVector{res}, ParameterVector{in});
    }

    template <typename T>
    static std::shared_ptr<Model> CreateFunction2Plane(const reference_tests::Tensor& input) {
        const auto in = std::make_shared<op::v0::Parameter>(input.type, input.shape);
        auto conv = std::make_shared<T>(in, false);
        auto res_y = std::make_shared<op::v0::Result>(conv->output(0));
        auto res_uv = std::make_shared<op::v0::Result>(conv->output(1));
        return std::make_shared<Model>(ResultVector{res_y, res_uv}, ParameterVector{in});
    }
};

TEST_F(ReferenceConvertColorToNV12LayerTest, CompareWithHardcodedRefs_red_u8_single_rgb) {
    auto input = std::vector<uint8_t>{255, 0, 0,   255, 0, 0,
                                      255, 0, 0,   255, 0, 0};
    auto input_shape = Shape{1, 2, 2, 3};
    auto exp_out = std::vector<uint8_t>{82, 82, 82, 82, 90, 240};
    auto out_shape = Shape{1, 3, 2, 1};

    reference_tests::Tensor inp_tensor(input_shape, element::u8, input);
    inputData = {inp_tensor.data};
    function = CreateFunction<op::v16::RGBtoNV12>(inp_tensor);
    reference_tests::Tensor exp_tensor(out_shape, element::u8, exp_out);
    refOutData = {exp_tensor.data};
    Exec();
}

TEST_F(ReferenceConvertColorToNV12LayerTest, CompareWithHardcodedRefs_green_f32_single_rgb) {
    auto input = std::vector<float>{0, 255.f, 0,   0, 255.f, 0,
                                    0, 255.f, 0,   0, 255.f, 0};
    auto input_shape = Shape{1, 2, 2, 3};
    auto exp_out = std::vector<float>{145.f, 145.f, 145.f, 145.f, 54.f, 34.f};
    auto out_shape = Shape{1, 3, 2, 1};

    reference_tests::Tensor inp_tensor(input_shape, element::f32, input);
    inputData = {inp_tensor.data};

    reference_tests::Tensor exp_tensor(out_shape, element::f32, exp_out);
    refOutData = {exp_tensor.data};

    function = CreateFunction<op::v16::RGBtoNV12>(inp_tensor);

    Exec();
}

TEST_F(ReferenceConvertColorToNV12LayerTest, CompareWithHardcodedRefs_blue_u8_single_bgr) {
    auto input = std::vector<uint8_t>{255, 0, 0,   255, 0, 0,
                                      255, 0, 0,   255, 0, 0};
    auto input_shape = Shape{1, 2, 2, 3};
    auto exp_out = std::vector<uint8_t>{41, 41, 41, 41, 240, 110};
    auto out_shape = Shape{1, 3, 2, 1};

    reference_tests::Tensor inp_tensor(input_shape, element::u8, input);
    inputData = {inp_tensor.data};
    function = CreateFunction<op::v16::BGRtoNV12>(inp_tensor);
    reference_tests::Tensor exp_tensor(out_shape, element::u8, exp_out);
    refOutData = {exp_tensor.data};
    Exec();
}

TEST_F(ReferenceConvertColorToNV12LayerTest, CompareWithHardcodedRefs_red_f32_two_rgb) {
    auto input = std::vector<float>{255.f, 0, 0,   255.f, 0, 0,
                                    255.f, 0, 0,   255.f, 0, 0};
    auto input_shape = Shape{1, 2, 2, 3};

    auto exp_y = std::vector<float>{82.f, 82.f, 82.f, 82.f};
    auto exp_y_shape = Shape{1, 2, 2, 1};

    auto exp_uv = std::vector<float>{90.f, 240.f};
    auto exp_uv_shape = Shape{1, 1, 1, 2};

    reference_tests::Tensor inp_tensor(input_shape, element::f32, input);
    inputData = {inp_tensor.data};

    reference_tests::Tensor exp_y_tensor(exp_y_shape, element::f32, exp_y);
    reference_tests::Tensor exp_uv_tensor(exp_uv_shape, element::f32, exp_uv);
    refOutData = {exp_y_tensor.data, exp_uv_tensor.data};

    function = CreateFunction2Plane<op::v16::RGBtoNV12>(inp_tensor);

    Exec();
}
