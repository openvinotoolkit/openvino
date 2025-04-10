// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <tuple>

#include "base_reference_test.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/nv12_to_bgr.hpp"
#include "openvino/op/nv12_to_rgb.hpp"

using namespace ov;
using namespace reference_tests;

class ReferenceConvertColorNV12LayerTest : public testing::Test, public CommonReferenceTest {
public:
    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        abs_threshold = 2.f;  // allow R, G, B absolute deviation to 2 (of max 255)
        threshold = 1.f;      // Ignore relative comparison (100%)
    }

public:
    template <typename T>
    static std::shared_ptr<Model> CreateFunction(const reference_tests::Tensor& input) {
        const auto in = std::make_shared<op::v0::Parameter>(input.type, input.shape);
        std::shared_ptr<Node> conv;
        conv = std::make_shared<T>(in);
        auto res = std::make_shared<op::v0::Result>(conv);
        return std::make_shared<Model>(ResultVector{res}, ParameterVector{in});
    }

    template <typename T>
    static std::shared_ptr<Model> CreateFunction2(const reference_tests::Tensor& input1,
                                                  const reference_tests::Tensor& input2) {
        const auto in1 = std::make_shared<op::v0::Parameter>(input1.type, input1.shape);
        const auto in2 = std::make_shared<op::v0::Parameter>(input2.type, input2.shape);
        std::shared_ptr<Node> conv;
        conv = std::make_shared<T>(in1, in2);
        auto res = std::make_shared<op::v0::Result>(conv);
        return std::make_shared<Model>(ResultVector{res}, ParameterVector{in1, in2});
    }
};

TEST_F(ReferenceConvertColorNV12LayerTest, CompareWithHardcodedRefs_r_u8_single_rgb) {
    auto input = std::vector<uint8_t>{0x51, 0x51, 0x51, 0x51, 0x5a, 0xf0};
    auto input_shape = Shape{1, 3, 2, 1};
    auto exp_out = std::vector<uint8_t>{0xff, 0, 0, 0xff, 0, 0, 0xff, 0, 0, 0xff, 0, 0};
    auto out_shape = Shape{1, 2, 2, 3};
    reference_tests::Tensor inp_tensor(input_shape, element::u8, input);
    inputData = {inp_tensor.data};
    function = CreateFunction<op::v8::NV12toRGB>(inp_tensor);
    reference_tests::Tensor exp_tensor_u8(out_shape, element::u8, exp_out);
    refOutData = {exp_tensor_u8.data};
    Exec();
}

TEST_F(ReferenceConvertColorNV12LayerTest, CompareWithHardcodedRefs_color_u8_single_bgr) {
    auto input = std::vector<uint8_t>{0x51, 0xeb, 0x51, 0xeb, 0x6d, 0xb8};
    auto input_shape = Shape{1, 3, 2, 1};
    auto exp_out = std::vector<uint8_t>{37, 37, 164, 215, 216, 255, 37, 37, 164, 215, 216, 255};
    auto out_shape = Shape{1, 2, 2, 3};

    reference_tests::Tensor inp_tensor(input_shape, element::u8, input);
    inputData = {inp_tensor.data};

    reference_tests::Tensor exp_tensor_u8(out_shape, element::u8, exp_out);
    refOutData = {exp_tensor_u8.data};

    function = CreateFunction<op::v8::NV12toBGR>(inp_tensor);

    Exec();
}

TEST_F(ReferenceConvertColorNV12LayerTest, CompareWithHardcodedRefs_g_fp32_single_rgb) {
    auto input = std::vector<float>{145.f, 145.f, 145.f, 145.f, 54.f, 34.f};
    auto input_shape = Shape{1, 3, 2, 1};
    auto exp_out = std::vector<float>{0, 255.f, 0, 0, 255.f, 0, 0, 255.f, 0, 0, 255.f, 0};
    auto out_shape = Shape{1, 2, 2, 3};

    reference_tests::Tensor inp_tensor(input_shape, element::f32, input);
    inputData = {inp_tensor.data};

    reference_tests::Tensor exp_tensor(out_shape, element::f32, exp_out);
    refOutData = {exp_tensor.data};

    function = CreateFunction<op::v8::NV12toRGB>(inp_tensor);

    Exec();
}

TEST_F(ReferenceConvertColorNV12LayerTest, CompareWithHardcodedRefs_batch_fp32_two_bgr) {
    auto input_y = std::vector<float>{81.f, 81.f, 81.f, 81.f, 145.f, 145.f, 145.f, 145.f, 41.f, 41.f, 41.f, 41.f};
    auto input_shape_y = Shape{3, 2, 2, 1};

    auto input_uv = std::vector<float>{90., 240., 54., 34., 240., 110.};
    auto input_shape_uv = Shape{3, 1, 1, 2};

    auto exp_out =
        std::vector<float>{0, 0,    255., 0, 0,    255., 0,    0, 255., 0,    0, 255., 0,    255., 0, 0,    255., 0,
                           0, 255., 0,    0, 255., 0,    255., 0, 0,    255., 0, 0,    255., 0,    0, 255., 0,    0};
    auto out_shape = Shape{3, 2, 2, 3};

    reference_tests::Tensor inp_tensor_y(input_shape_y, element::f32, input_y);
    reference_tests::Tensor inp_tensor_uv(input_shape_uv, element::f32, input_uv);
    inputData = {inp_tensor_y.data, inp_tensor_uv.data};

    reference_tests::Tensor exp_tensor(out_shape, element::f32, exp_out);
    refOutData = {exp_tensor.data};

    function = CreateFunction2<op::v8::NV12toBGR>(inp_tensor_y, inp_tensor_uv);

    Exec();
}

TEST_F(ReferenceConvertColorNV12LayerTest, CompareWithHardcodedRefs_color2x2_f32_two_rgb) {
    auto input_y = std::vector<float>{81, 235, 81, 235};
    auto input_shape_y = Shape{1, 2, 2, 1};

    auto input_uv = std::vector<float>{109, 184};
    auto input_shape_uv = Shape{1, 1, 1, 2};

    auto exp_out = std::vector<float>{164, 37, 37, 255, 216, 215, 164, 37, 37, 255, 216, 215};
    auto out_shape = Shape{1, 2, 2, 3};

    reference_tests::Tensor inp_tensor_y(input_shape_y, element::f32, input_y);
    reference_tests::Tensor inp_tensor_uv(input_shape_uv, element::f32, input_uv);
    inputData = {inp_tensor_y.data, inp_tensor_uv.data};

    reference_tests::Tensor exp_tensor(out_shape, element::f32, exp_out);
    refOutData = {exp_tensor.data};

    function = CreateFunction2<op::v8::NV12toRGB>(inp_tensor_y, inp_tensor_uv);

    Exec();
}
