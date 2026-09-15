// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "base_reference_test.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/bgr_to_nv12.hpp"
#include "openvino/op/nv12_to_bgr.hpp"
#include "openvino/op/nv12_to_rgb.hpp"
#include "openvino/op/rgb_to_nv12.hpp"

namespace ov::tests {
class ReferenceConvertColorToNV12LayerTest : public testing::Test, public reference_tests::CommonReferenceTest {
public:
    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        abs_threshold = 2.f;  // allow Y, U, V absolute deviation to 2 (of max 255)
        threshold = 1.f;      // Ignore relative comparison (100%)
    }

public:
    template <typename T>
    static std::shared_ptr<ov::Model> CreateFunction(const reference_tests::Tensor& input) {
        const auto in = std::make_shared<ov::op::v0::Parameter>(input.type, input.shape);
        auto conv = std::make_shared<T>(in);
        auto res = std::make_shared<ov::op::v0::Result>(conv);
        return std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{in});
    }

    template <typename T>
    static std::shared_ptr<ov::Model> CreateFunction2Plane(const reference_tests::Tensor& input) {
        const auto in = std::make_shared<ov::op::v0::Parameter>(input.type, input.shape);
        auto conv = std::make_shared<T>(in, false);
        auto res_y = std::make_shared<ov::op::v0::Result>(conv->output(0));
        auto res_uv = std::make_shared<ov::op::v0::Result>(conv->output(1));
        return std::make_shared<ov::Model>(ov::ResultVector{res_y, res_uv}, ov::ParameterVector{in});
    }

    template <typename NV12_to_Color, typename Color_to_NV12>
    static std::shared_ptr<ov::Model> CreateRoundTripFunction(const reference_tests::Tensor& input) {
        const auto in = std::make_shared<ov::op::v0::Parameter>(input.type, input.shape);
        auto to_color = std::make_shared<NV12_to_Color>(in);
        auto to_nv12 = std::make_shared<Color_to_NV12>(to_color->output(0));
        auto res = std::make_shared<ov::op::v0::Result>(to_nv12);
        return std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{in});
    }
};

TEST_F(ReferenceConvertColorToNV12LayerTest, CompareWithHardcodedRefs_red_u8_single_rgb) {
    auto input = std::vector<uint8_t>{255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0};
    auto input_shape = ov::Shape{1, 2, 2, 3};
    auto exp_out = std::vector<uint8_t>{82, 82, 82, 82, 90, 240};
    auto out_shape = ov::Shape{1, 3, 2, 1};

    reference_tests::Tensor inp_tensor(input_shape, ov::element::u8, input);
    inputData = {inp_tensor.data};
    function = CreateFunction<ov::op::v17::RGBtoNV12>(inp_tensor);
    reference_tests::Tensor exp_tensor(out_shape, ov::element::u8, exp_out);
    refOutData = {exp_tensor.data};
    Exec();
}

TEST_F(ReferenceConvertColorToNV12LayerTest, CompareWithHardcodedRefs_green_f32_single_rgb) {
    auto input = std::vector<float>{0, 255.f, 0, 0, 255.f, 0, 0, 255.f, 0, 0, 255.f, 0};
    auto input_shape = ov::Shape{1, 2, 2, 3};
    auto exp_out = std::vector<float>{145.f, 145.f, 145.f, 145.f, 54.f, 34.f};
    auto out_shape = ov::Shape{1, 3, 2, 1};

    reference_tests::Tensor inp_tensor(input_shape, ov::element::f32, input);
    inputData = {inp_tensor.data};

    reference_tests::Tensor exp_tensor(out_shape, ov::element::f32, exp_out);
    refOutData = {exp_tensor.data};

    function = CreateFunction<ov::op::v17::RGBtoNV12>(inp_tensor);

    Exec();
}

TEST_F(ReferenceConvertColorToNV12LayerTest, CompareWithHardcodedRefs_blue_u8_single_bgr) {
    auto input = std::vector<uint8_t>{255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0};
    auto input_shape = ov::Shape{1, 2, 2, 3};
    auto exp_out = std::vector<uint8_t>{41, 41, 41, 41, 240, 110};
    auto out_shape = ov::Shape{1, 3, 2, 1};

    reference_tests::Tensor inp_tensor(input_shape, ov::element::u8, input);
    inputData = {inp_tensor.data};
    function = CreateFunction<ov::op::v17::BGRtoNV12>(inp_tensor);
    reference_tests::Tensor exp_tensor(out_shape, ov::element::u8, exp_out);
    refOutData = {exp_tensor.data};
    Exec();
}

TEST_F(ReferenceConvertColorToNV12LayerTest, CompareWithHardcodedRefs_red_f32_two_rgb) {
    auto input = std::vector<float>{255.f, 0, 0, 255.f, 0, 0, 255.f, 0, 0, 255.f, 0, 0};
    auto input_shape = ov::Shape{1, 2, 2, 3};

    auto exp_y = std::vector<float>{82.f, 82.f, 82.f, 82.f};
    auto exp_y_shape = ov::Shape{1, 2, 2, 1};

    auto exp_uv = std::vector<float>{90.f, 240.f};
    auto exp_uv_shape = ov::Shape{1, 1, 1, 2};

    reference_tests::Tensor inp_tensor(input_shape, ov::element::f32, input);
    inputData = {inp_tensor.data};

    reference_tests::Tensor exp_y_tensor(exp_y_shape, ov::element::f32, exp_y);
    reference_tests::Tensor exp_uv_tensor(exp_uv_shape, ov::element::f32, exp_uv);
    refOutData = {exp_y_tensor.data, exp_uv_tensor.data};

    function = CreateFunction2Plane<ov::op::v17::RGBtoNV12>(inp_tensor);

    Exec();
}

TEST_F(ReferenceConvertColorToNV12LayerTest, CompareWithHardcodedRefs_red_f32_two_bgr) {
    auto input = std::vector<float>{0, 0, 255.f, 0, 0, 255.f, 0, 0, 255.f, 0, 0, 255.f};
    auto input_shape = ov::Shape{1, 2, 2, 3};

    auto exp_y = std::vector<float>{82.f, 82.f, 82.f, 82.f};
    auto exp_y_shape = ov::Shape{1, 2, 2, 1};

    auto exp_uv = std::vector<float>{90.f, 240.f};
    auto exp_uv_shape = ov::Shape{1, 1, 1, 2};

    reference_tests::Tensor inp_tensor(input_shape, ov::element::f32, input);
    inputData = {inp_tensor.data};

    reference_tests::Tensor exp_y_tensor(exp_y_shape, ov::element::f32, exp_y);
    reference_tests::Tensor exp_uv_tensor(exp_uv_shape, ov::element::f32, exp_uv);
    refOutData = {exp_y_tensor.data, exp_uv_tensor.data};

    function = CreateFunction2Plane<ov::op::v17::BGRtoNV12>(inp_tensor);

    Exec();
}

// clang-format off

// Multi-batch tests (N = 2)
TEST_F(ReferenceConvertColorToNV12LayerTest, CompareWithHardcodedRefs_multibatch_u8_single_rgb) {
    auto input = std::vector<uint8_t>{
        255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, // batch 0: (red)
        0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, // batch 1: (green)
    };
    auto input_shape = ov::Shape{2, 2, 2, 3};
    // Single-plane shape [N, H*3/2, W, 1] = [2, 3, 2, 1]
    auto exp_out = std::vector<uint8_t>{82, 82, 82, 82, 90, 240,     // batch 0
                                        145, 145, 145, 145, 54, 34}; // batch 1
    auto out_shape = ov::Shape{2, 3, 2, 1};

    reference_tests::Tensor inp_tensor(input_shape, ov::element::u8, input);
    inputData = {inp_tensor.data};
    function = CreateFunction<ov::op::v17::RGBtoNV12>(inp_tensor);
    reference_tests::Tensor exp_tensor(out_shape, ov::element::u8, exp_out);
    refOutData = {exp_tensor.data};
    Exec();
}

TEST_F(ReferenceConvertColorToNV12LayerTest, CompareWithHardcodedRefs_multibatch_f32_two_bgr) {
    auto input = std::vector<float>{
        0, 0, 255.f, 0, 0, 255.f, 0, 0, 255.f, 0, 0, 255.f,  // batch 0: (red)
        0, 255.f, 0, 0, 255.f, 0, 0, 255.f, 0, 0, 255.f, 0,  // batch 1: (green)
    };
    auto input_shape = ov::Shape{2, 2, 2, 3};

    auto exp_y = std::vector<float>{82.f, 82.f, 82.f, 82.f,      // batch 0
                                    145.f, 145.f, 145.f, 145.f};  // batch 1
    auto exp_y_shape = ov::Shape{2, 2, 2, 1};

    auto exp_uv = std::vector<float>{90.f, 240.f,  // batch 0
                                     54.f, 34.f};   // batch 1
    auto exp_uv_shape = ov::Shape{2, 1, 1, 2};

    reference_tests::Tensor inp_tensor(input_shape, ov::element::f32, input);
    inputData = {inp_tensor.data};

    reference_tests::Tensor exp_y_tensor(exp_y_shape, ov::element::f32, exp_y);
    reference_tests::Tensor exp_uv_tensor(exp_uv_shape, ov::element::f32, exp_uv);
    refOutData = {exp_y_tensor.data, exp_uv_tensor.data};

    function = CreateFunction2Plane<ov::op::v17::BGRtoNV12>(inp_tensor);
    Exec();
}

// Mixed-color 2×2 block tests
TEST_F(ReferenceConvertColorToNV12LayerTest, CompareWithHardcodedRefs_mixed_colors_u8_single_rgb) {
    auto input = std::vector<uint8_t>{
        255, 0,   0,
        0,   255, 0,
        0,   0,   255,
        255, 255, 255,
    };
    auto input_shape = ov::Shape{1, 2, 2, 3};

    auto exp_out = std::vector<uint8_t>{82, 145, 41, 235, 128, 128};
    auto out_shape = ov::Shape{1, 3, 2, 1};

    reference_tests::Tensor inp_tensor(input_shape, ov::element::u8, input);
    inputData = {inp_tensor.data};
    function = CreateFunction<ov::op::v17::RGBtoNV12>(inp_tensor);
    reference_tests::Tensor exp_tensor(out_shape, ov::element::u8, exp_out);
    refOutData = {exp_tensor.data};
    Exec();
}

TEST_F(ReferenceConvertColorToNV12LayerTest, CompareWithHardcodedRefs_mixed_colors_u8_single_bgr) {
    auto input = std::vector<uint8_t>{
        0,   0,   255,
        0,   255, 0,
        255, 0,   0,
        255, 255, 255,
    };
    auto input_shape = ov::Shape{1, 2, 2, 3};
    auto exp_out = std::vector<uint8_t>{82, 145, 41, 235, 128, 128};
    auto out_shape = ov::Shape{1, 3, 2, 1};

    reference_tests::Tensor inp_tensor(input_shape, ov::element::u8, input);
    inputData = {inp_tensor.data};
    function = CreateFunction<ov::op::v17::BGRtoNV12>(inp_tensor);
    reference_tests::Tensor exp_tensor(out_shape, ov::element::u8, exp_out);
    refOutData = {exp_tensor.data};
    Exec();
}

TEST_F(ReferenceConvertColorToNV12LayerTest, CompareWithHardcodedRefs_mixed_colors_u8_two_bgr) {
    auto input = std::vector<uint8_t>{
        0,   0,   255,
        0,   255, 0,
        255, 0,   0,
        255, 255, 255,
    };
    auto input_shape = ov::Shape{1, 2, 2, 3};

    auto exp_y = std::vector<uint8_t>{82, 145, 41, 235};
    auto exp_y_shape = ov::Shape{1, 2, 2, 1};
    auto exp_uv = std::vector<uint8_t>{128, 128};
    auto exp_uv_shape = ov::Shape{1, 1, 1, 2};

    reference_tests::Tensor inp_tensor(input_shape, ov::element::u8, input);
    inputData = {inp_tensor.data};

    reference_tests::Tensor exp_y_tensor(exp_y_shape, ov::element::u8, exp_y);
    reference_tests::Tensor exp_uv_tensor(exp_uv_shape, ov::element::u8, exp_uv);
    refOutData = {exp_y_tensor.data, exp_uv_tensor.data};

    function = CreateFunction2Plane<ov::op::v17::BGRtoNV12>(inp_tensor);
    Exec();
}

// Mixed non-primary RGB values:
//   (128,64,0)   -> Y=81,  U=90,  V=161
//   (0,128,255)  -> Y=106, U=203, V=63
//   (200,0,128)  -> Y=80,  U=155, V=207
//   (64,200,100) -> Y=143, U=104, V=75
//   avg U = (90+203+155+104)/4 = 138, avg V = 506/4 ~= 127

TEST_F(ReferenceConvertColorToNV12LayerTest, CompareWithHardcodedRefs_mixed_rgb_values_u8_single_rgb) {
    auto input = std::vector<uint8_t>{
        128, 64,  0,
        0,   128, 255,
        200, 0,   128,
        64,  200, 100,
    };
    auto input_shape = ov::Shape{1, 2, 2, 3};

    auto exp_out = std::vector<uint8_t>{81, 106, 80, 143, 138, 127};
    auto out_shape = ov::Shape{1, 3, 2, 1};

    reference_tests::Tensor inp_tensor(input_shape, ov::element::u8, input);
    inputData = {inp_tensor.data};
    function = CreateFunction<ov::op::v17::RGBtoNV12>(inp_tensor);
    reference_tests::Tensor exp_tensor(out_shape, ov::element::u8, exp_out);
    refOutData = {exp_tensor.data};
    Exec();
}

TEST_F(ReferenceConvertColorToNV12LayerTest, CompareWithHardcodedRefs_mixed_rgb_values_u8_two_rgb) {
    auto input = std::vector<uint8_t>{
        128, 64,  0,
        0,   128, 255,
        200, 0,   128,
        64,  200, 100,
    };
    auto input_shape = ov::Shape{1, 2, 2, 3};

    auto exp_y = std::vector<uint8_t>{81, 106, 80, 143};
    auto exp_y_shape = ov::Shape{1, 2, 2, 1};
    auto exp_uv = std::vector<uint8_t>{138, 127};
    auto exp_uv_shape = ov::Shape{1, 1, 1, 2};

    reference_tests::Tensor inp_tensor(input_shape, ov::element::u8, input);
    inputData = {inp_tensor.data};

    reference_tests::Tensor exp_y_tensor(exp_y_shape, ov::element::u8, exp_y);
    reference_tests::Tensor exp_uv_tensor(exp_uv_shape, ov::element::u8, exp_uv);
    refOutData = {exp_y_tensor.data, exp_uv_tensor.data};

    function = CreateFunction2Plane<ov::op::v17::RGBtoNV12>(inp_tensor);
    Exec();
}

// 4×4 image tests

TEST_F(ReferenceConvertColorToNV12LayerTest, CompareWithHardcodedRefs_4x4_f32_single_rgb) {
    auto input = std::vector<float>{
        255.f, 0, 0, 255.f, 0, 0, 255.f, 0, 0, 255.f, 0, 0,  // row 0 (red)
        255.f, 0, 0, 255.f, 0, 0, 255.f, 0, 0, 255.f, 0, 0,  // row 1 (red)
        0, 255.f, 0, 0, 255.f, 0, 0, 255.f, 0, 0, 255.f, 0,  // row 2 (green)
        0, 255.f, 0, 0, 255.f, 0, 0, 255.f, 0, 0, 255.f, 0,  // row 3 (green)
    };
    auto input_shape = ov::Shape{1, 4, 4, 3};

    auto exp_out = std::vector<float>{
        81.535f, 81.535f, 81.535f, 81.535f,  // Y row 0 (red)
        81.535f, 81.535f, 81.535f, 81.535f,  // Y row 1 (red)
        144.52f, 144.52f, 144.52f, 144.52f,  // Y row 2 (green)
        144.52f, 144.52f, 144.52f, 144.52f,  // Y row 3 (green)
        90.26f, 239.945f, 90.26f, 239.945f,  // UV row 0
        53.795f, 34.16f,  53.795f, 34.16f,   // UV row 1
    };
    auto out_shape = ov::Shape{1, 6, 4, 1};

    reference_tests::Tensor inp_tensor(input_shape, ov::element::f32, input);
    inputData = {inp_tensor.data};
    function = CreateFunction<ov::op::v17::RGBtoNV12>(inp_tensor);
    reference_tests::Tensor exp_tensor(out_shape, ov::element::f32, exp_out);
    refOutData = {exp_tensor.data};
    Exec();
}

TEST_F(ReferenceConvertColorToNV12LayerTest, CompareWithHardcodedRefs_4x4_f32_two_bgr) {
    auto input = std::vector<float>{
        255.f, 0, 0, 255.f, 0, 0, 255.f, 0, 0, 255.f, 0, 0,  // row 0 (blue)
        255.f, 0, 0, 255.f, 0, 0, 255.f, 0, 0, 255.f, 0, 0,  // row 1 (blue)
        0, 255.f, 0, 0, 255.f, 0, 0, 255.f, 0, 0, 255.f, 0,  // row 2 (green)
        0, 255.f, 0, 0, 255.f, 0, 0, 255.f, 0, 0, 255.f, 0,  // row 3 (green)
    };
    auto input_shape = ov::Shape{1, 4, 4, 3};

    // Y plane [1, 4, 4, 1]
    auto exp_y = std::vector<float>{
        40.99f, 40.99f, 40.99f, 40.99f,
        40.99f, 40.99f, 40.99f, 40.99f,
        144.52f, 144.52f, 144.52f, 144.52f,
        144.52f, 144.52f, 144.52f, 144.52f,
    };
    auto exp_y_shape = ov::Shape{1, 4, 4, 1};

    auto exp_uv = std::vector<float>{
        239.945f, 109.895f, 239.945f, 109.895f,
        53.795f,  34.16f,   53.795f,  34.16f,
    };
    auto exp_uv_shape = ov::Shape{1, 2, 2, 2};

    reference_tests::Tensor inp_tensor(input_shape, ov::element::f32, input);
    inputData = {inp_tensor.data};

    reference_tests::Tensor exp_y_tensor(exp_y_shape, ov::element::f32, exp_y);
    reference_tests::Tensor exp_uv_tensor(exp_uv_shape, ov::element::f32, exp_uv);
    refOutData = {exp_y_tensor.data, exp_uv_tensor.data};

    function = CreateFunction2Plane<ov::op::v17::BGRtoNV12>(inp_tensor);
    Exec();
}

// clang-format on

// Round-trip tests: NV12 (single-plane) -> decode to color -> re-encode to NV12.
TEST_F(ReferenceConvertColorToNV12LayerTest, RoundTrip_red_u8_NV12toRGBtoNV12) {
    // NV12 (single-plane) for a 2×2 all-red image: Y=82, U=90, V=240.
    auto nv12_data = std::vector<uint8_t>{82, 82, 82, 82, 90, 240};
    auto nv12_shape = ov::Shape{1, 3, 2, 1};  // [N, H*3/2, W, 1]

    reference_tests::Tensor inp_tensor(nv12_shape, ov::element::u8, nv12_data);
    inputData = {inp_tensor.data};
    function = CreateRoundTripFunction<ov::op::v8::NV12toRGB, ov::op::v17::RGBtoNV12>(inp_tensor);
    reference_tests::Tensor exp_tensor(nv12_shape, ov::element::u8, nv12_data);
    refOutData = {exp_tensor.data};
    Exec();
}

TEST_F(ReferenceConvertColorToNV12LayerTest, RoundTrip_green_f32_NV12toBGRtoNV12) {
    // NV12 (single-plane) for a 2×2 all-green image: Y=145, U=54, V=34.
    auto nv12_data = std::vector<float>{145.f, 145.f, 145.f, 145.f, 54.f, 34.f};
    auto nv12_shape = ov::Shape{1, 3, 2, 1};

    reference_tests::Tensor inp_tensor(nv12_shape, ov::element::f32, nv12_data);
    inputData = {inp_tensor.data};
    function = CreateRoundTripFunction<ov::op::v8::NV12toBGR, ov::op::v17::BGRtoNV12>(inp_tensor);
    reference_tests::Tensor exp_tensor(nv12_shape, ov::element::f32, nv12_data);
    refOutData = {exp_tensor.data};
    Exec();
}
}  // namespace ov::tests
