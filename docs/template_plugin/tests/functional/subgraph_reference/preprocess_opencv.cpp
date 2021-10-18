// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef OPENCV_TEMPLATE_TESTS

#include <gtest/gtest.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>

#include <openvino/core/preprocess/pre_post_process.hpp>
#include <shared_test_classes/base/layer_test_utils.hpp>
#include <shared_test_classes/single_layer/convert_color_nv12.hpp>

#include "base_reference_test.hpp"
#include "ngraph_functions/builders.hpp"

using namespace ov;
using namespace ov::preprocess;
using namespace reference_tests;
namespace {

class PreprocessOpenCVReferenceTest : public testing::Test, public CommonReferenceTest {
public:
    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
    }
    void Validate() override {
        threshold = 1.f;
        abs_threshold = 1.f;
        // No pixels with deviation of more than 1 color step
        CommonReferenceTest::Validate();
        // Less than 2% of deviations with 1 color step. 2% is experimental value
        // For very precise (acceptable) float calculations - 1.4% deviation with G-API/OpenCV is observed
        LayerTestsDefinitions::NV12TestUtils::ValidateColors(refOutData[0].data<uint8_t>(),
                actualOutData[0].data<uint8_t>(), refOutData[0].get_size(), 0.02);
    }
};

} // namespace

static std::shared_ptr<Function> create_simple_function_nv12(element::Type type, const PartialShape& shape) {
    auto data1 = std::make_shared<op::v0::Parameter>(type, shape);
    data1->set_friendly_name("input1");
    data1->get_output_tensor(0).set_names({"tensor_input1", "input1"});
    auto c = op::v0::Constant::create(type, {1}, {0});
    auto op = std::make_shared<op::v1::Add>(data1, c);
    op->set_friendly_name("Add0");
    auto res = std::make_shared<op::v0::Result>(op);
    res->set_friendly_name("Result1");
    res->get_output_tensor(0).set_names({"tensor_output1", "Result1", "Convert1"});
    return std::make_shared<ov::Function>(ResultVector{res}, ParameterVector{data1});
}

TEST_F(PreprocessOpenCVReferenceTest, convert_nv12_full_color_range) {
    size_t height = 64; // 64/2 = 32 values for R
    size_t width = 64;  // 64/2 = 32 values for G
    int b_step = 5;
    int b_dim = 255 / b_step + 1;

    // Test various possible r/g/b values within dimensions
    auto ov20_input_yuv = LayerTestsDefinitions::NV12TestUtils::color_test_image(height, width, b_step);

    auto full_height = height * b_dim;
    auto func_shape = Shape{1, full_height, width, 3};
    function = create_simple_function_nv12(element::u8, func_shape);

    inputData.clear();

    function = PrePostProcessor().input(InputInfo()
                                                .tensor(InputTensorInfo().set_color_format(
                                                        ColorFormat::NV12_SINGLE_PLANE))
                                                .preprocess(PreProcessSteps().convert_color(ColorFormat::BGR)))
            .build(function);

    const auto &param = function->get_parameters()[0];
    inputData.emplace_back(param->get_element_type(), param->get_shape(), ov20_input_yuv.data());

    // Calculate reference expected values from OpenCV
    cv::Mat picYV12 = cv::Mat(static_cast<int>(full_height) * 3 / 2,
                              static_cast<int>(width),
                              CV_8UC1,
                              ov20_input_yuv.data());
    cv::Mat picBGR;
    cv::cvtColor(picYV12, picBGR, CV_YUV2BGR_NV12);
    refOutData.emplace_back(param->get_element_type(), func_shape, picBGR.data);

    // Exec now
    Exec();
}

TEST_F(PreprocessOpenCVReferenceTest, convert_nv12_colored) {
    auto input_yuv = std::vector<uint8_t> {235, 81, 235, 81, 109, 184};
    auto func_shape = Shape{1, 2, 2, 3};
    function = create_simple_function_nv12(element::u8, func_shape);

    inputData.clear();

    function = PrePostProcessor().input(InputInfo()
                                                .tensor(InputTensorInfo().set_color_format(
                                                        ColorFormat::NV12_SINGLE_PLANE))
                                                .preprocess(PreProcessSteps().convert_color(ColorFormat::BGR))
                                                )
            .build(function);

    const auto &param = function->get_parameters()[0];
    inputData.emplace_back(param->get_element_type(), param->get_shape(), input_yuv.data());

    // Calculate reference expected values from OpenCV
    cv::Mat picYV12 = cv::Mat(3, 2, CV_8UC1, input_yuv.data());
    cv::Mat picBGR;
    cv::cvtColor(picYV12, picBGR, CV_YUV2BGR_NV12);
    refOutData.emplace_back(param->get_element_type(), func_shape, picBGR.data);
    // Exec now
    Exec();
}

#endif // OPENCV_TEMPLATE_TESTS