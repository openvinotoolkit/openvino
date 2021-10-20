// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ie_ngraph_utils.hpp>
#include <openvino/core/preprocess/pre_post_process.hpp>
#include <shared_test_classes/base/layer_test_utils.hpp>
#include <shared_test_classes/single_layer/convert_color_nv12.hpp>
#include <vector>

#include "base_reference_cnn_test.hpp"
#include "ngraph_functions/builders.hpp"

using namespace ov;
using namespace ov::preprocess;
using namespace reference_tests;
namespace {

class ReferencePreprocessLegacyTest : public testing::Test, public ReferenceCNNTest {
public:
    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
    }
};

} // namespace

static std::shared_ptr<Function> create_simple_function(element::Type type, const PartialShape& shape) {
    auto data1 = std::make_shared<op::v0::Parameter>(type, shape);
    data1->set_friendly_name("input1");
    data1->get_output_tensor(0).set_names({"tensor_input1", "input1"});
    auto c = op::v0::Constant::create(type, {1}, {0});
    auto op = std::make_shared<op::v1::Add>(data1, c);
    op->set_friendly_name("Add0");
    auto res = std::make_shared<op::v0::Result>(op);
    res->set_friendly_name("Result1");
    res->get_output_tensor(0).set_names({"tensor_output1", "Result1", "Add0"});
    return std::make_shared<ov::Function>(ResultVector{res}, ParameterVector{data1});
}

static std::shared_ptr<Function> create_simple_function_nv12(const PartialShape& shape) {
    auto data1 = std::make_shared<op::v0::Parameter>(element::u8, shape);
    data1->set_friendly_name("input1");
    data1->get_output_tensor(0).set_names({"tensor_input1", "input1"});
    auto op = std::make_shared<op::v0::Convert>(data1, element::f32);
    op->set_friendly_name("Convert1");
    auto res = std::make_shared<op::v0::Result>(op);
    res->set_friendly_name("Result1");
    res->get_output_tensor(0).set_names({"tensor_output1", "Result1", "Convert1"});
    return std::make_shared<ov::Function>(ResultVector{res}, ParameterVector{data1});
}

TEST_F(ReferencePreprocessLegacyTest, mean) {
    function = create_simple_function(element::f32, Shape{1, 3, 2, 2});
    function = PrePostProcessor().input(InputInfo().preprocess(PreProcessSteps().mean(1.f))).build(function);

    auto f2 = create_simple_function(element::f32, Shape{1, 3, 2, 2});
    legacy_network = InferenceEngine::CNNNetwork(f2);
    auto &preProcess = legacy_network.getInputsInfo().begin()->second->getPreProcess();
    preProcess.init(3);
    preProcess[0]->meanValue = 1;
    preProcess[1]->meanValue = 1;
    preProcess[2]->meanValue = 1;
    preProcess[0]->stdScale = 1;
    preProcess[1]->stdScale = 1;
    preProcess[2]->stdScale = 1;
    preProcess.setVariant(InferenceEngine::MEAN_VALUE);
    Exec();
}

TEST_F(ReferencePreprocessLegacyTest, mean_scale) {
    function = create_simple_function(element::f32, Shape{1, 3, 20, 20});
    function = PrePostProcessor().input(InputInfo().preprocess(PreProcessSteps().scale(2.f))).build(function);

    auto f2 = create_simple_function(element::f32, Shape{1, 3, 20, 20});
    legacy_network = InferenceEngine::CNNNetwork(f2);
    auto &preProcess = legacy_network.getInputsInfo().begin()->second->getPreProcess();
    preProcess.init(3);
    preProcess[0]->meanValue = 0;
    preProcess[1]->meanValue = 0;
    preProcess[2]->meanValue = 0;
    preProcess[0]->stdScale = 2;
    preProcess[1]->stdScale = 2;
    preProcess[2]->stdScale = 2;
    preProcess.setVariant(InferenceEngine::MEAN_VALUE);
    Exec();
}

TEST_F(ReferencePreprocessLegacyTest, resize) {
    function = create_simple_function(element::f32, Shape{1, 3, 5, 5});
    auto f2 = create_simple_function(element::f32, Shape{1, 3, 5, 5});
    legacy_network = InferenceEngine::CNNNetwork(f2);

    function = PrePostProcessor().input(InputInfo()
            .tensor(InputTensorInfo().set_layout("NCHW").set_spatial_static_shape(42, 30))
            .preprocess(PreProcessSteps().resize(ResizeAlgorithm::RESIZE_LINEAR))
            .network(InputNetworkInfo().set_layout("NCHW")))
                    .build(function);

    auto &preProcess = legacy_network.getInputsInfo().begin()->second->getPreProcess();
    preProcess.setResizeAlgorithm(InferenceEngine::ResizeAlgorithm::RESIZE_BILINEAR);
    Exec();
}

class ConvertNV12WithLegacyTest: public ReferencePreprocessLegacyTest {
public:
    // Create OV20 function with pre-processing +  legacy network + reference NV12 inputs
    void SetupAndExec(size_t height, size_t width, std::vector<uint8_t>& ov20_input_yuv) {
        function = create_simple_function_nv12(Shape{1, 3, height, width});
        auto f2 = create_simple_function_nv12(Shape{1, 3, height, width});
        legacy_network = InferenceEngine::CNNNetwork(f2);
        inputData.clear();
        legacy_input_blobs.clear();

        function = PrePostProcessor().input(InputInfo()
                                                    .tensor(InputTensorInfo().set_color_format(
                                                            ColorFormat::NV12_SINGLE_PLANE))
                                                    .preprocess(PreProcessSteps().convert_color(ColorFormat::BGR))
                                                    .network(InputNetworkInfo().set_layout("NCHW")))
                .build(function);

        const auto &param = function->get_parameters()[0];
        inputData.emplace_back(param->get_element_type(), param->get_shape(), ov20_input_yuv.data());

        // Legacy way
        legacy_network.getInputsInfo().begin()->second->setLayout(InferenceEngine::Layout::NCHW);
        legacy_network.getInputsInfo().begin()->second->setPrecision(InferenceEngine::Precision::U8);

        auto &preProcess = legacy_network.getInputsInfo().begin()->second->getPreProcess();
        preProcess.setColorFormat(InferenceEngine::NV12);
        // Fill legacy blob
        auto legacy_input_y = std::vector<uint8_t>(ov20_input_yuv.begin(),
                                                   ov20_input_yuv.begin() + ov20_input_yuv.size() * 2 / 3);
        auto legacy_input_uv = std::vector<uint8_t>(ov20_input_yuv.begin() + ov20_input_yuv.size() * 2 / 3,
                                                    ov20_input_yuv.end());
        const InferenceEngine::TensorDesc y_plane_desc(InferenceEngine::Precision::U8,
                                                       {1, 1, height, width},
                                                       InferenceEngine::Layout::NHWC);
        const InferenceEngine::TensorDesc uv_plane_desc(InferenceEngine::Precision::U8,
                                                        {1, 2, height / 2, width / 2},
                                                        InferenceEngine::Layout::NHWC);

        auto y_blob = InferenceEngine::make_shared_blob<uint8_t>(y_plane_desc, legacy_input_y.data());
        auto uv_blob = InferenceEngine::make_shared_blob<uint8_t>(uv_plane_desc, legacy_input_uv.data());
        legacy_input_blobs["input1"] = InferenceEngine::make_shared_blob<InferenceEngine::NV12Blob>(y_blob, uv_blob);

        // Exec now
        Exec();
    }

    void Validate() override {
        threshold = 1.f;
        abs_threshold = 1.f;
        // No pixels with deviation of more than 1 color step
        ReferencePreprocessLegacyTest::Validate();

        // Less than 2% of deviations with 1 color step. 2% is experimental value
        // For very precise (acceptable) float calculations - 1.4% deviation with G-API/OpenCV is observed
        LayerTestsDefinitions::NV12TestUtils::ValidateColors(outputs_legacy[0].data<float>(),
                                            outputs_ov20[0].data<float>(), outputs_legacy[0].get_size(), 0.02);
    }
};

TEST_F(ConvertNV12WithLegacyTest, convert_nv12_full_color_range) {
    size_t height = 128;
    size_t width = 128;
    int b_step = 5;
    int b_dim = 255 / b_step + 1;

    // Test various possible r/g/b values within dimensions
    auto ov20_input_yuv = LayerTestsDefinitions::NV12TestUtils::color_test_image(height, width, b_step);

    SetupAndExec(height * b_dim, width, ov20_input_yuv);
}

TEST_F(ConvertNV12WithLegacyTest, convert_nv12_colored) {
    auto input_yuv = std::vector<uint8_t> {235, 81, 235, 81, 109, 184};
    SetupAndExec(2, 2, input_yuv);
}