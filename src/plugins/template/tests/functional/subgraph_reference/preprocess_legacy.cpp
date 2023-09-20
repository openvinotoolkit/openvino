// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <vector>

#include "base_reference_cnn_test.hpp"
#include "openvino/core/preprocess/pre_post_process.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "shared_test_classes/single_layer/convert_color_i420.hpp"
#include "shared_test_classes/single_layer/convert_color_nv12.hpp"

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

}  // namespace

static std::shared_ptr<Model> create_simple_function(element::Type type, const PartialShape& shape) {
    auto data1 = std::make_shared<op::v0::Parameter>(type, shape);
    data1->set_friendly_name("input1");
    data1->get_output_tensor(0).set_names({"tensor_input1", "input1"});
    auto c = op::v0::Constant::create(type, {1}, {0});
    auto op = std::make_shared<op::v1::Add>(data1, c);
    op->set_friendly_name("Add0");
    auto res = std::make_shared<op::v0::Result>(op);
    res->set_friendly_name("Result1");
    res->get_output_tensor(0).set_names({"tensor_output1", "Result1", "Add0"});
    return std::make_shared<ov::Model>(ResultVector{res}, ParameterVector{data1});
}

TEST_F(ReferencePreprocessLegacyTest, mean) {
    function = create_simple_function(element::f32, Shape{1, 3, 2, 2});
    auto p = PrePostProcessor(function);
    p.input().preprocess().mean(1.f);
    p.build();

    auto f2 = create_simple_function(element::f32, Shape{1, 3, 2, 2});
    legacy_network = InferenceEngine::CNNNetwork(f2);
    auto& preProcess = legacy_network.getInputsInfo().begin()->second->getPreProcess();
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
    auto p = PrePostProcessor(function);
    p.input().preprocess().scale(2.f);
    p.build();

    auto f2 = create_simple_function(element::f32, Shape{1, 3, 20, 20});
    legacy_network = InferenceEngine::CNNNetwork(f2);
    auto& preProcess = legacy_network.getInputsInfo().begin()->second->getPreProcess();
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

    auto p = PrePostProcessor(function);
    p.input().tensor().set_layout("NCHW").set_spatial_static_shape(42, 30);
    p.input().preprocess().resize(ResizeAlgorithm::RESIZE_LINEAR);
    p.input().model().set_layout("NCHW");
    p.build();

    auto& preProcess = legacy_network.getInputsInfo().begin()->second->getPreProcess();
    preProcess.setResizeAlgorithm(InferenceEngine::ResizeAlgorithm::RESIZE_BILINEAR);
    Exec();
}

TEST_F(ReferencePreprocessLegacyTest, bgrx_to_bgr) {
    const int h = 160;
    const int w = 160;
    auto rgbx_input = std::vector<uint8_t>(h * w * 4, 0);
    for (auto i = 0; i < h * w * 4; i++) {
        rgbx_input[i] = i % 256;
    }
    function = create_simple_function(element::f32, Shape{1, 3, h, w});
    auto f2 = create_simple_function(element::f32, Shape{1, 3, h, w});
    legacy_network = InferenceEngine::CNNNetwork(f2);

    auto p = PrePostProcessor(function);
    auto& input = p.input();
    input.tensor().set_color_format(ColorFormat::BGRX).set_element_type(element::u8);
    input.preprocess().convert_color(ColorFormat::BGR);
    input.model().set_layout("NCHW");
    function = p.build();
    inputData.emplace_back(element::u8, Shape{1, h, w, 4}, rgbx_input.data());

    InferenceEngine::TensorDesc rgbx_plane_desc(InferenceEngine::Precision::U8,
                                                {1, 4, h, w},
                                                InferenceEngine::Layout::NHWC);
    legacy_network.getInputsInfo().begin()->second->setLayout(InferenceEngine::NHWC);
    auto& preProcess = legacy_network.getInputsInfo().begin()->second->getPreProcess();
    preProcess.setColorFormat(InferenceEngine::ColorFormat::BGRX);
    legacy_input_blobs["input1"] = InferenceEngine::make_shared_blob<uint8_t>(rgbx_plane_desc, rgbx_input.data());

    Exec();
}

TEST_F(ReferencePreprocessLegacyTest, rgbx_to_bgr) {
    const int h = 160;
    const int w = 160;
    auto rgbx_input = std::vector<uint8_t>(h * w * 4, 0);
    for (auto i = 0; i < h * w * 4; i++) {
        rgbx_input[i] = i % 256;
    }
    function = create_simple_function(element::f32, Shape{1, 3, h, w});
    auto f2 = create_simple_function(element::f32, Shape{1, 3, h, w});
    legacy_network = InferenceEngine::CNNNetwork(f2);

    auto p = PrePostProcessor(function);
    auto& input = p.input();
    input.tensor().set_color_format(ColorFormat::RGBX).set_element_type(element::u8);
    input.preprocess().convert_color(ColorFormat::BGR);
    input.model().set_layout("NCHW");
    function = p.build();
    inputData.emplace_back(element::u8, Shape{1, h, w, 4}, rgbx_input.data());

    InferenceEngine::TensorDesc rgbx_plane_desc(InferenceEngine::Precision::U8,
                                                {1, 4, h, w},
                                                InferenceEngine::Layout::NHWC);
    legacy_network.getInputsInfo().begin()->second->setLayout(InferenceEngine::NHWC);
    auto& preProcess = legacy_network.getInputsInfo().begin()->second->getPreProcess();
    preProcess.setColorFormat(InferenceEngine::ColorFormat::RGBX);
    legacy_input_blobs["input1"] = InferenceEngine::make_shared_blob<uint8_t>(rgbx_plane_desc, rgbx_input.data());

    Exec();
}
