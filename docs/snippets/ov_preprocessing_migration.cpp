// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <openvino/runtime/core.hpp>
#include <openvino/opsets/opset8.hpp>
#include <openvino/core/preprocess/pre_post_process.hpp>

#ifndef IN_OV_COMPONENT
#    define IN_OV_COMPONENT
#    define WAS_OV_LIBRARY_DEFINED
#endif

#include "inference_engine.hpp"

#ifdef WAS_OV_LIBRARY_DEFINED
#    undef IN_OV_COMPONENT
#    undef WAS_OV_LIBRARY_DEFINED
#endif

int main_new() {
    std::string model_path;
    std::string tensor_name;

    ov::Core core;
    std::shared_ptr<ov::Model> model = core.read_model(model_path);
    ov::preprocess::PrePostProcessor ppp(model);

    {
    //! [ov_mean_scale]
ov::preprocess::PrePostProcessor ppp(model);
ov::preprocess::InputInfo& input = ppp.input(tensor_name);
// we only need to know where is C dimension
input.model().set_layout("...C");
// specify scale and mean values, order of operations is important
input.preprocess().mean(116.78f).scale({ 57.21f, 57.45f, 57.73f });
// insert preprocessing operations to the 'model'
model = ppp.build();
    //! [ov_mean_scale]
    }

    {
    //! [ov_conversions]
ov::preprocess::PrePostProcessor ppp(model);
ov::preprocess::InputInfo& input = ppp.input(tensor_name);
input.tensor().set_layout("NHWC").set_element_type(ov::element::u8);
input.model().set_layout("NCHW");
// layout and precision conversion is inserted automatically,
// because tensor format != model input format
model = ppp.build();
    //! [ov_conversions]
    }

    {
    //! [ov_color_space]
ov::preprocess::PrePostProcessor ppp(model);
ov::preprocess::InputInfo& input = ppp.input(tensor_name);
input.tensor().set_color_format(ov::preprocess::ColorFormat::NV12_TWO_PLANES);
// add NV12 to BGR conversion
input.preprocess().convert_color(ov::preprocess::ColorFormat::BGR);
// and insert operations to the model
model = ppp.build();
    //! [ov_color_space]
    }

    {
    //! [ov_image_scale]
ov::preprocess::PrePostProcessor ppp(model);
ov::preprocess::InputInfo& input = ppp.input(tensor_name);
// scale from the specified tensor size
input.tensor().set_spatial_static_shape(448, 448);
// need to specify H and W dimensions in model, others are not important
input.model().set_layout("??HW");
// scale to model shape
input.preprocess().resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR);
// and insert operations to the model
model = ppp.build();
    //! [ov_image_scale]
    }

return 0;
}

int main_old() {
    std::string model_path;
    std::string operation_name;

    InferenceEngine::Core core;
    InferenceEngine::CNNNetwork network = core.ReadNetwork(model_path);

    {
    //! [mean_scale]
auto preProcess = network.getInputsInfo()[operation_name]->getPreProcess();
preProcess.init(3);
preProcess[0]->meanValue = 116.78f;
preProcess[1]->meanValue = 116.78f;
preProcess[2]->meanValue = 116.78f;
preProcess[0]->stdScale = 57.21f;
preProcess[1]->stdScale = 57.45f;
preProcess[2]->stdScale = 57.73f;
preProcess.setVariant(InferenceEngine::MEAN_VALUE);
    //! [mean_scale]
    }

    {
    //! [conversions]
auto inputInfo = network.getInputsInfo()[operation_name];
inputInfo->setPrecision(InferenceEngine::Precision::U8);
inputInfo->setLayout(InferenceEngine::Layout::NHWC);
// model input layout is always NCHW in Inference Engine
// for shapes with 4 dimensions
    //! [conversions]
    }

    {
    //! [image_scale]
auto preProcess = network.getInputsInfo()[operation_name]->getPreProcess();
// Inference Engine supposes input for resize is always in NCHW layout
// while for OpenVINO Runtime API 2.0 `H` and `W` dimensions must be specified
// Also, current code snippet supposed resize from dynamic shapes
preProcess.setResizeAlgorithm(InferenceEngine::ResizeAlgorithm::RESIZE_BILINEAR);
    //! [image_scale]
    }

    return 0;
}
