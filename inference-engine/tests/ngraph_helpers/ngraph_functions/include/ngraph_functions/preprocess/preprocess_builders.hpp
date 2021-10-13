// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/preprocess/pre_post_process.hpp"

namespace ov {
namespace builder {
namespace preprocess {

using preprocess_func = std::tuple<std::function<std::shared_ptr<Function>()>, std::string, float>;

inline std::vector<preprocess_func> generic_preprocess_functions();


/// -------- Functions ---------------

inline std::shared_ptr<Function> create_preprocess_1input(element::Type type,
                                                          const PartialShape& shape) {
    auto data1 = std::make_shared<op::v0::Parameter>(type, shape);
    data1->set_friendly_name("input1");
    data1->output(0).get_tensor().set_names({"input1"});
    std::shared_ptr<op::v0::Result> res;
    if (type == element::f32) {
        res = std::make_shared<op::v0::Result>(data1);
    } else {
        auto convert = std::make_shared<op::v0::Convert>(data1, element::f32);
        res = std::make_shared<op::v0::Result>(convert);
    }
    res->set_friendly_name("Result");
    return std::make_shared<Function>(ResultVector{res}, ParameterVector{data1});
}

inline std::shared_ptr<Function> create_preprocess_2inputs(element::Type type,
                                                           const PartialShape& shape) {
    auto data1 = std::make_shared<op::v0::Parameter>(type, shape);
    data1->set_friendly_name("input1");
    data1->output(0).get_tensor().set_names({"input1"});
    auto data2 = std::make_shared<op::v0::Parameter>(type, shape);
    data2->set_friendly_name("input2");
    data2->output(0).get_tensor().set_names({"input2"});
    std::shared_ptr<op::v0::Result> res1, res2;
    if (type == element::f32) {
        res1 = std::make_shared<op::v0::Result>(data1);
        res2 = std::make_shared<op::v0::Result>(data2);
    } else {
        auto convert1 = std::make_shared<op::v0::Convert>(data1, element::f32);
        res1 = std::make_shared<op::v0::Result>(convert1);
        auto convert2 = std::make_shared<op::v0::Convert>(data2, element::f32);
        res2 = std::make_shared<op::v0::Result>(convert2);
    }
    res1->set_friendly_name("Result1");
    res2->set_friendly_name("Result2");
    return std::make_shared<Function>(ResultVector{res1, res2}, ParameterVector{data1, data2});
}

inline std::shared_ptr<Function> mean_only() {
    using namespace ov::preprocess;
    auto function = create_preprocess_1input(element::f32, Shape{1, 3, 24, 24});
    function = PrePostProcessor().input(InputInfo().preprocess(PreProcessSteps().mean(1.1f))).build(function);
    return function;
}

inline std::shared_ptr<Function> scale_only() {
    using namespace ov::preprocess;
    auto function = create_preprocess_1input(element::f32, Shape{1, 3, 24, 24});
    function = PrePostProcessor().input(InputInfo().preprocess(PreProcessSteps().scale(2.1f))).build(function);
    return function;
}

inline std::shared_ptr<Function> mean_scale() {
    using namespace ov::preprocess;
    auto function = create_preprocess_1input(element::f32, Shape{1, 3, 24, 24});
    function = PrePostProcessor().input(InputInfo().preprocess(PreProcessSteps().mean(1.1f).scale(2.1f))).build(function);
    return function;
}

inline std::shared_ptr<Function> scale_mean() {
    using namespace ov::preprocess;
    auto function = create_preprocess_1input(element::f32, Shape{1, 3, 24, 24});
    function = PrePostProcessor().input(InputInfo().preprocess(PreProcessSteps().scale(2.1f).mean(1.1f))).build(function);
    return function;
}

inline std::shared_ptr<Function> mean_vector() {
    using namespace ov::preprocess;
    auto function = create_preprocess_1input(element::f32, Shape{1, 3, 24, 24});
    function = PrePostProcessor().input(InputInfo()
                                                .tensor(InputTensorInfo().set_layout("NCHW"))
                                                .preprocess(PreProcessSteps().mean({2.2f, 3.3f, 4.4f}))).build(function);
    return function;
}

inline std::shared_ptr<Function> scale_vector() {
    using namespace ov::preprocess;
    auto function = create_preprocess_1input(element::f32, Shape{1, 3, 24, 24});
    function = PrePostProcessor().input(InputInfo()
                                                .tensor(InputTensorInfo().set_layout("NCHW"))
                                                .preprocess(PreProcessSteps().scale({2.2f, 3.3f, 4.4f}))).build(function);
    return function;
}

inline std::shared_ptr<Function> convert_element_type_and_mean() {
    using namespace ov::preprocess;
    auto function = create_preprocess_1input(element::u8, Shape{1, 3, 24, 24});
    function = PrePostProcessor()
            .input(InputInfo()
                           .preprocess(PreProcessSteps()
                                               .convert_element_type(element::f32)
                                               .mean(0.2f)
                                               .convert_element_type(element::u8)))
            .build(function);
    return function;
}

inline std::shared_ptr<Function> tensor_element_type_and_mean() {
    using namespace ov::preprocess;
    auto function = create_preprocess_1input(element::u8, Shape{1, 3, 12, 12});
    function = PrePostProcessor()
            .input(InputInfo()
                           .tensor(InputTensorInfo().set_element_type(element::f32))
                           .preprocess(PreProcessSteps().mean(0.1f).convert_element_type(element::u8)))
            .build(function);
    return function;
}

inline std::shared_ptr<Function> custom_preprocessing() {
    using namespace ov::preprocess;
    auto function = create_preprocess_1input(element::i32, Shape{3, 4, 10, 20});
    function = PrePostProcessor()
            .input(InputInfo().preprocess(PreProcessSteps().custom([](const Output<Node>& node) {
                auto abs = std::make_shared<op::v0::Abs>(node);
                abs->set_friendly_name(node.get_node_shared_ptr()->get_friendly_name() + "/abs");
                return abs;
            })))
            .build(function);
    return function;
}

inline std::shared_ptr<Function> lvalues_multiple_ops() {
    using namespace ov::preprocess;
    auto function = create_preprocess_1input(element::u8, Shape{1, 3, 3, 3});
    auto p = PrePostProcessor();
    auto p1 = std::move(p);
    p = std::move(p1);
    auto inputInfo = InputInfo();
    auto inputInfo2 = std::move(inputInfo);
    inputInfo = std::move(inputInfo2);
    {
        auto inputTensorInfo = InputTensorInfo();
        auto inputTensorInfo2 = std::move(inputTensorInfo);
        inputTensorInfo = std::move(inputTensorInfo2);
        auto& same = inputTensorInfo.set_element_type(element::f32);
        same.set_layout("?CHW");
        inputInfo.tensor(std::move(same));
    }
    {
        auto preprocessSteps = PreProcessSteps();
        auto preprocessSteps2 = std::move(preprocessSteps);
        preprocessSteps = std::move(preprocessSteps2);
        preprocessSteps.mean(1.f);
        preprocessSteps.scale(2.f);
        preprocessSteps.mean({1.1f, 2.2f, 3.3f});
        preprocessSteps.scale({2.f, 3.f, 4.f});
        preprocessSteps.custom([](const Output<Node>& node) {
            auto abs = std::make_shared<op::v0::Abs>(node);
            abs->set_friendly_name(node.get_node_shared_ptr()->get_friendly_name() + "/abs");
            return abs;
        });
        auto& same = preprocessSteps.convert_element_type(element::u8);
        inputInfo.preprocess(std::move(same));
    }
    p.input(std::move(inputInfo));
    function = p.build(function);
    return function;
}

inline std::shared_ptr<Function> two_inputs_basic() {
    using namespace ov::preprocess;
    auto function = create_preprocess_2inputs(element::f32, Shape{1, 3, 1, 1});
    function = PrePostProcessor().input(InputInfo(1).preprocess(PreProcessSteps().mean(1.f).scale(2.0f))).build(function);
    return function;
}

inline std::shared_ptr<Function> reuse_network_layout() {
    using namespace ov::preprocess;
    auto function = create_preprocess_1input(element::f32, PartialShape{4, 3, 2, 1});
    function->get_parameters().front()->set_layout("NC??");
    function = PrePostProcessor()
            .input(InputInfo().preprocess(PreProcessSteps().mean({1.1f, 2.2f, 3.3f}).scale({2.f, 3.f, 4.f})))
            .build(function);
    return function;
}

inline std::shared_ptr<Function> tensor_layout() {
    using namespace ov::preprocess;
    auto function = create_preprocess_1input(element::f32, PartialShape{4, 3, 2, 1});
    function->get_parameters().front()->set_layout("NC??");
    function = PrePostProcessor()
            .input(InputInfo()
                           .tensor(InputTensorInfo().set_layout("NC??"))
                           .preprocess(PreProcessSteps().mean({1.1f, 2.2f, 3.3f}).scale({2.f, 3.f, 4.f})))
            .build(function);
    return function;
}

inline std::shared_ptr<Function> resize_linear() {
    using namespace ov::preprocess;
    auto function = create_preprocess_1input(element::f32, PartialShape{1, 3, 10, 10});
    function = PrePostProcessor()
            .input(InputInfo()
                           .tensor(InputTensorInfo().set_spatial_static_shape(20, 20))
                           .preprocess(PreProcessSteps().resize(ResizeAlgorithm::RESIZE_LINEAR))
                           .network(InputNetworkInfo().set_layout("NCHW")))
            .build(function);
    return function;
}

inline std::shared_ptr<Function> resize_nearest() {
    using namespace ov::preprocess;
    auto function = create_preprocess_1input(element::f32, PartialShape{1, 3, 10, 10});
    function = PrePostProcessor()
            .input(InputInfo()
                           .tensor(InputTensorInfo().set_spatial_static_shape(20, 20))
                           .preprocess(PreProcessSteps().resize(ResizeAlgorithm::RESIZE_NEAREST))
                           .network(InputNetworkInfo().set_layout("NCHW")))
            .build(function);
    return function;
}

inline std::shared_ptr<Function> resize_linear_nhwc() {
    using namespace ov::preprocess;
    auto function = create_preprocess_1input(element::f32, PartialShape{1, 10, 10, 3});
    function = PrePostProcessor()
            .input(InputInfo()
                           .tensor(InputTensorInfo().set_spatial_static_shape(20, 20))
                           .preprocess(PreProcessSteps().resize(ResizeAlgorithm::RESIZE_LINEAR))
                           .network(InputNetworkInfo().set_layout("NHWC")))
            .build(function);
    return function;
}

inline std::shared_ptr<Function> resize_cubic() {
    using namespace ov::preprocess;
    auto function = create_preprocess_1input(element::f32, PartialShape{1, 3, 20, 20});
    function = PrePostProcessor()
            .input(InputInfo()
                           .tensor(InputTensorInfo().set_spatial_static_shape(10, 10))
                           .preprocess(PreProcessSteps().resize(ResizeAlgorithm::RESIZE_CUBIC))
                           .network(InputNetworkInfo().set_layout("NCHW")))
            .build(function);
    return function;
}

inline std::shared_ptr<Function> resize_and_convert_layout() {
    using namespace ov::preprocess;
    auto function = create_preprocess_1input(element::f32, PartialShape{1, 30, 20, 3});
    function = PrePostProcessor()
            .input(InputInfo()
                           .tensor(InputTensorInfo()
                                           .set_layout("NHWC")
                                           .set_spatial_static_shape(40, 30))
                           .preprocess(PreProcessSteps()
                                               .convert_layout()
                                               .resize(ResizeAlgorithm::RESIZE_LINEAR))
                           .network(InputNetworkInfo().set_layout("NCHW")))
            .build(function);
    return function;
}

inline std::shared_ptr<Function> resize_and_convert_layout_i8() {
    using namespace ov::preprocess;
    auto function = create_preprocess_1input(element::i8, PartialShape{1, 30, 20, 3});
    function = PrePostProcessor()
            .input(InputInfo()
                           .tensor(InputTensorInfo()
                                           .set_layout("NHWC")
                                           .set_spatial_static_shape(40, 30))
                           .preprocess(PreProcessSteps()
                                               .convert_layout()
                                               .resize(ResizeAlgorithm::RESIZE_LINEAR))
                           .network(InputNetworkInfo().set_layout("NCHW")))
            .build(function);
    return function;
}

inline std::shared_ptr<Function> cvt_color_nv12_to_rgb_single_plane() {
    using namespace ov::preprocess;
    auto function = create_preprocess_1input(element::f32, PartialShape{1, 20, 20, 3});
    function = PrePostProcessor()
            .input(InputInfo()
                           .tensor(InputTensorInfo().set_color_format(ColorFormat::NV12_SINGLE_PLANE))
                           .preprocess(PreProcessSteps().convert_color(ColorFormat::RGB)))
            .build(function);
    return function;
}

inline std::shared_ptr<Function> cvt_color_nv12_to_bgr_two_planes() {
    using namespace ov::preprocess;
    auto function = create_preprocess_1input(element::f32, PartialShape{1, 20, 20, 3});
    function = PrePostProcessor()
            .input(InputInfo()
                           .tensor(InputTensorInfo().set_color_format(ColorFormat::NV12_TWO_PLANES))
                           .preprocess(PreProcessSteps().convert_color(ColorFormat::BGR)))
            .build(function);
    return function;
}

inline std::shared_ptr<Function> cvt_color_nv12_cvt_layout_resize() {
    using namespace ov::preprocess;
    auto function = create_preprocess_1input(element::f32, PartialShape{1, 3, 10, 10});
    function = PrePostProcessor()
            .input(InputInfo()
                           .tensor(InputTensorInfo()
                                .set_color_format(ColorFormat::NV12_TWO_PLANES)
                                .set_element_type(element::u8)
                                .set_spatial_static_shape(20, 20))
                           .preprocess(PreProcessSteps()
                                .convert_color(ColorFormat::RGB)
                                .convert_layout()
                                .convert_element_type(element::f32)
                                .resize(ResizeAlgorithm::RESIZE_LINEAR))
                           .network(InputNetworkInfo().set_layout("NCHW")))
            .build(function);
    return function;
}

inline std::vector<preprocess_func> generic_preprocess_functions() {
    return std::vector<preprocess_func> {
            preprocess_func(mean_only, "mean_only", 0.01f),
            preprocess_func(scale_only, "scale_only", 0.01f),
            preprocess_func(mean_scale, "mean_scale", 0.01f),
            preprocess_func(scale_mean, "scale_mean", 0.01f),
            preprocess_func(mean_vector, "mean_vector", 0.01f),
            preprocess_func(scale_vector, "scale_vector", 0.01f),
            preprocess_func(convert_element_type_and_mean, "convert_element_type_and_mean", 0.01f),
            preprocess_func(tensor_element_type_and_mean, "tensor_element_type_and_mean", 0.01f),
            preprocess_func(custom_preprocessing, "custom_preprocessing", 0.01f),
            preprocess_func(lvalues_multiple_ops, "lvalues_multiple_ops", 0.01f),
            preprocess_func(two_inputs_basic, "two_inputs_basic", 0.01f),
            preprocess_func(reuse_network_layout, "reuse_network_layout", 0.01f),
            preprocess_func(tensor_layout, "tensor_layout", 0.01f),
            preprocess_func(resize_linear, "resize_linear", 0.01f),
            preprocess_func(resize_nearest, "resize_nearest", 0.01f),
            preprocess_func(resize_linear_nhwc, "resize_linear_nhwc", 0.01f),
            preprocess_func(resize_cubic, "resize_cubic", 0.01f),
            preprocess_func(resize_and_convert_layout, "resize_and_convert_layout", 0.01f),
            preprocess_func(resize_and_convert_layout_i8, "resize_and_convert_layout_i8", 0.01f),
            preprocess_func(cvt_color_nv12_to_rgb_single_plane, "cvt_color_nv12_to_rgb_single_plane", 2.f),
            preprocess_func(cvt_color_nv12_to_bgr_two_planes, "cvt_color_nv12_to_bgr_two_planes", 2.f),
            preprocess_func(cvt_color_nv12_cvt_layout_resize, "cvt_color_nv12_cvt_layout_resize", 2.f),
    };
}

}  // namespace preprocess
}  // namespace builder
}  // namespace ov
