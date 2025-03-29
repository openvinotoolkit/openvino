// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/model.hpp"
#include "openvino/core/preprocess/pre_post_process.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"

namespace ov {
namespace builder {
namespace preprocess {

struct preprocess_func {
    preprocess_func() = default;
    preprocess_func(const std::function<std::shared_ptr<Model>()>& f,
                    const std::string& name,
                    float acc,
                    const std::vector<Shape>& shapes = {})
        : m_function(f),
          m_name(name),
          m_accuracy(acc),
          m_shapes(shapes) {}
    std::function<std::shared_ptr<Model>()> m_function = nullptr;
    std::string m_name = {};
    float m_accuracy = 0.01f;
    std::vector<Shape> m_shapes = {};
};

inline std::vector<preprocess_func> generic_preprocess_functions();

using postprocess_func = preprocess_func;
inline std::vector<postprocess_func> generic_postprocess_functions();

/// -------- Functions ---------------

inline std::shared_ptr<Model> create_preprocess_1input(ov::element::Type type, const PartialShape& shape) {
    auto data1 = std::make_shared<op::v0::Parameter>(type, shape);
    data1->set_friendly_name("input1");
    data1->output(0).get_tensor().set_names({"input1"});
    std::shared_ptr<op::v0::Result> res;
    auto op1 = std::make_shared<op::v0::Abs>(data1);
    op1->set_friendly_name("abs1");
    if (type == element::f32) {
        res = std::make_shared<op::v0::Result>(op1);
    } else {
        auto convert = std::make_shared<op::v0::Convert>(data1, element::f32);
        res = std::make_shared<op::v0::Result>(op1);
    }
    res->set_friendly_name("Result1");
    res->output(0).get_tensor().set_names({"Result1"});
    return std::make_shared<Model>(ResultVector{res}, ParameterVector{data1});
}

inline std::shared_ptr<Model> create_preprocess_2inputs(ov::element::Type type, const PartialShape& shape) {
    auto data1 = std::make_shared<op::v0::Parameter>(type, shape);
    data1->set_friendly_name("input1");
    data1->output(0).get_tensor().set_names({"input1"});
    auto data2 = std::make_shared<op::v0::Parameter>(type, shape);
    data2->set_friendly_name("input2");
    data2->output(0).get_tensor().set_names({"input2"});
    std::shared_ptr<op::v0::Result> res1, res2;
    auto op1 = std::make_shared<op::v0::Abs>(data1);
    auto op2 = std::make_shared<op::v0::Abs>(data2);
    if (type == element::f32) {
        res1 = std::make_shared<op::v0::Result>(op1);
        res2 = std::make_shared<op::v0::Result>(op2);
    } else {
        auto convert1 = std::make_shared<op::v0::Convert>(op1, element::f32);
        res1 = std::make_shared<op::v0::Result>(convert1);
        auto convert2 = std::make_shared<op::v0::Convert>(op2, element::f32);
        res2 = std::make_shared<op::v0::Result>(convert2);
    }
    res1->set_friendly_name("Result1");
    res1->output(0).get_tensor().set_names({"Result1"});
    res2->set_friendly_name("Result2");
    res2->output(0).get_tensor().set_names({"Result2"});
    return std::make_shared<Model>(ResultVector{res1, res2}, ParameterVector{data1, data2});
}

inline std::shared_ptr<Model> create_preprocess_2inputs_trivial() {
    auto data1 = std::make_shared<op::v0::Parameter>(ov::element::f32, Shape{1, 3, 1, 1});
    auto data2 = std::make_shared<op::v0::Parameter>(ov::element::f32, Shape{1, 3, 1, 1});

    data1->set_friendly_name("input1");
    data1->output(0).get_tensor().set_names({"input1"});

    data2->set_friendly_name("input2");
    data2->output(0).get_tensor().set_names({"input2"});

    auto res1 = std::make_shared<op::v0::Result>(data1);
    auto res2 = std::make_shared<op::v0::Result>(data2);

    return std::make_shared<Model>(ResultVector{res1, res2}, ParameterVector{data1, data2});
}

inline std::shared_ptr<Model> mean_only() {
    using namespace ov::preprocess;
    auto function = create_preprocess_1input(ov::element::f32, Shape{1, 3, 24, 24});
    auto p = PrePostProcessor(function);
    p.input().preprocess().mean(1.1f);
    function = p.build();
    return function;
}

inline std::shared_ptr<Model> scale_only() {
    using namespace ov::preprocess;
    auto function = create_preprocess_1input(ov::element::f32, Shape{1, 3, 24, 24});
    auto p = PrePostProcessor(function);
    p.input().preprocess().scale(2.1f);
    function = p.build();
    return function;
}

inline std::shared_ptr<Model> mean_scale() {
    using namespace ov::preprocess;
    auto function = create_preprocess_1input(ov::element::f32, Shape{1, 3, 24, 24});
    auto p = PrePostProcessor(function);
    p.input().preprocess().mean(1.1f).scale(2.1f);
    function = p.build();
    return function;
}

inline std::shared_ptr<Model> scale_mean() {
    using namespace ov::preprocess;
    auto function = create_preprocess_1input(ov::element::f32, Shape{1, 3, 24, 24});
    auto p = PrePostProcessor(function);
    p.input().preprocess().scale(2.1f).mean(1.1f);
    function = p.build();
    return function;
}

inline std::shared_ptr<Model> mean_vector() {
    using namespace ov::preprocess;
    auto function = create_preprocess_1input(ov::element::f32, Shape{1, 3, 24, 24});
    auto p = PrePostProcessor(function);
    p.input().tensor().set_layout("NCHW");
    p.input().preprocess().mean({2.2f, 3.3f, 4.4f});
    function = p.build();
    return function;
}

inline std::shared_ptr<Model> scale_vector() {
    using namespace ov::preprocess;
    auto function = create_preprocess_1input(ov::element::f32, Shape{1, 3, 24, 24});
    auto p = PrePostProcessor(function);
    p.input().tensor().set_layout("NCHW");
    p.input().preprocess().scale({2.2f, 3.3f, 4.4f});
    function = p.build();
    return function;
}

inline std::shared_ptr<Model> convert_element_type_and_mean() {
    using namespace ov::preprocess;
    auto function = create_preprocess_1input(ov::element::u8, Shape{1, 3, 24, 24});
    auto p = PrePostProcessor(function);
    p.input().preprocess().convert_element_type(ov::element::f32).mean(0.2f).convert_element_type(ov::element::u8);
    function = p.build();
    return function;
}

inline std::shared_ptr<Model> tensor_element_type_and_mean() {
    using namespace ov::preprocess;
    auto function = create_preprocess_1input(ov::element::u8, Shape{1, 3, 12, 12});
    auto p = PrePostProcessor(function);
    p.input().tensor().set_element_type(ov::element::f32);
    p.input().preprocess().mean(0.1f).convert_element_type(ov::element::u8);
    function = p.build();
    return function;
}

inline std::shared_ptr<Model> custom_preprocessing() {
    using namespace ov::preprocess;
    auto function = create_preprocess_1input(ov::element::i32, Shape{3, 4, 10, 20});
    auto p = PrePostProcessor(function);
    p.input().preprocess().custom([](const Output<Node>& node) {
        auto abs = std::make_shared<op::v0::Abs>(node);
        abs->set_friendly_name(node.get_node_shared_ptr()->get_friendly_name() + "/abs");
        return abs;
    });
    function = p.build();
    return function;
}

inline std::shared_ptr<Model> multiple_ops() {
    using namespace ov::preprocess;
    auto function = create_preprocess_1input(ov::element::u8, Shape{1, 3, 3, 3});
    auto p = PrePostProcessor(function);
    auto p1 = std::move(p);
    p = std::move(p1);
    p.input().tensor().set_element_type(ov::element::f32).set_layout("?CHW");
    p.input()
        .preprocess()
        .mean(1.f)
        .scale(2.f)
        .mean({1.1f, 2.2f, 3.3f})
        .scale({2.f, 3.f, 4.f})
        .custom([](const Output<Node>& node) {
            auto abs = std::make_shared<op::v0::Abs>(node);
            abs->set_friendly_name(node.get_node_shared_ptr()->get_friendly_name() + "/abs");
            return abs;
        });
    p.input().preprocess().convert_element_type(ov::element::u8);
    function = p.build();
    return function;
}

inline std::shared_ptr<Model> two_inputs_basic() {
    using namespace ov::preprocess;
    auto function = create_preprocess_2inputs(ov::element::f32, Shape{1, 3, 1, 1});
    auto p = PrePostProcessor(function);
    p.input(1).preprocess().mean(1.f).scale(2.0f);
    function = p.build();
    return function;
}

inline std::shared_ptr<Model> two_inputs_trivial() {
    using namespace ov::preprocess;
    auto function = create_preprocess_2inputs_trivial();
    auto p = PrePostProcessor(function);
    p.input(1).preprocess().mean(1.f).scale(2.0f);
    function = p.build();
    return function;
}

inline std::shared_ptr<Model> reuse_network_layout() {
    using namespace ov::preprocess;
    auto function = create_preprocess_1input(ov::element::f32, PartialShape{4, 3, 2, 1});
    function->get_parameters().front()->set_layout("NC??");
    auto p = PrePostProcessor(function);
    p.input().preprocess().mean({1.1f, 2.2f, 3.3f}).scale({2.f, 3.f, 4.f});
    function = p.build();
    return function;
}

inline std::shared_ptr<Model> tensor_layout() {
    using namespace ov::preprocess;
    auto function = create_preprocess_1input(ov::element::f32, PartialShape{4, 3, 2, 1});
    function->get_parameters().front()->set_layout("NC??");
    auto p = PrePostProcessor(function);
    p.input().tensor().set_layout("NC??");
    p.input().preprocess().mean({1.1f, 2.2f, 3.3f}).scale({2.f, 3.f, 4.f});
    function = p.build();
    return function;
}

inline std::shared_ptr<Model> resize_linear() {
    using namespace ov::preprocess;
    auto function = create_preprocess_1input(ov::element::f32, PartialShape{1, 3, 10, 10});
    auto p = PrePostProcessor(function);
    p.input().tensor().set_spatial_static_shape(20, 20);
    p.input().preprocess().resize(ResizeAlgorithm::RESIZE_LINEAR);
    p.input().model().set_layout("NCHW");
    function = p.build();
    return function;
}

inline std::shared_ptr<Model> resize_nearest() {
    using namespace ov::preprocess;
    auto function = create_preprocess_1input(ov::element::f32, PartialShape{1, 3, 10, 10});
    auto p = PrePostProcessor(function);
    p.input().tensor().set_spatial_static_shape(20, 20);
    p.input().preprocess().resize(ResizeAlgorithm::RESIZE_NEAREST);
    p.input().model().set_layout("NCHW");
    function = p.build();
    return function;
}

inline std::shared_ptr<Model> resize_linear_nhwc() {
    using namespace ov::preprocess;
    auto function = create_preprocess_1input(ov::element::f32, PartialShape{1, 10, 10, 3});
    auto p = PrePostProcessor(function);
    p.input().tensor().set_spatial_static_shape(20, 20);
    p.input().preprocess().resize(ResizeAlgorithm::RESIZE_LINEAR);
    p.input().model().set_layout("NHWC");
    function = p.build();
    return function;
}

inline std::shared_ptr<Model> resize_cubic() {
    using namespace ov::preprocess;
    auto function = create_preprocess_1input(ov::element::f32, PartialShape{1, 3, 20, 20});
    auto p = PrePostProcessor(function);
    p.input().tensor().set_spatial_static_shape(10, 10);
    p.input().preprocess().resize(ResizeAlgorithm::RESIZE_CUBIC);
    p.input().model().set_layout("NCHW");
    function = p.build();
    return function;
}

inline std::shared_ptr<Model> resize_and_convert_layout() {
    using namespace ov::preprocess;
    auto function = create_preprocess_1input(ov::element::f32, PartialShape{1, 30, 20, 3});
    auto p = PrePostProcessor(function);
    p.input().tensor().set_layout("NHWC").set_spatial_static_shape(40, 30);
    p.input().preprocess().convert_layout().resize(ResizeAlgorithm::RESIZE_LINEAR);
    p.input().model().set_layout("NCHW");
    function = p.build();
    return function;
}

inline std::shared_ptr<Model> convert_layout_by_dims() {
    using namespace ov::preprocess;
    auto function = create_preprocess_1input(ov::element::f32, PartialShape{1, 30, 20, 3});
    auto p = PrePostProcessor(function);
    p.input().preprocess().convert_layout({0, 3, 1, 2});
    function = p.build();
    return function;
}

inline std::shared_ptr<Model> convert_layout_hwc_to_nchw() {
    using namespace ov::preprocess;
    auto function = create_preprocess_1input(ov::element::f32, PartialShape{1, 3, 30, 20});
    auto p = PrePostProcessor(function);
    p.input().tensor().set_layout("HWC").set_element_type(ov::element::u8);
    p.input().model().set_layout("NCHW");
    function = p.build();
    return function;
}

inline std::shared_ptr<Model> resize_and_convert_layout_i8() {
    using namespace ov::preprocess;
    auto function = create_preprocess_1input(ov::element::i8, PartialShape{1, 30, 20, 3});
    auto p = PrePostProcessor(function);
    p.input().tensor().set_layout("NHWC").set_spatial_static_shape(40, 30);
    p.input().preprocess().convert_layout().resize(ResizeAlgorithm::RESIZE_LINEAR);
    p.input().model().set_layout("NCHW");
    function = p.build();
    return function;
}

inline std::shared_ptr<Model> cvt_color_nv12_to_rgb_single_plane() {
    using namespace ov::preprocess;
    auto function = create_preprocess_1input(ov::element::f32, PartialShape{1, 20, 20, 3});
    auto p = PrePostProcessor(function);
    p.input().tensor().set_color_format(ColorFormat::NV12_SINGLE_PLANE);
    p.input().preprocess().convert_color(ColorFormat::RGB);
    function = p.build();
    return function;
}

inline std::shared_ptr<Model> cvt_color_nv12_to_bgr_two_planes() {
    using namespace ov::preprocess;
    auto function = create_preprocess_1input(ov::element::f32, PartialShape{1, 20, 20, 3});
    auto p = PrePostProcessor(function);
    p.input().tensor().set_color_format(ColorFormat::NV12_TWO_PLANES);
    p.input().preprocess().convert_color(ColorFormat::BGR);
    function = p.build();
    return function;
}

inline std::shared_ptr<Model> cvt_color_nv12_cvt_layout_resize() {
    using namespace ov::preprocess;
    auto function = create_preprocess_1input(ov::element::f32, PartialShape{1, 3, 10, 10});
    auto p = PrePostProcessor(function);
    p.input()
        .tensor()
        .set_color_format(ColorFormat::NV12_TWO_PLANES)
        .set_element_type(ov::element::u8)
        .set_spatial_static_shape(20, 20);
    p.input()
        .preprocess()
        .convert_color(ColorFormat::RGB)
        .convert_layout()
        .convert_element_type(ov::element::f32)
        .resize(ResizeAlgorithm::RESIZE_LINEAR);
    p.input().model().set_layout("NCHW");
    function = p.build();
    return function;
}

inline std::shared_ptr<Model> cvt_color_i420_to_rgb_single_plane() {
    using namespace ov::preprocess;
    auto function = create_preprocess_1input(ov::element::f32, PartialShape{1, 20, 20, 3});
    auto p = PrePostProcessor(function);
    p.input().tensor().set_color_format(ColorFormat::I420_SINGLE_PLANE);
    p.input().preprocess().convert_color(ColorFormat::RGB);
    return p.build();
}

inline std::shared_ptr<Model> cvt_color_i420_to_bgr_three_planes() {
    using namespace ov::preprocess;
    auto function = create_preprocess_1input(ov::element::f32, PartialShape{1, 20, 20, 3});
    auto p = PrePostProcessor(function);
    p.input().tensor().set_color_format(ColorFormat::I420_THREE_PLANES);
    p.input().preprocess().convert_color(ColorFormat::BGR);
    return p.build();
}

inline std::shared_ptr<Model> cvt_color_bgrx_to_bgr() {
    using namespace ov::preprocess;
    auto function = create_preprocess_2inputs(ov::element::f32, PartialShape{1, 160, 160, 3});
    auto p = PrePostProcessor(function);
    p.input(0).tensor().set_color_format(ColorFormat::BGRX);
    p.input(0).preprocess().convert_color(ColorFormat::BGR);
    p.input(1).tensor().set_color_format(ColorFormat::RGBX);
    p.input(1).preprocess().convert_color(ColorFormat::BGR);
    return p.build();
}

inline std::shared_ptr<Model> resize_dynamic() {
    using namespace ov::preprocess;
    auto function = create_preprocess_1input(ov::element::f32, PartialShape{1, 3, 20, 20});
    auto p = PrePostProcessor(function);
    p.input().tensor().set_spatial_dynamic_shape();
    p.input().preprocess().resize(ResizeAlgorithm::RESIZE_LINEAR);
    p.input().model().set_layout("NCHW");
    function = p.build();
    return function;
}

inline std::shared_ptr<Model> crop_basic() {
    using namespace ov::preprocess;
    auto function = create_preprocess_1input(ov::element::f32, PartialShape{1, 3, 10, 10});
    auto p = PrePostProcessor(function);
    p.input().tensor().set_shape({1, 3, 40, 40});
    p.input().preprocess().crop({0, 0, 5, 10}, {1, 3, 15, 20});
    function = p.build();
    return function;
}

inline std::shared_ptr<Model> crop_negative() {
    using namespace ov::preprocess;
    auto function = create_preprocess_1input(ov::element::f32, PartialShape{1, 3, -1, -1});
    auto p = PrePostProcessor(function);
    p.input().tensor().set_shape({1, 3, 40, 40});
    p.input().preprocess().crop({0, 0, 5, 10}, {1, 3, -5, -5});
    function = p.build();
    return function;
}

inline std::shared_ptr<Model> crop_dynamic() {
    using namespace ov::preprocess;
    auto function = create_preprocess_1input(ov::element::f32, PartialShape{1, 3, -1, -1});
    auto p = PrePostProcessor(function);
    p.input().tensor().set_shape(PartialShape{1, 3, -1, -1});
    p.input().preprocess().crop({0, 0, 50, 50}, {1, 3, 60, 150});
    function = p.build();
    return function;
}

inline std::shared_ptr<Model> pad_constant() {
    using namespace ov::preprocess;
    auto function = create_preprocess_1input(ov::element::f32, PartialShape{1, 3, 10, 10});
    auto p = PrePostProcessor(function);
    p.input().tensor().set_shape({1, 3, 9, 5});
    p.input().preprocess().pad({0, 0, 2, 2}, {0, 0, -1, 3}, 0, PaddingMode::CONSTANT);
    function = p.build();
    return function;
}

inline std::shared_ptr<Model> pad_edge() {
    using namespace ov::preprocess;
    auto function = create_preprocess_1input(ov::element::f32, PartialShape{1, 3, 10, 10});
    auto p = PrePostProcessor(function);
    p.input().tensor().set_shape({1, 3, 9, 5});
    p.input().preprocess().pad({0, 0, 2, 2}, {0, 0, -1, 3}, 0, PaddingMode::EDGE);
    function = p.build();
    return function;
}

inline std::vector<preprocess_func> generic_preprocess_functions() {
    return std::vector<preprocess_func>{
        preprocess_func(mean_only, "mean_only", 0.01f),
        preprocess_func(scale_only, "scale_only", 0.01f),
        preprocess_func(mean_scale, "mean_scale", 0.01f),
        preprocess_func(scale_mean, "scale_mean", 0.01f),
        preprocess_func(mean_vector, "mean_vector", 0.01f),
        preprocess_func(scale_vector, "scale_vector", 0.01f),
        preprocess_func(convert_element_type_and_mean, "convert_element_type_and_mean", 0.01f),
        preprocess_func(tensor_element_type_and_mean, "tensor_element_type_and_mean", 0.01f),
        preprocess_func(custom_preprocessing, "custom_preprocessing", 0.01f),
        preprocess_func(multiple_ops, "multiple_ops", 0.01f),
        preprocess_func(two_inputs_basic, "two_inputs_basic", 0.01f),
        preprocess_func(two_inputs_trivial, "two_inputs_trivial", 0.01f),
        preprocess_func(reuse_network_layout, "reuse_network_layout", 0.01f),
        preprocess_func(tensor_layout, "tensor_layout", 0.01f),
        preprocess_func(resize_linear, "resize_linear", 0.01f),
        preprocess_func(resize_nearest, "resize_nearest", 0.01f),
        preprocess_func(resize_linear_nhwc, "resize_linear_nhwc", 0.01f),
        preprocess_func(resize_cubic, "resize_cubic", 0.01f),
        preprocess_func(resize_dynamic, "resize_dynamic", 0.01f, {Shape{1, 3, 223, 323}}),
        preprocess_func(crop_basic, "crop_basic", 0.000001f),
        preprocess_func(crop_negative, "crop_negative", 0.000001f),
        preprocess_func(crop_dynamic, "crop_dynamic", 0.000001f, {Shape{1, 3, 123, 123}}),
        preprocess_func(convert_layout_by_dims, "convert_layout_by_dims", 0.01f),
        preprocess_func(convert_layout_hwc_to_nchw, "convert_layout_hwc_to_nchw", 0.01f),
        preprocess_func(resize_and_convert_layout, "resize_and_convert_layout", 0.01f),
        preprocess_func(resize_and_convert_layout_i8, "resize_and_convert_layout_i8", 0.01f),
        preprocess_func(cvt_color_nv12_to_rgb_single_plane, "cvt_color_nv12_to_rgb_single_plane", 1.f),
        preprocess_func(cvt_color_nv12_to_bgr_two_planes, "cvt_color_nv12_to_bgr_two_planes", 1.f),
        preprocess_func(cvt_color_nv12_cvt_layout_resize, "cvt_color_nv12_cvt_layout_resize", 1.f),
        preprocess_func(cvt_color_i420_to_rgb_single_plane, "cvt_color_i420_to_rgb_single_plane", 1.f),
        preprocess_func(cvt_color_i420_to_bgr_three_planes, "cvt_color_i420_to_bgr_three_planes", 1.f),
        preprocess_func(cvt_color_bgrx_to_bgr, "cvt_color_bgrx_to_bgr", 0.01f),
        preprocess_func(pad_constant, "pad_constant", 0.01f),
        preprocess_func(pad_edge, "pad_edge", 0.01f)};
}

inline std::shared_ptr<Model> cvt_color_rgb_to_bgr() {
    using namespace ov::preprocess;
    auto function = create_preprocess_1input(ov::element::f32, PartialShape{1, 20, 30, 3});
    auto p = PrePostProcessor(function);
    p.output().model().set_layout("NHWC").set_color_format(ColorFormat::RGB);
    p.output().postprocess().convert_color(ColorFormat::BGR);
    function = p.build();
    return function;
}

inline std::shared_ptr<Model> cvt_color_bgr_to_rgb() {
    using namespace ov::preprocess;
    auto function = create_preprocess_1input(ov::element::f32, PartialShape{1, 20, 30, 3});
    auto p = PrePostProcessor(function);
    p.output().model().set_layout("NHWC").set_color_format(ColorFormat::BGR);
    p.output().postprocess().convert_color(ColorFormat::RGB);
    function = p.build();
    return function;
}

inline std::vector<postprocess_func> generic_postprocess_functions() {
    return std::vector<postprocess_func>{
        postprocess_func(cvt_color_rgb_to_bgr, "convert_color_rgb_to_bgr", 1e-5f),
        postprocess_func(cvt_color_bgr_to_rgb, "convert_color_bgr_to_rgb", 1e-5f),
    };
}

}  // namespace preprocess
}  // namespace builder
}  // namespace ov
