// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/preprocess/pre_post_process.hpp"

namespace ov {
namespace builder {
namespace preprocess {

using preprocess_func = std::tuple<std::function<std::shared_ptr<Function>()>, std::string>;

inline std::vector<preprocess_func> generic_preprocess_functions();


/// -------- Functions ---------------

inline std::shared_ptr<Function> create_preprocess_1input(element::Type type,
                                                          const PartialShape& shape) {
    auto data1 = std::make_shared<op::v0::Parameter>(type, shape);
    data1->set_friendly_name("input1");
    data1->output(0).get_tensor().set_names({"input1"});
    auto res = std::make_shared<op::v0::Result>(data1);
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
    auto res1 = std::make_shared<op::v0::Result>(data1);
    res1->set_friendly_name("Result1");
    auto res2 = std::make_shared<op::v0::Result>(data2);
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
    auto function = create_preprocess_1input(element::i8, Shape{1, 3, 24, 24});
    function = PrePostProcessor()
            .input(InputInfo()
                           .preprocess(PreProcessSteps()
                                               .convert_element_type(element::f32)
                                               .mean(0.2f)
                                               .convert_element_type(element::i8)))
            .build(function);
    return function;
}

inline std::shared_ptr<Function> tensor_element_type_and_mean() {
    using namespace ov::preprocess;
    auto function = create_preprocess_1input(element::i8, Shape{1, 3, 12, 12});
    function = PrePostProcessor()
            .input(InputInfo()
                           .tensor(InputTensorInfo().set_element_type(element::f32))
                           .preprocess(PreProcessSteps().mean(0.1f).convert_element_type(element::i8)))
            .build(function);
    return function;
}

inline std::shared_ptr<Function> custom_preprocessing() {
    using namespace ov::preprocess;
    auto function = create_preprocess_1input(element::i32, Shape{3, 4, 10, 20});
    function = PrePostProcessor()
            .input(InputInfo().preprocess(PreProcessSteps().custom([](const std::shared_ptr<Node>& node) {
                auto abs = std::make_shared<op::v0::Abs>(node);
                abs->set_friendly_name(node->get_friendly_name() + "/abs");
                return abs;
            })))
            .build(function);
    return function;
}

inline std::shared_ptr<Function> lvalues_multiple_ops() {
    using namespace ov::preprocess;
    auto function = create_preprocess_1input(element::i8, Shape{1, 3, 3, 3});
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
        preprocessSteps.custom([](const std::shared_ptr<Node>& node) {
            auto abs = std::make_shared<op::v0::Abs>(node);
            abs->set_friendly_name(node->get_friendly_name() + "/abs");
            return abs;
        });
        auto& same = preprocessSteps.convert_element_type(element::i8);
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

inline std::vector<preprocess_func> generic_preprocess_functions() {
    return std::vector<preprocess_func> {
            preprocess_func(mean_only, "mean_only"),
            preprocess_func(scale_only, "scale_only"),
            preprocess_func(mean_scale, "mean_scale"),
            preprocess_func(scale_mean, "scale_mean"),
            preprocess_func(mean_vector, "mean_vector"),
            preprocess_func(scale_vector, "scale_vector"),
            preprocess_func(convert_element_type_and_mean, "convert_element_type_and_mean"),
            preprocess_func(tensor_element_type_and_mean, "tensor_element_type_and_mean"),
            preprocess_func(custom_preprocessing, "custom_preprocessing"),
            preprocess_func(lvalues_multiple_ops, "lvalues_multiple_ops"),
            preprocess_func(two_inputs_basic, "two_inputs_basic"),
            preprocess_func(reuse_network_layout, "reuse_network_layout"),
            preprocess_func(tensor_layout, "tensor_layout"),
    };
}

}  // namespace preprocess
}  // namespace builder
}  // namespace ov
