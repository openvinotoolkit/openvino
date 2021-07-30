// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ngraph/opsets/opset7.hpp>
#include <transformations/preprocessing/subtract_mean_inputs.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/serialize.hpp>
#include <ngraph/pass/manager.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ngraph::pass;

static std::shared_ptr<ngraph::Function> create_simple_function(const ngraph::PartialShape &shape) {
    auto data1 = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32, shape);
    data1->set_friendly_name("input1");
    auto res = std::make_shared<ngraph::opset7::Result>(data1);
    res->set_friendly_name("Result");
    return std::make_shared<ngraph::Function>(ngraph::ResultVector{res}, ngraph::ParameterVector{data1});
}

static std::shared_ptr<ngraph::Function> create_function_2inputs(const ngraph::PartialShape &shape) {
    auto data1 = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32, shape);
    data1->set_friendly_name("input1");
    auto data2 = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32, shape);
    data2->set_friendly_name("input2");
    auto add = std::make_shared<ngraph::opset7::Add>(data1, data2);
    add->set_friendly_name("add");
    auto res = std::make_shared<ngraph::opset7::Result>(add);
    res->set_friendly_name("Result");
    return std::make_shared<ngraph::Function>(ngraph::ResultVector{res}, ngraph::ParameterVector{data1, data2});
}

TEST(TransformationTests, SubtractMeanInputs_test_single_value) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto c1 = ngraph::opset7::Constant::create(ngraph::element::f32,
                                                   ngraph::Shape{1},
                                                   {2.0f});
        auto c2 = ngraph::opset7::Constant::create(ngraph::element::f32,
                                                   ngraph::Shape{1},
                                                   {4.0f});
        f = create_function_2inputs(ngraph::Shape{3, 1, 2});

        Manager m;
        m.register_pass<InitNodeInfo>();
        SubtractMeanInputs::MeanMap map;
        map.insert({"input1", c1});
        map.insert({"input2", c2});
        m.register_pass<SubtractMeanInputs>(map);
        m.run_passes(f);
        ASSERT_NO_THROW(f->validate_nodes_and_infer_types());
    }

    {
        auto data1 = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32, ngraph::Shape{3, 1, 2});
        data1->set_friendly_name("input1");
        auto sub_const1 = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {2.f});
        sub_const1->set_friendly_name("input1/subtract/SubtractMean_Value");
        auto div1 = std::make_shared<ngraph::opset7::Subtract>(data1, sub_const1);
        div1->set_friendly_name("input1/subtract/SubtractMean");
        auto data2 = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32, ngraph::Shape{3, 1, 2});
        data2->set_friendly_name("input2");
        auto sub_const2 = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {4.f});
        sub_const2->set_friendly_name("input2/subtract/SubtractMean_Value");
        auto div2 = std::make_shared<ngraph::opset7::Subtract>(data2, sub_const2);
        div2->set_friendly_name("input2/subtract/SubtractMean");
        auto add = std::make_shared<ngraph::opset7::Add>(div1, div2);
        add->set_friendly_name("add");
        auto res = std::make_shared<ngraph::opset7::Result>(add);
        res->set_friendly_name("Result");

        f_ref = std::make_shared<ngraph::Function>(ngraph::ResultVector{res}, ngraph::ParameterVector{data1, data2});
    }

    const FunctionsComparator func_comparator =
            FunctionsComparator::with_default()
                    .enable(FunctionsComparator::NAMES_ALL)
                    .enable(FunctionsComparator::CONST_VALUES);
    const FunctionsComparator::Result res = func_comparator(f, f_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

TEST(TransformationTests, SubtractMeanInputs_test_same_value) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto c = ngraph::opset7::Constant::create(ngraph::element::f32,
                                                  ngraph::Shape{1},
                                                  {2.0f});
        f = create_function_2inputs(ngraph::Shape{3, 1, 2});

        Manager m;
        m.register_pass<InitNodeInfo>();
        SubtractMeanInputs::MeanMap map;
        map.insert({"input1", c});
        map.insert({"input2", c});
        m.register_pass<SubtractMeanInputs>(map);
        m.run_passes(f);
        ASSERT_NO_THROW(f->validate_nodes_and_infer_types());
    }

    {
        auto data1 = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32, ngraph::Shape{3, 1, 2});
        data1->set_friendly_name("input1");
        auto sub_const1 = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {2.f});
        sub_const1->set_friendly_name("input1/subtract/SubtractMean_Value");
        auto div1 = std::make_shared<ngraph::opset7::Subtract>(data1, sub_const1);
        div1->set_friendly_name("input1/subtract/SubtractMean");
        auto data2 = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32, ngraph::Shape{3, 1, 2});
        data2->set_friendly_name("input2");
        auto sub_const2 = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {2.f});
        sub_const2->set_friendly_name("input2/subtract/SubtractMean_Value");
        auto div2 = std::make_shared<ngraph::opset7::Subtract>(data2, sub_const2);
        div2->set_friendly_name("input2/subtract/SubtractMean");
        auto add = std::make_shared<ngraph::opset7::Add>(div1, div2);
        add->set_friendly_name("add");
        auto res = std::make_shared<ngraph::opset7::Result>(add);
        res->set_friendly_name("Result");

        f_ref = std::make_shared<ngraph::Function>(ngraph::ResultVector{res}, ngraph::ParameterVector{data1, data2});
    }

    const FunctionsComparator func_comparator =
            FunctionsComparator::with_default()
            .enable(FunctionsComparator::NAMES_ALL)
            .enable(FunctionsComparator::CONST_VALUES);
    const FunctionsComparator::Result res = func_comparator(f, f_ref);
    ASSERT_TRUE(res.valid) << res.message;

    // Verify serialization as there are 2 nodes with same friendly name
    std::stringstream xml, bin;
    Manager manager;
    manager.register_pass<ngraph::pass::Serialize>(
            xml, bin, ngraph::pass::Serialize::Version::IR_V10);
    ASSERT_NO_THROW(manager.run_passes(f));
}

TEST(TransformationTests, SubtractMeanInputs_SubtractMeanSubtractMean) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto c1 = ngraph::opset7::Constant::create(ngraph::element::f32,
                                                   ngraph::Shape{1},
                                                   {2.0f});
        auto c2 = ngraph::opset7::Constant::create(ngraph::element::f32,
                                                   ngraph::Shape{1},
                                                   {4.0f});
        f = create_simple_function(ngraph::Shape{3, 1, 2});

        Manager m;
        m.register_pass<InitNodeInfo>();
        SubtractMeanInputs::MeanMap map1, map2;
        map1.insert({"input1", c1});
        map2.insert({"input1", c2});
        m.register_pass<SubtractMeanInputs>(map1);
        m.register_pass<SubtractMeanInputs>(map2);
        m.run_passes(f);
        ASSERT_NO_THROW(f->validate_nodes_and_infer_types());
    }

    {
        auto data1 = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32, ngraph::Shape{3, 1, 2});
        auto sub_const1 = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {4.f});
        auto div1 = std::make_shared<ngraph::opset7::Subtract>(data1, sub_const1);
        auto sub_const2 = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {2.f});
        auto div2 = std::make_shared<ngraph::opset7::Subtract>(div1, sub_const2);
        auto res = std::make_shared<ngraph::opset7::Result>(div2);

        f_ref = std::make_shared<ngraph::Function>(ngraph::ResultVector{res}, ngraph::ParameterVector{data1});
    }

    const FunctionsComparator func_comparator =
            FunctionsComparator::with_default()
            .enable(FunctionsComparator::CONST_VALUES);
    const FunctionsComparator::Result res = func_comparator(f, f_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

TEST(TransformationTests, SubtractMeanInputs_1input) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    ngraph::PartialShape shape = {ngraph::Dimension::dynamic(), 3, 5, 5};
    {
        f = create_function_2inputs(shape);
        auto c2 = ngraph::opset7::Constant::create(ngraph::element::f32,
                                                   ngraph::Shape{1, 3, 1, 1},
                                                   {1.f, 2.f, 4.f});

        Manager m;
        m.register_pass<InitNodeInfo>();
        SubtractMeanInputs::MeanMap map;
        map.insert({"input2", c2});
        m.register_pass<SubtractMeanInputs>(map);
        m.run_passes(f);
        ASSERT_NO_THROW(f->validate_nodes_and_infer_types());
    }

    {
        auto data1 = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32, shape);
        data1->set_friendly_name("input1");
        auto data2 = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32, shape);
        data2->set_friendly_name("input2");
        auto sub_const2 = ngraph::opset7::Constant::create(
                ngraph::element::f32, ngraph::Shape{1, 3, 1, 1}, {1.f, 2.f, 4.f});
        sub_const2->set_friendly_name("input2/subtract/SubtractMean_Value");
        auto div2 = std::make_shared<ngraph::opset7::Subtract>(data2, sub_const2);
        div2->set_friendly_name("input2/subtract/SubtractMean");
        auto add = std::make_shared<ngraph::opset7::Add>(data1, div2);
        add->set_friendly_name("add");
        auto res = std::make_shared<ngraph::opset7::Result>(add);
        res->set_friendly_name("Result");

        f_ref = std::make_shared<ngraph::Function>(ngraph::ResultVector{res}, ngraph::ParameterVector{data1, data2});
    }

    const FunctionsComparator func_comparator =
            FunctionsComparator::with_default()
                    .enable(FunctionsComparator::NAMES_ALL)
                    .enable(FunctionsComparator::CONST_VALUES);
    const FunctionsComparator::Result res = func_comparator(f, f_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

TEST(TransformationTests, SubtractMeanInputs_bad_constant_dim) {
    std::shared_ptr<ngraph::Function> f = create_simple_function(ngraph::Shape{1, 3, 4, 5});
    auto c1 = ngraph::opset7::Constant::create(ngraph::element::f32,
                                               ngraph::Shape{1, 1, 3, 1},
                                               {1.f, 2.f, 4.f});
    ngraph::pass::Manager m;
    m.register_pass<ngraph::pass::InitNodeInfo>();
    SubtractMeanInputs::MeanMap map;
    map.insert({"input1", c1});
    m.register_pass<ngraph::pass::SubtractMeanInputs>(map);
    ASSERT_THROW(m.run_passes(f), ngraph::ngraph_error);
}

TEST(TransformationTests, SubtractMeanInputs_bad_constant_type) {
    std::shared_ptr<ngraph::Function> f = create_simple_function(ngraph::Shape{1, 3, 4, 5});
    auto c1 = ngraph::opset7::Constant::create(ngraph::element::i32,
                                               ngraph::Shape{1, 3, 1, 1},
                                               {1.f, 2.f, 4.f});
    ngraph::pass::Manager m;
    m.register_pass<ngraph::pass::InitNodeInfo>();
    SubtractMeanInputs::MeanMap map;
    map.insert({"input1", c1});
    m.register_pass<ngraph::pass::SubtractMeanInputs>(map);
    ASSERT_THROW(m.run_passes(f), ngraph::ngraph_error);
}
