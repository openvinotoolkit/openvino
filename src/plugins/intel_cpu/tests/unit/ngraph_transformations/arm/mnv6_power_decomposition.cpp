// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <ngraph/function.hpp>
#include <transformations/cpu_opset/arm/pass/mvn6_power_decomposition.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>
#include <ngraph/pass/manager.hpp>
#include <ov_ops/type_relaxed.hpp>

using namespace testing;
using namespace ov::intel_cpu;

TEST(TransformationTests, MVN6PowerDecompositionWithSquarePower) {
    std::shared_ptr<ov::Model> m(nullptr);
    auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{20});
    auto k = ov::opset8::Constant::create(ov::element::f32, ov::Shape{1}, {2});
    auto b = std::make_shared<ov::opset8::Power>(input, k);

    m = std::make_shared<ov::Model>(ov::OutputVector{b}, ov::ParameterVector{input});
    {
        ov::pass::Manager manager;
        manager.register_pass<MVN6PowerDecomposition>();
        manager.run_passes(m);
    }

    auto list_of_ops = m->get_ordered_ops();
    ASSERT_STRCASEEQ(list_of_ops[0]->get_type_info().name, "Parameter");
    ASSERT_STRCASEEQ(list_of_ops[1]->get_type_info().name, "Multiply");
    ASSERT_STRCASEEQ(list_of_ops[2]->get_type_info().name, "Result");
}

TEST(TransformationTests, MVN6PowerDecompositionWithoutSquarePower) {
    std::shared_ptr<ov::Model> m(nullptr);
    auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{20});
    auto k = ov::opset8::Constant::create(ov::element::f32, ov::Shape{1}, {3});
    auto b = std::make_shared<ov::opset8::Power>(input, k);

    m = std::make_shared<ov::Model>(ov::OutputVector{b}, ov::ParameterVector{input});
    {
        ov::pass::Manager manager;
        manager.register_pass<MVN6PowerDecomposition>();
        manager.run_passes(m);
    }

    auto list_of_ops = m->get_ordered_ops();
    ASSERT_STRCASEEQ(list_of_ops[0]->get_type_info().name, "Parameter");
    ASSERT_STRCASEEQ(list_of_ops[1]->get_type_info().name, "Constant");
    ASSERT_STRCASEEQ(list_of_ops[2]->get_type_info().name, "Power");
    ASSERT_STRCASEEQ(list_of_ops[3]->get_type_info().name, "Result");
}