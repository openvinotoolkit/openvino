// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ngraph/function.hpp>
#include <ngraph/pass/manager.hpp>
#include <ngraph/variant.hpp>

#include <snippets/snippets_isa.hpp>
#include <snippets/pass/assign_registers.hpp>

#include <transformations/init_node_info.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ngraph;

//  todo: Rewrite this test using Snippets test infrastructure. See ./include/canonicalization.hpp for example

TEST(TransformationTests, AssignRegisters) {
    std::shared_ptr<Function> f(nullptr);
    {
        auto p0 = std::make_shared<opset1::Parameter>(element::f32, Shape(1));
        auto p1 = std::make_shared<opset1::Parameter>(element::f32, Shape(1));
        auto y00 = std::make_shared<snippets::isa::Load>(p0); y00->set_friendly_name("y00");
        auto y01 = std::make_shared<snippets::isa::Load>(p1); y01->set_friendly_name("y01");
        auto y02 = std::make_shared<opset1::Multiply>(y00, y01); y02->set_friendly_name("y02");
        auto y03 = std::make_shared<snippets::isa::Store>(y02); y03->set_friendly_name("y03");

        f = std::make_shared<Function>(NodeVector{y03}, ParameterVector{p0, p1});

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<snippets::pass::AssignRegisters>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    // instead of comparing to a reference function check that registers are correctly assigned
    // and stored to runtime info
    {
        std::map<std::string, size_t> ref_registers {
            {"y00", 0},
            {"y01", 1},
            {"y02", 2}
        };

        auto total_ops = 0;
        for (auto& op : f->get_ordered_ops()) {
            auto& rt = op->get_rt_info();

            auto it_rinfo = rt.find("reginfo");
            if (it_rinfo != rt.end()) {
                auto reginfo = it_rinfo->second.as<std::vector<size_t>>();
                auto reg = reginfo[0];
                ASSERT_TRUE(ref_registers[op->get_friendly_name()] == reg);
                total_ops++;
            }
        }
        ASSERT_EQ(total_ops, ref_registers.size());
    }
}

TEST(TransformationTests, AssignRegisters2) {
    std::shared_ptr<Function> f(nullptr);
    {
        auto p0 = std::make_shared<opset1::Parameter>(ngraph::element::f32, Shape());
        auto p1 = std::make_shared<opset1::Parameter>(ngraph::element::f32, Shape());
        auto p2 = std::make_shared<opset1::Parameter>(ngraph::element::f32, Shape());
        auto p3 = std::make_shared<opset1::Parameter>(ngraph::element::f32, Shape());
        auto p4 = std::make_shared<opset1::Parameter>(ngraph::element::f32, Shape());
        auto p5 = std::make_shared<opset1::Parameter>(ngraph::element::f32, Shape());
        auto p6 = std::make_shared<opset1::Parameter>(ngraph::element::f32, Shape());
        auto p7 = std::make_shared<opset1::Parameter>(ngraph::element::f32, Shape());

        auto c0 = std::make_shared<snippets::isa::Scalar>(ngraph::element::f32, Shape(), 3.14f); c0->set_friendly_name("r00");
        auto c1 = std::make_shared<snippets::isa::Scalar>(ngraph::element::f32, Shape(), 6.6260701e-34f); c1->set_friendly_name("r01");

        auto y00 = std::make_shared<snippets::isa::Load>(p0); y00->set_friendly_name("r02");
        auto y01 = std::make_shared<snippets::isa::Load>(p1); y01->set_friendly_name("r03");
        auto y02 = std::make_shared<opset1::Multiply>(y00, c0); y02->set_friendly_name("r04");
        auto y03 = std::make_shared<opset1::Multiply>(y01, c1); y03->set_friendly_name("r05");
        auto y04 = std::make_shared<snippets::isa::Load>(p2); y04->set_friendly_name("r06");
        auto y05 = std::make_shared<snippets::isa::Load>(p3); y05->set_friendly_name("r07");
        auto y06 = std::make_shared<opset1::Add>(y02, y03); y06->set_friendly_name("r08");
        auto y07 = std::make_shared<opset1::Multiply>(y04, c0); y07->set_friendly_name("r09");
        auto y08 = std::make_shared<opset1::Multiply>(y05, c1); y08->set_friendly_name("r10");
        auto y09 = std::make_shared<snippets::isa::Load>(p4); y09->set_friendly_name("r11");
        auto y10 = std::make_shared<snippets::isa::Load>(p5); y10->set_friendly_name("r12");
        auto y11 = std::make_shared<opset1::Add>(y07, y08); y11->set_friendly_name("r13");
        auto y12 = std::make_shared<opset1::Multiply>(y09, c0); y12->set_friendly_name("r14");
        auto y13 = std::make_shared<opset1::Multiply>(y10, c1); y13->set_friendly_name("r15");
        auto y14 = std::make_shared<snippets::isa::Load>(p6); y14->set_friendly_name("r16");
        auto y15 = std::make_shared<opset1::Add>(y12, y13); y15->set_friendly_name("r17");
        auto y16 = std::make_shared<snippets::isa::Load>(p7); y16->set_friendly_name("r18");
        auto y17 = std::make_shared<opset1::Multiply>(y14, c0); y17->set_friendly_name("r19");
        auto y18 = std::make_shared<opset1::Multiply>(y16, c1); y18->set_friendly_name("r20");
        auto y19 = std::make_shared<opset1::Add>(y06, y11); y19->set_friendly_name("r21");
        auto y20 = std::make_shared<opset1::Add>(y17, y18); y20->set_friendly_name("r22");
        auto y21 = std::make_shared<opset1::Add>(y15, y19); y21->set_friendly_name("r23");
        auto y22 = std::make_shared<opset1::Add>(y20, y21); y22->set_friendly_name("r24");
        auto y23 = std::make_shared<snippets::isa::Store>(y22);

        f = std::make_shared<Function>(NodeVector{y23}, ParameterVector{p0, p1, p2, p3, p4, p5, p6, p7});

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<snippets::pass::AssignRegisters>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    // instead of comparing to a reference function check that registers are correctly assigned
    // and stored to runtime info
    {
        std::map<std::string, size_t> ref_registers {
            {"r00", 1}, {"r01", 3}, {"r02", 5}, {"r03", 5}, {"r04", 2}, {"r05", 6}, {"r06", 6}, {"r07", 6},
            {"r08", 5}, {"r09", 2}, {"r10", 1}, {"r11", 4}, {"r12", 4}, {"r13", 6}, {"r14", 2}, {"r15", 5},
            {"r16", 0}, {"r17", 4}, {"r18", 0}, {"r19", 2}, {"r20", 4}, {"r21", 1}, {"r22", 0}, {"r23", 6},
            {"r24", 1}
        };

        auto total_ops = 0;
        for (auto& op : f->get_ordered_ops()) {
            auto& rt = op->get_rt_info();
            auto it_rinfo = rt.find("reginfo");
            if (it_rinfo != rt.end()) {
                auto reginfo = it_rinfo->second.as<std::vector<size_t>>();
                auto reg = reginfo[0];
                ASSERT_TRUE(ref_registers[op->get_friendly_name()] == reg);
                total_ops++;
            }
        }
        ASSERT_EQ(total_ops, ref_registers.size());
    }
}
