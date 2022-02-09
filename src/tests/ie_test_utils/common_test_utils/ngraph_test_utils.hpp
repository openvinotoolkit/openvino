// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <limits>
#include <memory>
#include <queue>

#include <ngraph/dimension.hpp>
#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/pass/pass.hpp>
#include <ngraph/pass/manager.hpp>
#include <ngraph/opsets/opset6.hpp>
#include <ngraph/op/util/framework_node.hpp>
#include <transformations/init_node_info.hpp>
#include <openvino/core/model.hpp>

#include "ie_common.h"

#include "test_common.hpp"

#include "graph_comparator.hpp"
#include "test_tools.hpp"

#define DYN ngraph::Dimension::dynamic()

using TransformationTests = CommonTestUtils::TestsCommon;

class TransformationTestsF : public  CommonTestUtils::TestsCommon {
public:
    TransformationTestsF() : comparator(FunctionsComparator::no_default()) {
        m_unh = std::make_shared<ngraph::pass::UniqueNamesHolder>();
        comparator.enable(FunctionsComparator::CmpValues::PRECISIONS);
        comparator.enable(FunctionsComparator::CmpValues::RUNTIME_KEYS);
        // TODO: enable attributes and constant values comparison by default XXX-68694
        // comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
        // comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    }

    void SetUp() override {
        manager.register_pass<ngraph::pass::InitUniqueNames>(m_unh);
        manager.register_pass<ngraph::pass::InitNodeInfo>();
    }

    void TearDown() override {
        auto cloned_function = ngraph::clone_function(*function);
        if (!function_ref) {
            function_ref = cloned_function;
        }

        manager.register_pass<ngraph::pass::CheckUniqueNames>(m_unh, m_soft_names_comparison);
        manager.run_passes(function);
        if (!m_disable_rt_info_check) {
            ASSERT_NO_THROW(check_rt_info(function));
        }

        auto res = comparator.compare(function, function_ref);
        ASSERT_TRUE(res.valid) << res.message;

        if (m_enable_accuracy_check) {
            accuracy_check(cloned_function, function);
        }
    }

    // TODO: this is temporary solution to disable rt info checks that must be applied by default
    // first tests must be fixed then this method must be removed XXX-68696
    void disable_rt_info_check() {
        m_disable_rt_info_check = true;
    }

    void enable_soft_names_comparison() {
        m_soft_names_comparison = true;
    }

    void enable_accuracy_check() {
        m_enable_accuracy_check = true;
    }

    void accuracy_check(std::shared_ptr<ov::Model> ref_function, std::shared_ptr<ov::Model> cur_function);

    std::shared_ptr<ov::Model> function, function_ref;
    ngraph::pass::Manager manager;
    FunctionsComparator comparator;

private:
    std::shared_ptr<ngraph::pass::UniqueNamesHolder> m_unh;
    bool m_disable_rt_info_check{false};
    bool m_soft_names_comparison{true};
    bool m_enable_accuracy_check{false};
};

void init_unique_names(std::shared_ptr<ngraph::Function> f, const std::shared_ptr<ngraph::pass::UniqueNamesHolder>& unh);

void check_unique_names(std::shared_ptr<ngraph::Function> f, const std::shared_ptr<ngraph::pass::UniqueNamesHolder>& unh);
