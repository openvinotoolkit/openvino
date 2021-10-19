// Copyright (C) 2018-2021 Intel Corporation
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
        // TODO: enable attributes and constant values comparison by default XXX-68694
        // comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
        // comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    }

    void SetUp() override {
        manager.register_pass<ngraph::pass::InitUniqueNames>(m_unh);
        manager.register_pass<ngraph::pass::InitNodeInfo>();
    }

    void TearDown() override {
        if (!function_ref) {
            function_ref = ngraph::clone_function(*function);
        }

        manager.register_pass<ngraph::pass::CheckUniqueNames>(m_unh);
        manager.run_passes(function);
        if (!m_disable_rt_info_check) {
            ASSERT_NO_THROW(check_rt_info(function));
        }

        auto res = comparator.compare(function, function_ref);
        ASSERT_TRUE(res.valid) << res.message;
    }

    void disable_rt_info_check() {
        m_disable_rt_info_check = true;
    }

    std::shared_ptr<ngraph::Function> function, function_ref;
    ngraph::pass::Manager manager;
    FunctionsComparator comparator;

private:
    std::shared_ptr<ngraph::pass::UniqueNamesHolder> m_unh;
    bool m_disable_rt_info_check{false};
};
