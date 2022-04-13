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

#define DYN ngraph::Dimension::dynamic()

using TransformationTests = CommonTestUtils::TestsCommon;

class TransformationTestsF : public  CommonTestUtils::TestsCommon {
public:
    TransformationTestsF();

    void SetUp() override;

    void TearDown() override;

    // TODO: this is temporary solution to disable rt info checks that must be applied by default
    // first tests must be fixed then this method must be removed XXX-68696
    void disable_rt_info_check();

    void enable_soft_names_comparison();

    std::shared_ptr<ov::Model> function, function_ref;
    ngraph::pass::Manager manager;
    FunctionsComparator comparator;

private:
    std::shared_ptr<ngraph::pass::UniqueNamesHolder> m_unh;
    bool m_disable_rt_info_check{false};
    bool m_soft_names_comparison{true};
};

void init_unique_names(std::shared_ptr<ngraph::Function> f, const std::shared_ptr<ngraph::pass::UniqueNamesHolder>& unh);

void check_unique_names(std::shared_ptr<ngraph::Function> f, const std::shared_ptr<ngraph::pass::UniqueNamesHolder>& unh);

template <typename T>
size_t count_ops_of_type(const std::shared_ptr<ngraph::Function>& f) {
    size_t count = 0;
    for (auto op : f->get_ops()) {
        if (ngraph::is_type<T>(op)) {
            count++;
        }
    }

    return count;
}
