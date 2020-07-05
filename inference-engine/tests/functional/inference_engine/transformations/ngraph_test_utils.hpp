// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <queue>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/dimension.hpp>

#include "common_test_utils/test_common.hpp"

#define DYN ngraph::Dimension::dynamic()

using TransformationTests = CommonTestUtils::TestsCommon;

bool compare(const std::vector<float>& expectedValues, const std::shared_ptr<ngraph::opset1::Constant>& constant);

std::pair<bool, std::string> compare_functions(
    const std::shared_ptr<ngraph::Function>& f1,
    const std::shared_ptr<ngraph::Function>& f2,
    const bool compareConstValues = false);

void check_rt_info(const std::shared_ptr<ngraph::Function> & f);

void visualize_function(std::shared_ptr<ngraph::Function> f, const std::string & file_name);

template<typename T>
std::vector<std::shared_ptr<T>> get(const std::shared_ptr<ngraph::Function>& f) {
    std::vector<std::shared_ptr<T>> nodes;

    std::queue<std::shared_ptr<ngraph::Node>> q;
    for (const auto result : f->get_results()) {
        q.push(result);
    }

    while (!q.empty()) {
        auto node = q.front();
        q.pop();

        std::shared_ptr<T> op = as_type_ptr<T>(node);
        if (op != nullptr) {
            nodes.push_back(op);
        }

        for (size_t i = 0; i < node->inputs().size(); ++i) {
            q.push(node->get_input_node_shared_ptr(i));
        }
    }

    return nodes;
}
