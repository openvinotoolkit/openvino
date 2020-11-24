// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <queue>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/dimension.hpp>
#include <ngraph/pass/pass.hpp>

#include "test_common.hpp"

#define DYN ngraph::Dimension::dynamic()

using TransformationTests = CommonTestUtils::TestsCommon;

bool compare(const std::vector<float>& expectedValues, const std::shared_ptr<ngraph::opset1::Constant>& constant);

std::pair<bool, std::string> compare_functions(
    const std::shared_ptr<ngraph::Function>& f1,
    const std::shared_ptr<ngraph::Function>& f2,
    const bool compareConstValues = false,
    const bool compareNames = false,
    const bool compareRuntimeKeys = false,
    const bool comparePrecisions = true);

void check_rt_info(const std::shared_ptr<ngraph::Function> & f);


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

        std::shared_ptr<T> op = ngraph::as_type_ptr<T>(node);
        if (op != nullptr) {
            nodes.push_back(op);
        }

        for (size_t i = 0; i < node->inputs().size(); ++i) {
            q.push(node->get_input_node_shared_ptr(i));
        }
    }

    return nodes;
}

namespace ngraph {
namespace pass {

class InjectionPass;

} // namespace pass
} // namespace ngraph

class ngraph::pass::InjectionPass : public ngraph::pass::FunctionPass {
public:
    using injection_callback = std::function<void(std::shared_ptr<ngraph::Function>)>;

    explicit InjectionPass(injection_callback callback) : FunctionPass(), m_callback(std::move(callback)) {}

    bool run_on_function(std::shared_ptr<ngraph::Function> f) override {
        m_callback(f);
        return false;
    }

private:
    injection_callback m_callback;
};

template <typename T>
size_t count_ops_of_type(std::shared_ptr<ngraph::Function> f) {
    size_t count = 0;
    for (auto op : f->get_ops()) {
        if (ngraph::is_type<T>(op)) {
            count++;
        }
    }

    return count;
}

class TestOpMultiOut : public ngraph::op::Op {
public:
    NGRAPH_RTTI_DECLARATION;
    TestOpMultiOut() = default;

    TestOpMultiOut(const ngraph::Output<Node>& output_1, const ngraph::Output<Node>& output_2)
        : Op({output_1, output_2}) {
        validate_and_infer_types();
    }
    void validate_and_infer_types() override {
        set_output_size(2);
        set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
        set_output_type(1, get_input_element_type(1), get_input_partial_shape(1));
    }

    std::shared_ptr<Node>
        clone_with_new_inputs(const ngraph::OutputVector& new_args) const override {
        return std::make_shared<TestOpMultiOut>(new_args.at(0), new_args.at(1));
    }
};
