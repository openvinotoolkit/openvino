// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include <ngraph/function.hpp>
#include <ngraph/dimension.hpp>
#include <ngraph/pass/pass.hpp>

#include "test_common.hpp"

#define DYN ngraph::Dimension::dynamic()

using TransformationTests = CommonTestUtils::TestsCommon;

std::pair<bool, std::string> compare_functions(const std::shared_ptr<ngraph::Function> & f1, const std::shared_ptr<ngraph::Function> & f2);

void check_rt_info(const std::shared_ptr<ngraph::Function> & f);


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
