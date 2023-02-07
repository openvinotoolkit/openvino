// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_test_utils.hpp"

TransformationTestsF::TransformationTestsF()
    : model(function),
      model_ref(function_ref),
      comparator(FunctionsComparator::no_default()) {
    m_unh = std::make_shared<ngraph::pass::UniqueNamesHolder>();
    comparator.enable(FunctionsComparator::CmpValues::NODES);
    comparator.enable(FunctionsComparator::CmpValues::PRECISIONS);
    comparator.enable(FunctionsComparator::CmpValues::RUNTIME_KEYS);
    comparator.enable(FunctionsComparator::CmpValues::SUBGRAPH_DESCRIPTORS);
    // TODO: enable attributes and constant values comparison by default XXX-98039
    // comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    // comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    // comparator.enable(FunctionsComparator::CmpValues::NAMES);
}

void TransformationTestsF::SetUp() {
    manager.register_pass<ngraph::pass::InitUniqueNames>(m_unh);
    manager.register_pass<ngraph::pass::InitNodeInfo>();
}

void TransformationTestsF::TearDown() {
    OPENVINO_ASSERT(function != nullptr, "Test Model is not initialized.");
    auto cloned_function = ngraph::clone_function(*function);
    if (!function_ref) {
        function_ref = cloned_function;
    }

    manager.register_pass<ngraph::pass::CheckUniqueNames>(m_unh, m_soft_names_comparison);
    manager.run_passes(function);
    if (!m_disable_rt_info_check) {
    ASSERT_NO_THROW(check_rt_info(function));
    }

    if (comparator.should_compare(FunctionsComparator::ACCURACY)) {
        auto acc_comparator = FunctionsComparator::no_default();
        acc_comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
        auto res = acc_comparator.compare(function, cloned_function);
        ASSERT_TRUE(res.valid) << res.message;
        comparator.disable(FunctionsComparator::CmpValues::ACCURACY);
    }
    auto res = comparator.compare(function, function_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

void TransformationTestsF::disable_rt_info_check() {
    m_disable_rt_info_check = true;
}

void TransformationTestsF::enable_soft_names_comparison() {
    m_soft_names_comparison = true;
}

void init_unique_names(std::shared_ptr<ngraph::Function> f, const std::shared_ptr<ngraph::pass::UniqueNamesHolder>& unh) {
    ngraph::pass::Manager manager;
    manager.register_pass<ngraph::pass::InitUniqueNames>(unh);
    manager.run_passes(f);
}

void check_unique_names(std::shared_ptr<ngraph::Function> f, const std::shared_ptr<ngraph::pass::UniqueNamesHolder>& unh) {
    ngraph::pass::Manager manager;
    manager.register_pass<ngraph::pass::CheckUniqueNames>(unh, true);
    manager.run_passes(f);
}

std::shared_ptr<ov::opset8::Constant> create_zero_constant(const ov::element::Type_t& et, const ov::Shape& shape) {
    return ov::opset8::Constant::create(et, shape, {0});
}