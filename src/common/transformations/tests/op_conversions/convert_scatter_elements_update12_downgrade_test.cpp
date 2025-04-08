// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_scatter_elements_update12_downgrade.hpp"

#include <gtest/gtest.h>

#include <memory>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/opsets/opset12.hpp"
#include "openvino/opsets/opset3.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/utils/utils.hpp"
using namespace ov;
using namespace testing;

namespace {
using Reduction = ov::opset12::ScatterElementsUpdate::Reduction;

std::shared_ptr<ov::Model> create_v12_model(const Reduction reduction_type, const bool use_init_value) {
    const auto input = std::make_shared<ov::opset12::Parameter>(ov::element::f32, ov::Shape{1, 3, 100, 100});
    const auto indices = std::make_shared<ov::opset12::Parameter>(ov::element::i32, ov::Shape{1, 1, 5, 5});
    const auto updates = std::make_shared<ov::opset12::Parameter>(ov::element::f32, ov::Shape{1, 1, 5, 5});
    const auto axis = std::make_shared<ov::opset12::Parameter>(ov::element::i64, ov::Shape{});

    const auto seu = std::make_shared<ov::opset12::ScatterElementsUpdate>(input,
                                                                          indices,
                                                                          updates,
                                                                          axis,
                                                                          reduction_type,
                                                                          use_init_value);

    seu->set_friendly_name("seu12");

    return std::make_shared<ov::Model>(seu->outputs(), ov::ParameterVector{input, indices, updates, axis});
}

std::shared_ptr<ov::Model> create_v3_model() {
    const auto input = std::make_shared<ov::opset3::Parameter>(ov::element::f32, ov::Shape{1, 3, 100, 100});
    const auto indices = std::make_shared<ov::opset3::Parameter>(ov::element::i32, ov::Shape{1, 1, 5, 5});
    const auto updates = std::make_shared<ov::opset3::Parameter>(ov::element::f32, ov::Shape{1, 1, 5, 5});
    const auto axis = std::make_shared<ov::opset3::Parameter>(ov::element::i64, ov::Shape{});

    const auto seu = std::make_shared<ov::opset3::ScatterElementsUpdate>(input, indices, updates, axis);

    seu->set_friendly_name("seu3");

    return std::make_shared<ov::Model>(seu->outputs(), ov::ParameterVector{input, indices, updates, axis});
}

}  // namespace

TEST_F(TransformationTestsF, ConvertScatterElementsUpdate12ToScatterElementsUpdate3_no_reduction_use_init_value) {
    manager.register_pass<ov::pass::ConvertScatterElementsUpdate12ToScatterElementsUpdate3>();
    model = create_v12_model(Reduction::NONE, true);
    model_ref = create_v3_model();
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, ConvertScatterElementsUpdate12ToScatterElementsUpdate3_no_reduction) {
    manager.register_pass<ov::pass::ConvertScatterElementsUpdate12ToScatterElementsUpdate3>();
    model = create_v12_model(Reduction::NONE, false);
    model_ref = create_v3_model();
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, ConvertScatterElementsUpdate12ToScatterElementsUpdate3_reduction_use_init_value) {
    manager.register_pass<ov::pass::ConvertScatterElementsUpdate12ToScatterElementsUpdate3>();
    model = create_v12_model(Reduction::MEAN, true);
}

TEST_F(TransformationTestsF, ConvertScatterElementsUpdate12ToScatterElementsUpdate3_reduction) {
    manager.register_pass<ov::pass::ConvertScatterElementsUpdate12ToScatterElementsUpdate3>();
    model = create_v12_model(Reduction::PROD, false);
}
