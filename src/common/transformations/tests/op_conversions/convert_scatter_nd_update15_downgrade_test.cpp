// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_scatter_nd_update15_downgrade.hpp"

#include <gtest/gtest.h>

#include <memory>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/opsets/opset15.hpp"
#include "openvino/opsets/opset4.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/utils/utils.hpp"
using namespace ov;
using namespace testing;

namespace {
using Reduction = ov::opset15::ScatterNDUpdate::Reduction;

std::shared_ptr<ov::Model> create_v15_model(const Reduction reduction_type) {
    const auto data = std::make_shared<ov::opset15::Parameter>(ov::element::f32, ov::Shape{1000, 256, 10, 15});
    const auto indices = std::make_shared<ov::opset15::Parameter>(ov::element::i32, ov::Shape{25, 125, 3});
    const auto updates = std::make_shared<ov::opset15::Parameter>(ov::element::f32, ov::Shape{25, 125, 15});
    const auto scatter_nd = std::make_shared<ov::opset15::ScatterNDUpdate>(data, indices, updates, reduction_type);
    scatter_nd->set_friendly_name("scatter_nd15");
    return std::make_shared<ov::Model>(scatter_nd->outputs(), ov::ParameterVector{data, indices, updates});
}

std::shared_ptr<ov::Model> create_v3_model() {
    const auto data = std::make_shared<ov::opset15::Parameter>(ov::element::f32, ov::Shape{1000, 256, 10, 15});
    const auto indices = std::make_shared<ov::opset15::Parameter>(ov::element::i32, ov::Shape{25, 125, 3});
    const auto updates = std::make_shared<ov::opset15::Parameter>(ov::element::f32, ov::Shape{25, 125, 15});
    const auto scatter_nd = std::make_shared<ov::opset4::ScatterNDUpdate>(data, indices, updates);
    scatter_nd->set_friendly_name("scatter_nd15");

    return std::make_shared<ov::Model>(scatter_nd->outputs(), ov::ParameterVector{data, indices, updates});
}

}  // namespace

TEST_F(TransformationTestsF, ConvertScatterNDUpdate15ToScatterNDUpdate3_no_reduction) {
    manager.register_pass<ov::pass::ConvertScatterNDUpdate15ToScatterNDUpdate3>();
    model = create_v15_model(Reduction::NONE);
    model_ref = create_v3_model();
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, ConvertScatterNDUpdate15ToScatterNDUpdate3_reduction) {
    manager.register_pass<ov::pass::ConvertScatterNDUpdate15ToScatterNDUpdate3>();
    model = create_v15_model(Reduction::PROD);
}
