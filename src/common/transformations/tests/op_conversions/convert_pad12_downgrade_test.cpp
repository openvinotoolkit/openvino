// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_pad12_downgrade.hpp"

#include <gtest/gtest.h>

#include <memory>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset12.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/utils/utils.hpp"
using namespace ov;
using namespace testing;

namespace {
std::shared_ptr<ov::Model> create_v12_model(const ov::op::PadMode pad_mode, const int16_t pad_v = -1) {
    const auto input = std::make_shared<ov::opset12::Parameter>(ov::element::i16, ov::Shape{1, 3, 100, 100});
    const auto pads_begin =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 2, 1, 0});
    const auto pads_end =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 1, 1, 0});

    std::shared_ptr<ov::opset12::Pad> pad;
    if (pad_v != -1) {
        const auto pad_value =
            std::make_shared<ov::op::v0::Constant>(ov::element::i16, ov::Shape{}, std::vector<int16_t>{pad_v});
        pad = std::make_shared<ov::opset12::Pad>(input, pads_begin, pads_end, pad_value, pad_mode);
    } else {
        pad = std::make_shared<ov::opset12::Pad>(input, pads_begin, pads_end, pad_mode);
    }
    pad->set_friendly_name("pad12");

    return std::make_shared<ov::Model>(pad->outputs(), ov::ParameterVector{input});
}

std::shared_ptr<ov::Model> create_v1_model(const ov::op::PadMode pad_mode, const int16_t pad_v) {
    const auto input = std::make_shared<ov::opset1::Parameter>(ov::element::i16, ov::Shape{1, 3, 100, 100});
    const auto pads_begin =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 2, 1, 0});
    const auto pads_end =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 1, 1, 0});
    const auto pad_value =
        std::make_shared<ov::op::v0::Constant>(ov::element::i16, ov::Shape{}, std::vector<int16_t>{pad_v});

    const auto pad = std::make_shared<ov::opset1::Pad>(input, pads_begin, pads_end, pad_value, pad_mode);
    pad->set_friendly_name("pad1");

    return std::make_shared<ov::Model>(pad->outputs(), ov::ParameterVector{input});
}

}  // namespace

TEST_F(TransformationTestsF, ConvertPad12ToPad1) {
    manager.register_pass<ov::pass::ConvertPad12ToPad1>();
    model = create_v12_model(ov::op::PadMode::CONSTANT);
    model_ref = create_v1_model(ov::op::PadMode::CONSTANT, 0);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, ConvertPad12ToPad1_explicit_pad_value) {
    manager.register_pass<ov::pass::ConvertPad12ToPad1>();
    model = create_v12_model(ov::op::PadMode::CONSTANT, 5);
    model_ref = create_v1_model(ov::op::PadMode::CONSTANT, 5);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, ConvertPad12ToPad1_symmetric) {
    manager.register_pass<ov::pass::ConvertPad12ToPad1>();
    model = create_v12_model(ov::op::PadMode::SYMMETRIC);
    model_ref = create_v1_model(ov::op::PadMode::SYMMETRIC, 0);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, ConvertPad12ToPad1_symmetric_explicit_pad_value) {
    manager.register_pass<ov::pass::ConvertPad12ToPad1>();
    model = create_v12_model(ov::op::PadMode::SYMMETRIC, 5);
    model_ref = create_v1_model(ov::op::PadMode::SYMMETRIC, 5);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}
