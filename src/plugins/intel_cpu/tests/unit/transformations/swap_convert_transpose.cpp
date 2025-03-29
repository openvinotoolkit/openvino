// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include <transformations/cpu_opset/common/pass/swap_convert_transpose.hpp>
#include <transformations/init_node_info.hpp>

#include "openvino/opsets/opset1.hpp"

using namespace testing;

class SwapConvertTransposeTest: public TransformationTestsF {
public:
    SwapConvertTransposeTest() : TransformationTestsF() {
        comparator.enable(FunctionsComparator::CmpValues::NAMES);
    }
};

TEST_F(SwapConvertTransposeTest, SwapConvertTranspose) {
    const ov::Shape shape{1, 224, 224, 3};
    const std::vector<int64_t> input_order = {0, 3, 1, 2};
    const ov::element::Type in_type = ov::element::u8;
    const ov::element::Type out_type = ov::element::f32;
    const std::string transpose_name = "Transpose";

    {
        auto input = std::make_shared<ov::op::v0::Parameter>(in_type, shape);

        auto convert = std::make_shared<ov::op::v0::Convert>(input, out_type);

        auto transpose_const = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{input_order.size()}, input_order);
        auto transpose = std::make_shared<ov::op::v1::Transpose>(convert, transpose_const);
        transpose->set_friendly_name(transpose_name);

        model = std::make_shared<ov::Model>(ov::NodeVector{transpose}, ov::ParameterVector{input});
        manager.register_pass<ov::intel_cpu::SwapConvertTranspose>();
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(in_type, shape);

        auto transpose_const = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{input_order.size()}, input_order);
        auto transpose = std::make_shared<ov::op::v1::Transpose>(input, transpose_const);

        auto convert = std::make_shared<ov::op::v0::Convert>(transpose, out_type);

        transpose->set_friendly_name(transpose_name + "_original");
        convert->set_friendly_name(transpose_name);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{convert}, ov::ParameterVector{input});
    }
}

TEST_F(SwapConvertTransposeTest, SwapConvertTransposeImpossible) {
    const ov::Shape shape{1, 224, 224, 3};
    const std::vector<int64_t> input_order = {0, 3, 1, 2};
    const ov::element::Type in_type = ov::element::u8;
    const ov::element::Type out_type = ov::element::f32;
    const std::string transpose_name = "Transpose";

    {
        auto input = std::make_shared<ov::op::v0::Parameter>(in_type, shape);

        auto convert = std::make_shared<ov::op::v0::Convert>(input, out_type);

        auto transpose0_const = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{input_order.size()}, input_order);
        auto transpose0 = std::make_shared<ov::op::v1::Transpose>(convert, transpose0_const);
        transpose0->set_friendly_name(transpose_name + "_0");

        auto transpose1_const = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{input_order.size()}, input_order);
        auto transpose1 = std::make_shared<ov::op::v1::Transpose>(convert, transpose1_const);
        transpose1->set_friendly_name(transpose_name + "_1");

        model = std::make_shared<ov::Model>(ov::NodeVector{transpose0, transpose1}, ov::ParameterVector{input});
        manager.register_pass<ov::intel_cpu::SwapConvertTranspose>();
    }
}
