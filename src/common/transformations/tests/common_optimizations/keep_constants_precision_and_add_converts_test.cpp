// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/opsets/opset1_decl.hpp"
#include "transformations/convert_precision.hpp"
#include "transformations/fp16_compression/mark_decompression_convert_constant_folding.hpp"
#include "transformations/utils/utils.hpp"

using namespace testing;
using namespace ov;
using namespace ov::opset1;
using const_node_ptr = const std::shared_ptr<const Node>;

TEST_F(TransformationTestsF, KeepConstantsPrecisionAndAddConvertsTestBase) {
    {
        auto input = std::make_shared<Parameter>(element::f32, Shape{3, 2, 2});
        auto weights = Constant::create(element::f32, Shape{1, 2, 2}, {1});
        auto matmul = std::make_shared<MatMul>(input, weights);

        model = std::make_shared<Model>(OutputVector{matmul}, ParameterVector{input});

        manager.register_pass<pass::KeepConstantsPrecisionAndAddConverts>();
        manager.get_pass_config()->set_callback<pass::KeepConstantsPrecisionAndAddConverts>(
            [](const_node_ptr& node) -> bool {
                auto next_node = node->get_output_target_inputs(0).begin()->get_node();
                if (is_type<op::v0::Convert>(next_node)) {
                    next_node = next_node->get_output_target_inputs(0).begin()->get_node();
                }
                return !is_type<op::v0::MatMul>(next_node);
            });

        const precisions_map precisions = {{element::f32, element::f16}};
        const type_to_fuse_map empty_fuse_map = {};
        const bool keep_precision_sensitive_in_fp32_1 = true;
        manager.register_pass<pass::ConvertPrecision>(precisions, empty_fuse_map, keep_precision_sensitive_in_fp32_1);
    }
    {
        auto input = std::make_shared<Parameter>(element::f16, Shape{3, 2, 2});
        auto weights = Constant::create(element::f32, Shape{1, 2, 2}, {1});
        auto convert_weights = std::make_shared<Convert>(weights, element::f16);
        auto matmul = std::make_shared<MatMul>(input, convert_weights);

        model_ref = std::make_shared<Model>(OutputVector{matmul}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, KeepConstantsPrecisionAndAddConvertsTestWithCompressedConvert) {
    {
        auto input = std::make_shared<Parameter>(element::f16, Shape{3, 2, 2});
        auto weights = Constant::create(element::f32, Shape{1, 2, 2}, {1});
        auto convert_weights = std::make_shared<Convert>(weights, element::f16);
        mark_as_decompression(convert_weights);
        auto matmul = std::make_shared<MatMul>(input, convert_weights);

        model = std::make_shared<Model>(OutputVector{matmul}, ParameterVector{input});

        manager.register_pass<pass::KeepConstantsPrecisionAndAddConverts>();
        manager.get_pass_config()->set_callback<pass::KeepConstantsPrecisionAndAddConverts>(
            [](const_node_ptr& node) -> bool {
                auto next_node = node->get_output_target_inputs(0).begin()->get_node();
                if (is_type<op::v0::Convert>(next_node)) {
                    next_node = next_node->get_output_target_inputs(0).begin()->get_node();
                }
                return !is_type<op::v0::MatMul>(next_node);
            });

        const precisions_map precisions = {{element::f32, element::f16}};
        const type_to_fuse_map empty_fuse_map = {};
        const bool keep_precision_sensitive_in_fp32_1 = true;
        manager.register_pass<pass::ConvertPrecision>(precisions, empty_fuse_map, keep_precision_sensitive_in_fp32_1);
    }
    {
        auto input = std::make_shared<Parameter>(element::f16, Shape{3, 2, 2});
        auto weights = Constant::create(element::f32, Shape{1, 2, 2}, {1});
        auto convert_weights = std::make_shared<Convert>(weights, element::f16);
        auto matmul = std::make_shared<MatMul>(input, convert_weights);

        model_ref = std::make_shared<Model>(OutputVector{matmul}, ParameterVector{input});
    }
}
