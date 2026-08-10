// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/core/model.hpp"
#include "openvino/op/bitwise_and.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/manager.hpp"
#include "plugin/transformations/preserve_boolean_attn_mask_precision.hpp"
#include "transformations/convert_precision.hpp"

namespace ov::test::intel_gpu {

namespace v0 = ov::op::v0;

TEST(GPUTransformPipelineTest, PreservesBooleanAttentionMaskProducerChain) {
    auto query = std::make_shared<v0::Parameter>(element::f16, PartialShape{1, 2, 4, 8});
    auto key = std::make_shared<v0::Parameter>(element::f16, PartialShape{1, 2, 4, 8});
    auto value = std::make_shared<v0::Parameter>(element::f16, PartialShape{1, 2, 4, 8});
    auto mask_lhs = std::make_shared<v0::Parameter>(element::boolean, PartialShape{4, 4});
    auto mask_rhs = std::make_shared<v0::Parameter>(element::boolean, PartialShape{4, 4});

    auto mask_and = std::make_shared<ov::op::v13::BitwiseAnd>(mask_lhs, mask_rhs);
    auto unsqueeze_batch = std::make_shared<v0::Unsqueeze>(
        mask_and,
        v0::Constant::create(element::i64, Shape{1}, {0}));
    auto unsqueeze_heads = std::make_shared<v0::Unsqueeze>(
        unsqueeze_batch,
        v0::Constant::create(element::i64, Shape{1}, {1}));
    auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(
        query, key, value, unsqueeze_heads, false);

    auto model = std::make_shared<Model>(OutputVector{sdpa},
                                         ParameterVector{query, key, value, mask_lhs, mask_rhs});

    ov::pass::Manager manager;
    manager.register_pass<ov::intel_gpu::PreserveBooleanAttnMaskPrecision>();
    precisions_map precision_map = {{element::boolean, element::u8}};
    manager.register_pass<ov::pass::ConvertPrecision>(precision_map);
    manager.run_passes(model);

    EXPECT_EQ(mask_lhs->get_output_element_type(0), element::boolean);
    EXPECT_EQ(mask_rhs->get_output_element_type(0), element::boolean);
    EXPECT_EQ(mask_and->get_output_element_type(0), element::boolean);
    EXPECT_EQ(unsqueeze_batch->get_output_element_type(0), element::boolean);
    EXPECT_EQ(unsqueeze_heads->get_output_element_type(0), element::boolean);
    EXPECT_EQ(sdpa->get_input_element_type(3), element::boolean);
}

}  // namespace ov::test::intel_gpu
