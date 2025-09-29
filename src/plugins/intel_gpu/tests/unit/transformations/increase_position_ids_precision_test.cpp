// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <limits>
#include <string>
#include <memory>

#include "intel_gpu/op/gemm.hpp"
#include "ov_ops/rotary_positional_embeddings.hpp"

#include <openvino/pass/manager.hpp>
#include <openvino/core/model.hpp>
#include "openvino/core/coordinate_diff.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/sin.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/broadcast.hpp"
#include "ov_ops/rms.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"

#include <plugin/transformations/increase_position_ids_precision.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>
#include "openvino/pass/visualize_tree.hpp"

#include "common_test_utils/ov_test_utils.hpp"

using namespace testing;
using namespace ov::intel_gpu;

TEST_F(TransformationTestsF, IncreasePositionIdsPrecisionWithoutUnsqueeze) {
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{ -1, -1 });
        auto input_convert = std::make_shared<ov::op::v0::Convert>(input, ov::element::i32);
        auto input_unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(input_convert, std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1}));
        auto input_convert_fp = std::make_shared<ov::op::v0::Convert>(input_unsqueeze, ov::element::f16);
        auto rotary_embd_const = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1, 64, 1});

        auto gemm = std::make_shared<ov::intel_gpu::op::Gemm>(input_convert_fp, rotary_embd_const, std::vector<int64_t>{}, std::vector<int64_t>{}, std::vector<int64_t>{});
        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{gemm, gemm}, 2);

        auto cos = std::make_shared<ov::op::v0::Cos>(concat);
        auto sin = std::make_shared<ov::op::v0::Sin>(concat);

        auto rope_input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape::dynamic(4));
        auto rope = std::make_shared<ov::op::internal::RoPE>(ov::OutputVector{rope_input, cos, sin}, ov::op::internal::RoPE::Config());

        model = std::make_shared<ov::Model>(ov::OutputVector{rope}, ov::ParameterVector{input, rope_input});
        manager.register_pass<IncreasePositionIdsPrecision>();
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{ -1, -1 });
        auto input_convert = std::make_shared<ov::op::v0::Convert>(input, ov::element::i32);
        auto input_unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(input_convert, std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1}));
        auto rotary_embd_const = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1, 64, 1});

        auto input_convert_f32 = std::make_shared<ov::op::v0::Convert>(input_unsqueeze, ov::element::f32);
        auto rotary_embd_const_convert_f32 = std::make_shared<ov::op::v0::Convert>(rotary_embd_const, ov::element::f32);

        auto gemm = std::make_shared<ov::intel_gpu::op::Gemm>(input_convert_f32, rotary_embd_const_convert_f32, std::vector<int64_t>{}, std::vector<int64_t>{}, std::vector<int64_t>{});
        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{gemm, gemm}, 2);

        auto cos = std::make_shared<ov::op::v0::Cos>(concat);
        auto sin = std::make_shared<ov::op::v0::Sin>(concat);

        auto cos_convert = std::make_shared<ov::op::v0::Convert>(cos, ov::element::f16);
        auto sin_convert = std::make_shared<ov::op::v0::Convert>(sin, ov::element::f16);

        auto rope_input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape::dynamic(4));
        auto rope = std::make_shared<ov::op::internal::RoPE>(ov::OutputVector{rope_input, cos_convert, sin_convert}, ov::op::internal::RoPE::Config());

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{rope}, ov::ParameterVector{input, rope_input});
    }
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, IncreasePositionIdsPrecisionWithUnsqueeze) {
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{ -1, -1 });
        auto input_convert = std::make_shared<ov::op::v0::Convert>(input, ov::element::i32);
        auto input_unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(input_convert, std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1}));
        auto input_convert_fp = std::make_shared<ov::op::v0::Convert>(input_unsqueeze, ov::element::f16);
        auto rotary_embd_const = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1, 64, 1});

        auto gemm = std::make_shared<ov::intel_gpu::op::Gemm>(input_convert_fp, rotary_embd_const, std::vector<int64_t>{}, std::vector<int64_t>{}, std::vector<int64_t>{});
        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{gemm, gemm}, 2);

        auto cos = std::make_shared<ov::op::v0::Cos>(concat);
        auto sin = std::make_shared<ov::op::v0::Sin>(concat);

        auto cos_unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(cos, std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1}));
        auto sin_unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(sin, std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1}));

        auto rope_input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape::dynamic(4));
        auto rope = std::make_shared<ov::op::internal::RoPE>(ov::OutputVector{rope_input, cos_unsqueeze, sin_unsqueeze}, ov::op::internal::RoPE::Config());

        model = std::make_shared<ov::Model>(ov::OutputVector{rope}, ov::ParameterVector{input, rope_input});
        manager.register_pass<IncreasePositionIdsPrecision>();
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{ -1, -1 });
        auto input_convert = std::make_shared<ov::op::v0::Convert>(input, ov::element::i32);
        auto input_unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(input_convert, std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1}));
        auto rotary_embd_const = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1, 64, 1});

        auto input_convert_f32 = std::make_shared<ov::op::v0::Convert>(input_unsqueeze, ov::element::f32);
        auto rotary_embd_const_convert_f32 = std::make_shared<ov::op::v0::Convert>(rotary_embd_const, ov::element::f32);

        auto gemm = std::make_shared<ov::intel_gpu::op::Gemm>(input_convert_f32, rotary_embd_const_convert_f32, std::vector<int64_t>{}, std::vector<int64_t>{}, std::vector<int64_t>{});
        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{gemm, gemm}, 2);

        auto cos = std::make_shared<ov::op::v0::Cos>(concat);
        auto sin = std::make_shared<ov::op::v0::Sin>(concat);

        auto cos_convert = std::make_shared<ov::op::v0::Convert>(cos, ov::element::f16);
        auto sin_convert = std::make_shared<ov::op::v0::Convert>(sin, ov::element::f16);

        auto cos_unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(cos_convert, std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1}));
        auto sin_unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(sin_convert, std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1}));

        auto rope_input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape::dynamic(4));
        auto rope = std::make_shared<ov::op::internal::RoPE>(ov::OutputVector{rope_input, cos_unsqueeze, sin_unsqueeze}, ov::op::internal::RoPE::Config());

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{rope}, ov::ParameterVector{input, rope_input});
    }
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, IncreasePositionIdsMatmulWithoutUnsqueeze) {
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{ -1, -1 });
        auto input_convert = std::make_shared<ov::op::v0::Convert>(input, ov::element::i32);
        auto input_unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(input_convert, std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1}));
        auto input_convert_fp = std::make_shared<ov::op::v0::Convert>(input_unsqueeze, ov::element::f16);
        auto rotary_embd_const = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1, 64, 1});

        auto matmul = std::make_shared<ov::op::v0::MatMul>(input_convert_fp, rotary_embd_const);
        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{matmul, matmul}, 2);

        auto cos = std::make_shared<ov::op::v0::Cos>(concat);
        auto sin = std::make_shared<ov::op::v0::Sin>(concat);

        auto rope_input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape::dynamic(4));
        auto rope = std::make_shared<ov::op::internal::RoPE>(ov::OutputVector{rope_input, cos, sin}, ov::op::internal::RoPE::Config());

        model = std::make_shared<ov::Model>(ov::OutputVector{rope}, ov::ParameterVector{input, rope_input});
        manager.register_pass<IncreasePositionIdsPrecision>();
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{ -1, -1 });
        auto input_convert = std::make_shared<ov::op::v0::Convert>(input, ov::element::i32);
        auto input_unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(input_convert, std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1}));
        auto rotary_embd_const = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1, 64, 1});

        auto input_convert_f32 = std::make_shared<ov::op::v0::Convert>(input_unsqueeze, ov::element::f32);
        auto rotary_embd_const_convert_f32 = std::make_shared<ov::op::v0::Convert>(rotary_embd_const, ov::element::f32);

        auto matmul = std::make_shared<ov::op::v0::MatMul>(input_convert_f32, rotary_embd_const_convert_f32);
        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{matmul, matmul}, 2);

        auto cos = std::make_shared<ov::op::v0::Cos>(concat);
        auto sin = std::make_shared<ov::op::v0::Sin>(concat);

        auto cos_convert = std::make_shared<ov::op::v0::Convert>(cos, ov::element::f16);
        auto sin_convert = std::make_shared<ov::op::v0::Convert>(sin, ov::element::f16);

        auto rope_input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape::dynamic(4));
        auto rope = std::make_shared<ov::op::internal::RoPE>(ov::OutputVector{rope_input, cos_convert, sin_convert}, ov::op::internal::RoPE::Config());

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{rope}, ov::ParameterVector{input, rope_input});
    }
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, IncreasePositionIdsReshapeAfterMatmul) {
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{ -1, -1 });
        auto input_convert = std::make_shared<ov::op::v0::Convert>(input, ov::element::i32);
        auto input_unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(input_convert, std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1}));
        auto input_convert_fp = std::make_shared<ov::op::v0::Convert>(input_unsqueeze, ov::element::f16);
        auto rotary_embd_const = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1, 64, 1});
        auto reshape_dims = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{3});

        auto matmul = std::make_shared<ov::op::v0::MatMul>(input_convert_fp, rotary_embd_const);
        auto reshape = std::make_shared<ov::op::v1::Reshape>(matmul, reshape_dims, true);
        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{reshape, reshape}, 2);

        auto cos = std::make_shared<ov::op::v0::Cos>(concat);
        auto sin = std::make_shared<ov::op::v0::Sin>(concat);

        auto rope_input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape::dynamic(4));
        auto rope = std::make_shared<ov::op::internal::RoPE>(ov::OutputVector{rope_input, cos, sin}, ov::op::internal::RoPE::Config());

        model = std::make_shared<ov::Model>(ov::OutputVector{rope}, ov::ParameterVector{input, rope_input, reshape_dims});
        manager.register_pass<IncreasePositionIdsPrecision>();
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{ -1, -1 });
        auto input_convert = std::make_shared<ov::op::v0::Convert>(input, ov::element::i32);
        auto input_unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(input_convert, std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1}));
        auto rotary_embd_const = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1, 64, 1});
        auto reshape_dims = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{3});

        auto input_convert_f32 = std::make_shared<ov::op::v0::Convert>(input_unsqueeze, ov::element::f32);
        auto rotary_embd_const_convert_f32 = std::make_shared<ov::op::v0::Convert>(rotary_embd_const, ov::element::f32);

        auto matmul = std::make_shared<ov::op::v0::MatMul>(input_convert_f32, rotary_embd_const_convert_f32);
        auto reshape = std::make_shared<ov::op::v1::Reshape>(matmul, reshape_dims, true);
        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{reshape, reshape}, 2);

        auto cos = std::make_shared<ov::op::v0::Cos>(concat);
        auto sin = std::make_shared<ov::op::v0::Sin>(concat);

        auto cos_convert = std::make_shared<ov::op::v0::Convert>(cos, ov::element::f16);
        auto sin_convert = std::make_shared<ov::op::v0::Convert>(sin, ov::element::f16);

        auto rope_input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape::dynamic(4));
        auto rope = std::make_shared<ov::op::internal::RoPE>(ov::OutputVector{rope_input, cos_convert, sin_convert}, ov::op::internal::RoPE::Config());

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{rope}, ov::ParameterVector{input, rope_input, reshape_dims});
    }
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, IncreasePositionIdsLongRoPE) {
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{ -1 });
        auto input_convert = std::make_shared<ov::op::v0::Convert>(input, ov::element::i32);
        auto input_unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(input_convert, std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{-1, 1, 1}));
        auto input_convert_fp = std::make_shared<ov::op::v0::Convert>(input_unsqueeze, ov::element::f16);
        auto rotary_embd = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 48, 1});

        auto matmul = std::make_shared<ov::op::v0::MatMul>(rotary_embd, input_convert_fp);
        auto reshape = std::make_shared<ov::op::v1::Reshape>(matmul, std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{0, 1, 48}), true);
        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{reshape, reshape}, 2);

        auto cos = std::make_shared<ov::op::v0::Cos>(concat);
        auto sin = std::make_shared<ov::op::v0::Sin>(concat);

        auto const_scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 1, 1, 1 }, { 1.19043 });
        auto const_scale = std::make_shared<ov::op::v1::Multiply>(cos, const_scale_const);
        auto sin_scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 1, 1, 1 }, { 1.19043 });
        auto sin_scale = std::make_shared<ov::op::v1::Multiply>(sin, sin_scale_const);

        auto cos_unsqueeze = std::make_shared<ov::op::v1::Reshape>(const_scale, std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{-1, 1, 1, 96}), true);
        auto sin_unsqueeze = std::make_shared<ov::op::v1::Reshape>(sin_scale, std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{-1, 1, 1, 96}), true);

        auto rope_input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape::dynamic(4));
        auto rope = std::make_shared<ov::op::internal::RoPE>(ov::OutputVector{rope_input, cos_unsqueeze, sin_unsqueeze}, ov::op::internal::RoPE::Config());

        model = std::make_shared<ov::Model>(ov::OutputVector{rope}, ov::ParameterVector{input, rope_input, rotary_embd});
        manager.register_pass<IncreasePositionIdsPrecision>();
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{ -1 });
        auto input_convert = std::make_shared<ov::op::v0::Convert>(input, ov::element::i32);
        auto input_unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(input_convert, std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{-1, 1, 1}));
        auto input_convert_fp = std::make_shared<ov::op::v0::Convert>(input_unsqueeze, ov::element::f16);
        auto rotary_embd = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 48, 1});

        auto input_convert_f32 = std::make_shared<ov::op::v0::Convert>(input_unsqueeze, ov::element::f32);
        auto rotary_embd_convert_f32 = std::make_shared<ov::op::v0::Convert>(rotary_embd, ov::element::f32);

        auto matmul = std::make_shared<ov::op::v0::MatMul>(rotary_embd_convert_f32, input_convert_f32);
        auto reshape = std::make_shared<ov::op::v1::Reshape>(matmul, std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{0, 1, 48}), true);
        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{reshape, reshape}, 2);

        auto cos = std::make_shared<ov::op::v0::Cos>(concat);
        auto sin = std::make_shared<ov::op::v0::Sin>(concat);

        auto cos_convert = std::make_shared<ov::op::v0::Convert>(cos, ov::element::f16);
        auto sin_convert = std::make_shared<ov::op::v0::Convert>(sin, ov::element::f16);

        auto const_scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 1, 1, 1 }, { 1.19043 });
        auto const_scale = std::make_shared<ov::op::v1::Multiply>(cos_convert, const_scale_const);
        auto sin_scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 1, 1, 1 }, { 1.19043 });
        auto sin_scale = std::make_shared<ov::op::v1::Multiply>(sin_convert, sin_scale_const);

        auto cos_unsqueeze = std::make_shared<ov::op::v1::Reshape>(const_scale, std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{-1, 1, 1, 96}), true);
        auto sin_unsqueeze = std::make_shared<ov::op::v1::Reshape>(sin_scale, std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{-1, 1, 1, 96}), true);

        auto rope_input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape::dynamic(4));
        auto rope = std::make_shared<ov::op::internal::RoPE>(ov::OutputVector{rope_input, cos_unsqueeze, sin_unsqueeze}, ov::op::internal::RoPE::Config());

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{rope}, ov::ParameterVector{input, rope_input, rotary_embd});
    }
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, IncreasePositionIdsSliceGatherUnsqueezeRoPE) {
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{ -1 });
        auto input_convert = std::make_shared<ov::op::v0::Convert>(input, ov::element::i32);
        auto input_unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(input_convert, std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{-1, 1}));
        auto input_convert_fp = std::make_shared<ov::op::v0::Convert>(input_unsqueeze, ov::element::f16);
        auto rotary_embd = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{32, 1});

        auto matmul = std::make_shared<ov::op::v0::MatMul>(rotary_embd, input_convert_fp);
        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{matmul, matmul}, 2);

        auto cos = std::make_shared<ov::op::v0::Cos>(concat);
        auto sin = std::make_shared<ov::op::v0::Sin>(concat);

        auto constant_11 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 1 }, { 1 });
        auto constant_12 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 1 }, { 0 });
        auto constant_13 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 1 }, { 1 });
        auto sin_slice = std::make_shared<ov::op::v1::StridedSlice>(sin, constant_11, constant_12, constant_13, std::vector<int64_t>{}, std::vector<int64_t>{});
        auto sin_gather = std::make_shared<ov::op::v8::Gather>(sin_slice, constant_11, constant_12);
        auto sin_unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(sin_gather, constant_13);

        auto cos_slice = std::make_shared<ov::op::v1::StridedSlice>(cos, constant_11, constant_12, constant_13, std::vector<int64_t>{}, std::vector<int64_t>{});
        auto cos_gather = std::make_shared<ov::op::v8::Gather>(cos_slice, constant_11, constant_12);
        auto cos_unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(cos_gather, constant_13);

        auto rope_input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape::dynamic(4));
        auto rope = std::make_shared<ov::op::internal::RoPE>(ov::OutputVector{rope_input, cos_unsqueeze, sin_unsqueeze}, ov::op::internal::RoPE::Config());

        model = std::make_shared<ov::Model>(ov::OutputVector{rope}, ov::ParameterVector{input, rope_input, rotary_embd});
        manager.register_pass<IncreasePositionIdsPrecision>();
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{ -1 });
        auto input_convert = std::make_shared<ov::op::v0::Convert>(input, ov::element::i32);
        auto input_unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(input_convert, std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{-1, 1}));
        auto input_convert_fp = std::make_shared<ov::op::v0::Convert>(input_unsqueeze, ov::element::f16);
        auto rotary_embd = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{32, 1});

        auto input_convert_f32 = std::make_shared<ov::op::v0::Convert>(input_unsqueeze, ov::element::f32);
        auto rotary_embd_convert_f32 = std::make_shared<ov::op::v0::Convert>(rotary_embd, ov::element::f32);

        auto matmul = std::make_shared<ov::op::v0::MatMul>(rotary_embd_convert_f32, input_convert_f32);
        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{matmul, matmul}, 2);

        auto cos = std::make_shared<ov::op::v0::Cos>(concat);
        auto sin = std::make_shared<ov::op::v0::Sin>(concat);

        auto cos_convert = std::make_shared<ov::op::v0::Convert>(cos, ov::element::f16);
        auto sin_convert = std::make_shared<ov::op::v0::Convert>(sin, ov::element::f16);

        auto constant_11 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 1 }, { 1 });
        auto constant_12 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 1 }, { 0 });
        auto constant_13 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 1 }, { 1 });
        auto sin_slice = std::make_shared<ov::op::v1::StridedSlice>(sin_convert, constant_11, constant_12, constant_13, std::vector<int64_t>{}, std::vector<int64_t>{});
        auto sin_gather = std::make_shared<ov::op::v8::Gather>(sin_slice, constant_11, constant_12);
        auto sin_unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(sin_gather, constant_13);

        auto cos_slice = std::make_shared<ov::op::v1::StridedSlice>(cos_convert, constant_11, constant_12, constant_13, std::vector<int64_t>{}, std::vector<int64_t>{});
        auto cos_gather = std::make_shared<ov::op::v8::Gather>(cos_slice, constant_11, constant_12);
        auto cos_unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(cos_gather, constant_13);

        auto rope_input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape::dynamic(4));
        auto rope = std::make_shared<ov::op::internal::RoPE>(ov::OutputVector{rope_input, cos_unsqueeze, sin_unsqueeze}, ov::op::internal::RoPE::Config());

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{rope}, ov::ParameterVector{input, rope_input, rotary_embd});
    }
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, IncreasePositionIdsLTXVideo) {
    {
        auto input_1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 3, -1 });
        auto input_2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, -1, 2048 });

        auto constant_01 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 3 }, { 0, 2, 1 });
        auto transpose = std::make_shared<ov::op::v1::Transpose>(input_1, constant_01);
        auto constant_02 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 1 }, { -1 });
        auto unsqueeze_1 = std::make_shared<ov::op::v0::Unsqueeze>(transpose, constant_02);
        auto constant_03 = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 1, 1, 1, 341  }, { 3.14f });
        auto multiply = std::make_shared<ov::op::v1::Multiply>(unsqueeze_1, constant_03);

        auto constant_04 = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 1, 1, 1, 341 }, { 1e-6 });
        auto add = std::make_shared<ov::op::v1::Add>(multiply, constant_04);

        auto constant_05 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 4 }, { 0, 1, 3, 2 });
        auto transpose_1 = std::make_shared<ov::op::v1::Transpose>(add, constant_05);

        auto constant_06 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 3 }, { 0, 0, 1023 });
        auto reshape_1 = std::make_shared<ov::op::v1::Reshape>(transpose_1, constant_06, true);

        auto cos = std::make_shared<ov::op::v0::Cos>(reshape_1);
        auto sin = std::make_shared<ov::op::v0::Sin>(reshape_1);

        auto constant_07 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 2046 }, { 0 });
        auto constant_08 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 2046 }, { 0 });
        auto constant_09 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ }, { -1 });
        auto constant_10 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ }, { -1 });

        auto gather_1 = std::make_shared<ov::op::v8::Gather>(cos, constant_07, constant_09);
        auto gather_3 = std::make_shared<ov::op::v8::Gather>(sin, constant_08, constant_10);

        auto constant_11 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 3 }, { 0, 0, 0 });
        auto constant_12 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 3 }, { 0, 0, 2 });
        auto constant_13 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 3 }, { 1, 1, 1 });
        auto slice = std::make_shared<ov::op::v1::StridedSlice>(gather_1, constant_11, constant_12, constant_13, std::vector<int64_t>{}, std::vector<int64_t>{});

        auto shape_of = std::make_shared<ov::op::v3::ShapeOf>(slice);
        auto constant_14 = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ }, { 1 });
        auto constant_15 = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ }, { 0 });
        auto broadcast_ones_like = std::make_shared<ov::op::v3::Broadcast>(constant_14, shape_of);
        auto broadcast_zeros_like = std::make_shared<ov::op::v3::Broadcast>(constant_15, shape_of);

        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{broadcast_ones_like, gather_1}, -1);
        auto concat_1 = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{broadcast_zeros_like, gather_3}, -1);

        auto constant_16 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 1, 1, 2048 }, { 0 });
        auto rms = std::make_shared<ov::op::internal::RMS>(input_2, constant_16, 1e-19);

        auto constant_17 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 4 }, { 0, 0, 1024, 2 });
        auto reshape_2 = std::make_shared<ov::op::v1::Reshape>(rms, constant_17, true);

        auto constant_18 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ }, { -1 });
        auto split_1 = std::make_shared<ov::op::v1::Split>(reshape_2, constant_18, 2);

        auto constant_19 = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ }, { -1 });
        auto multiply_2 = std::make_shared<ov::op::v1::Multiply>(split_1->output(0), constant_19);

        auto constant_20 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ }, { -1 });
        auto squeeze_1 = std::make_shared<ov::op::v0::Squeeze>(multiply_2, constant_20);

        auto constant_21 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ }, { -1 });
        auto unsqueeze_2 = std::make_shared<ov::op::v0::Unsqueeze>(squeeze_1, constant_21);

        auto concat_stack_1 = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{unsqueeze_2, split_1->output(1)}, -1);
        auto constant_22 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 3 }, { 0, 0, 2048 });
        auto reshape_3 = std::make_shared<ov::op::v1::Reshape>(concat_stack_1, constant_22, true);

        auto multiply_3 = std::make_shared<ov::op::v1::Multiply>(rms, concat);
        auto multiply_4 = std::make_shared<ov::op::v1::Multiply>(reshape_3, concat_1);

        auto add_1 = std::make_shared<ov::op::v1::Add>(multiply_3, multiply_4);
        auto constant_23 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 4 }, { 0, 0, 32, 64 });
        auto reshape_4 = std::make_shared<ov::op::v1::Reshape>(add_1, constant_23, true);
        auto constant_24 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 4 }, { 0, 2, 1, 3 });
        auto transpose_3 = std::make_shared<ov::op::v1::Transpose>(reshape_4, constant_24);

        auto input_3 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, -1, 32, 64 });
        auto input_4 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, -1, 32, 64 });
        auto transpose_2 = std::make_shared<ov::op::v1::Transpose>(input_3, constant_24);
        auto transpose_4 = std::make_shared<ov::op::v1::Transpose>(input_4, constant_24);
        auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(transpose_2, transpose_3, transpose_4, true);

        model = std::make_shared<ov::Model>(ov::OutputVector{sdpa}, ov::ParameterVector{input_1, input_2, input_3, input_4});
        manager.register_pass<IncreasePositionIdsPrecision>();
    }
    {
        auto input_1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 3, -1 });
        auto input_2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, -1, 2048 });

        auto constant_01 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 3 }, { 0, 2, 1 });
        auto transpose = std::make_shared<ov::op::v1::Transpose>(input_1, constant_01);
        auto constant_02 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 1 }, { -1 });
        auto unsqueeze_1 = std::make_shared<ov::op::v0::Unsqueeze>(transpose, constant_02);
        auto constant_03 = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 1, 1, 1, 341  }, { 3.14f });

        auto convert_1 = std::make_shared<ov::op::v0::Convert>(unsqueeze_1, ov::element::f32);
        auto convert_2 = std::make_shared<ov::op::v0::Convert>(constant_03, ov::element::f32);
        auto multiply = std::make_shared<ov::op::v1::Multiply>(convert_1, convert_2);

        auto constant_04 = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 1, 1, 1, 341 }, { 1e-6 });
        auto convert_3 = std::make_shared<ov::op::v0::Convert>(constant_04, ov::element::f32);
        auto add = std::make_shared<ov::op::v1::Add>(multiply, convert_3);

        auto constant_05 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 4 }, { 0, 1, 3, 2 });
        auto transpose_1 = std::make_shared<ov::op::v1::Transpose>(add, constant_05);

        auto constant_06 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 3 }, { 0, 0, 1023 });
        auto reshape_1 = std::make_shared<ov::op::v1::Reshape>(transpose_1, constant_06, true);

        auto cos = std::make_shared<ov::op::v0::Cos>(reshape_1);
        auto sin = std::make_shared<ov::op::v0::Sin>(reshape_1);

        auto convert_4 = std::make_shared<ov::op::v0::Convert>(cos, ov::element::f16);
        auto convert_5 = std::make_shared<ov::op::v0::Convert>(sin, ov::element::f16);

        auto constant_07 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 2046 }, { 0 });
        auto constant_08 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 2046 }, { 0 });
        auto constant_09 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ }, { -1 });
        auto constant_10 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ }, { -1 });

        auto gather_1 = std::make_shared<ov::op::v8::Gather>(convert_4, constant_07, constant_09);
        auto gather_3 = std::make_shared<ov::op::v8::Gather>(convert_5, constant_08, constant_10);

        auto constant_11 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 3 }, { 0, 0, 0 });
        auto constant_12 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 3 }, { 0, 0, 2 });
        auto constant_13 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 3 }, { 1, 1, 1 });
        auto slice = std::make_shared<ov::op::v1::StridedSlice>(gather_1, constant_11, constant_12, constant_13, std::vector<int64_t>{}, std::vector<int64_t>{});

        auto shape_of = std::make_shared<ov::op::v3::ShapeOf>(slice);
        auto constant_14 = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ }, { 1 });
        auto constant_15 = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ }, { 0 });
        auto broadcast_ones_like = std::make_shared<ov::op::v3::Broadcast>(constant_14, shape_of);
        auto broadcast_zeros_like = std::make_shared<ov::op::v3::Broadcast>(constant_15, shape_of);

        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{broadcast_ones_like, gather_1}, -1);
        auto concat_1 = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{broadcast_zeros_like, gather_3}, -1);

        auto constant_16 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 1, 1, 2048 }, { 0 });
        auto rms = std::make_shared<ov::op::internal::RMS>(input_2, constant_16, 1e-19);

        auto constant_17 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 4 }, { 0, 0, 1024, 2 });
        auto reshape_2 = std::make_shared<ov::op::v1::Reshape>(rms, constant_17, true);

        auto constant_18 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ }, { -1 });
        auto split_1 = std::make_shared<ov::op::v1::Split>(reshape_2, constant_18, 2);

        auto constant_19 = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ }, { -1 });
        auto multiply_2 = std::make_shared<ov::op::v1::Multiply>(split_1->output(0), constant_19);

        auto constant_20 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ }, { -1 });
        auto squeeze_1 = std::make_shared<ov::op::v0::Squeeze>(multiply_2, constant_20);

        auto constant_21 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ }, { -1 });
        auto unsqueeze_2 = std::make_shared<ov::op::v0::Unsqueeze>(squeeze_1, constant_21);

        auto concat_stack_1 = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{unsqueeze_2, split_1->output(1)}, -1);
        auto constant_22 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 3 }, { 0, 0, 2048 });
        auto reshape_3 = std::make_shared<ov::op::v1::Reshape>(concat_stack_1, constant_22, true);

        auto multiply_3 = std::make_shared<ov::op::v1::Multiply>(rms, concat);
        auto multiply_4 = std::make_shared<ov::op::v1::Multiply>(reshape_3, concat_1);

        auto add_1 = std::make_shared<ov::op::v1::Add>(multiply_3, multiply_4);
        auto constant_23 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 4 }, { 0, 0, 32, 64 });
        auto reshape_4 = std::make_shared<ov::op::v1::Reshape>(add_1, constant_23, true);
        auto constant_24 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 4 }, { 0, 2, 1, 3 });
        auto transpose_3 = std::make_shared<ov::op::v1::Transpose>(reshape_4, constant_24);

        auto input_3 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, -1, 32, 64 });
        auto input_4 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, -1, 32, 64 });
        auto transpose_2 = std::make_shared<ov::op::v1::Transpose>(input_3, constant_24);
        auto transpose_4 = std::make_shared<ov::op::v1::Transpose>(input_4, constant_24);
        auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(transpose_2, transpose_3, transpose_4, true);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{sdpa}, ov::ParameterVector{input_1, input_2, input_3, input_4});
    }
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}
