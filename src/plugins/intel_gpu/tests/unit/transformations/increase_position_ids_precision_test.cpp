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
#include "openvino/op/sin.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"

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

        model = std::make_shared<ov::Model>(ov::NodeVector{ rope }, ov::ParameterVector{ input, rope_input });
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

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{ rope }, ov::ParameterVector{ input, rope_input });
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

        model = std::make_shared<ov::Model>(ov::NodeVector{ rope }, ov::ParameterVector{ input, rope_input });
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

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{ rope }, ov::ParameterVector{ input, rope_input });
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

        model = std::make_shared<ov::Model>(ov::NodeVector{ rope }, ov::ParameterVector{ input, rope_input });
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

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{ rope }, ov::ParameterVector{ input, rope_input });
    }
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}
