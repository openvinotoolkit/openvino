// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "intel_gpu/op/moe_compressed.hpp"
#include "intel_gpu/op/moe_3gemm_fused_compressed.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "plugin/transformations/fuse_moe_3gemm_compressed.hpp"

using namespace testing;
using namespace ov::intel_gpu;

namespace ov {
namespace test {
namespace intel_gpu {
TEST_F(TransformationTestsF, FuseMOE3GemmCompressedTest) {
    {
        // tokens:32, hidden_size:2048, iter_size:768, experts:128, topk:8
        auto hidden_states = std::make_shared<ov::op::v0::Parameter>(element::f16, Shape{32, 2048});
        auto routers = op::v0::Constant::create(element::f16, Shape{2048, 128}, {0.2});
        auto routing_weights = std::make_shared<ov::op::v0::MatMul>(hidden_states, routers);

        auto softmax = std::make_shared<ov::op::v8::Softmax>(routing_weights, 1);
        auto k = op::v0::Constant::create(element::i32, Shape{}, {8});
        auto topk = std::make_shared<ov::op::v11::TopK>(softmax, k, 1,
            ov::op::v11::TopK::Mode::MAX, ov::op::v11::TopK::SortType::SORT_VALUES);

        // weight output
        auto reduce_axis = op::v0::Constant::create(element::i64, Shape{1}, {1});
        auto reduce_sum = std::make_shared<ov::op::v1::ReduceSum>(topk->output(0), reduce_axis->output(0), true);
        auto norm = std::make_shared<ov::op::v1::Divide>(topk->output(0), reduce_sum->output(0));

        // 32
        auto shape_of = std::make_shared<ov::op::v3::ShapeOf>(topk->output(1));  // [2]{32, 8}
        auto gather_idx = op::v0::Constant::create(element::i64, Shape{}, {0});
        auto gather_axis = op::v0::Constant::create(element::i64, Shape{}, {0});
        auto gather = std::make_shared<ov::op::v8::Gather>(shape_of, gather_idx, gather_axis); // scalar: 32
        auto const_unsqueeze = op::v0::Constant::create(element::i64, Shape{1}, {0});
        auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(gather, const_unsqueeze);  // [1]{32}

        // 128
        auto const0 = op::v0::Constant::create(element::i64, Shape{}, {128});
        auto const1 = op::v0::Constant::create(element::i64, Shape{1}, {0});
        auto unsqueeze1 = std::make_shared<ov::op::v0::Unsqueeze>(const0, const1);  // [1]{128}
        auto concat = std::make_shared<ov::op::v0::Concat>(OutputVector{unsqueeze, unsqueeze1}, 0);  // [2]{32,128}
        auto const3 = op::v0::Constant::create(element::i64, Shape{1}, {1});
        auto concat1 = std::make_shared<ov::op::v0::Concat>(OutputVector{unsqueeze1, unsqueeze, const3}, 0);

        // [32, 128]
        auto zero = op::v0::Constant::create(element::f16, Shape{1}, {0});
        auto bc = std::make_shared<ov::op::v3::Broadcast>(zero, concat);
        auto scatter_axis = op::v0::Constant::create(element::i64, Shape{1}, {1});
        auto scatter = std::make_shared<ov::op::v12::ScatterElementsUpdate>(bc,                  // [32, 128]
                                                                            topk->output(1),     // [32, 8]
                                                                            norm,                // [32, 8]
                                                                            scatter_axis,        // [1]
                                                                            ov::op::v12::ScatterElementsUpdate::Reduction::SUM);
        auto transpose_shape = op::v0::Constant::create(element::i64, Shape{2}, {1, 0});
        auto transpose = std::make_shared<ov::op::v1::Transpose>(scatter, transpose_shape);  // [128, 32]
        auto reshape = std::make_shared<ov::op::v1::Reshape>(transpose, concat1, false);
        auto unsqueeze_const = op::v0::Constant::create(element::i64, Shape{1}, {3});
        auto unsqueeze_moe = std::make_shared<ov::op::v0::Unsqueeze>(reshape, unsqueeze_const); // [128, 1, 32, 1]

        // weight
        auto wei_gate = op::v0::Constant::create(element::u4, Shape{128, 768, 16, 128}, {1});
        auto scale_gate = op::v0::Constant::create(element::f16, Shape{128, 16, 768}, {0.01f});
        auto zp_gate = op::v0::Constant::create(element::u4, Shape{128, 16, 768}, {0});
        auto wei_up = op::v0::Constant::create(element::u4, Shape{128, 768, 16, 128}, {1});
        auto scale_up = op::v0::Constant::create(element::f16, Shape{128, 16, 768}, {0.01f});
        auto zp_up = op::v0::Constant::create(element::u4, Shape{128, 16, 768, 16}, {0});
        auto wei_down = op::v0::Constant::create(element::u4, Shape{128, 2048, 6, 128}, {1});
        auto scale_down = op::v0::Constant::create(element::f16, Shape{128, 6, 2048}, {0.01f});
        auto zp_down = op::v0::Constant::create(element::u4, Shape{128, 6, 2048}, {0});

        ov::intel_gpu::op::MOECompressed::Config config;
        config.hidden_size = 2048;
        config.inter_size = 768;
        config.num_expert = 128;
        config.group_size = 128;
        config.top_k = 8;
        config.out_type = ov::element::f16;
        auto moe_compressed = std::make_shared<ov::intel_gpu::op::MOECompressed>(
            ov::OutputVector{hidden_states, unsqueeze_moe, topk->output(1),
                wei_gate, scale_gate, zp_gate, wei_up, scale_up, zp_up, wei_down, scale_down, zp_down}, config);
        model = std::make_shared<ov::Model>(moe_compressed, ov::ParameterVector{hidden_states});
        manager.register_pass<FuseMOE3GemmCompressed>();
    }
    {
        // tokens:32, hidden_size:2048, iter_size:768, experts:128, topk:8
        auto hidden_states = std::make_shared<ov::op::v0::Parameter>(element::f16, Shape{32, 2048});
        auto routers = op::v0::Constant::create(element::f16, Shape{2048, 128}, {0.2});
        auto routing_weights = std::make_shared<ov::op::v0::MatMul>(hidden_states, routers);

        // weight
        auto wei_gate = op::v0::Constant::create(element::u4, Shape{128, 768, 16, 128}, {1});
        auto scale_gate = op::v0::Constant::create(element::f16, Shape{128, 16, 768}, {0.01f});
        auto zp_gate = op::v0::Constant::create(element::u4, Shape{128, 16, 768}, {0});
        auto wei_up = op::v0::Constant::create(element::u4, Shape{128, 768, 16, 128}, {1});
        auto scale_up = op::v0::Constant::create(element::f16, Shape{128, 16, 768}, {0.01f});
        auto zp_up = op::v0::Constant::create(element::u4, Shape{128, 16, 768, 16}, {0});
        auto wei_down = op::v0::Constant::create(element::u4, Shape{128, 2048, 6, 128}, {1});
        auto scale_down = op::v0::Constant::create(element::f16, Shape{128, 6, 2048}, {0.01f});
        auto zp_down = op::v0::Constant::create(element::u4, Shape{128, 6, 2048}, {0});

        ov::intel_gpu::op::MOECompressed::Config config;
        config.hidden_size = 2048;
        config.inter_size = 768;
        config.num_expert = 128;
        config.group_size = 128;
        config.top_k = 8;
        config.out_type = ov::element::f16;
        auto moe_3gemm_fused_compressed = std::make_shared<ov::intel_gpu::op::MOE3GemmFusedCompressed>(
            ov::OutputVector{hidden_states, routing_weights,
                wei_gate, scale_gate, zp_gate, wei_up, scale_up, zp_up, wei_down, scale_down, zp_down}, config);

        model_ref = std::make_shared<ov::Model>(moe_3gemm_fused_compressed, ov::ParameterVector{hidden_states});
    }
}
}  // namespace intel_gpu
}  // namespace test
}  // namespace ov
