#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "intel_gpu/op/moe_compressed.hpp"
#include "intel_gpu/op/placeholder.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/moe.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/variadic_split.hpp"
#include "plugin/transformations/convert_moe_to_compressed.hpp"

using namespace testing;
using namespace ov::intel_gpu;

namespace ov {
namespace test {
namespace intel_gpu {
TEST_F(TransformationTestsF, MoeCompressedTest) {
    {
        // Construct inputs
        auto hidden_states = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 2048});
        auto routing_weights = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 4});
        auto topk = std::make_shared<ov::op::v0::Parameter>(element::i32, Shape{2, 2});

        // Construct constant weights
        auto w0 = op::v0::Constant::create(element::u4, Shape{4, 768, 16, 128}, {1});
        auto zp0 = op::v0::Constant::create(element::u4, Shape{4, 768, 16, 1}, {0});
        auto scale0 = op::v0::Constant::create(element::f16, Shape{4, 768, 16, 1}, {0.01f});
        auto reshape0 = op::v0::Constant::create(element::i64, Shape{3}, {4, 768, 2048});

        // First projection
        auto w0_f16 = std::make_shared<op::v0::Convert>(w0, element::f16);
        auto zp0_f16 = std::make_shared<op::v0::Convert>(zp0, element::f16);
        auto sub0 = std::make_shared<op::v1::Subtract>(w0_f16, zp0_f16);
        auto mul0 = std::make_shared<op::v1::Multiply>(sub0, scale0);
        auto reshape_m0 = std::make_shared<op::v1::Reshape>(mul0, reshape0, false);
        auto convert_m0 = std::make_shared<op::v0::Convert>(reshape_m0, element::f32);

        // Second projection
        auto w1 = op::v0::Constant::create(element::u4, Shape{4, 768, 16, 128}, {1});
        auto zp1 = op::v0::Constant::create(element::u4, Shape{4, 768, 16, 1}, {0});
        auto scale1 = op::v0::Constant::create(element::f16, Shape{4, 768, 16, 1}, {0.01f});
        auto reshape1 = op::v0::Constant::create(element::i64, Shape{3}, {4, 768, 2048});

        auto w1_f16 = std::make_shared<op::v0::Convert>(w1, element::f16);
        auto zp1_f16 = std::make_shared<op::v0::Convert>(zp1, element::f16);
        auto sub1 = std::make_shared<op::v1::Subtract>(w1_f16, zp1_f16);
        auto mul1 = std::make_shared<op::v1::Multiply>(sub1, scale1);
        auto reshape_m1 = std::make_shared<op::v1::Reshape>(mul1, reshape1, false);
        auto convert_m1 = std::make_shared<op::v0::Convert>(reshape_m1, element::f32);

        // Third projection
        auto w2 = op::v0::Constant::create(element::u4, Shape{4, 768, 16, 128}, {1});
        auto zp2 = op::v0::Constant::create(element::u4, Shape{4, 768, 16, 1}, {0});
        auto scale2 = op::v0::Constant::create(element::f16, Shape{4, 768, 16, 1}, {0.01f});
        auto reshape2 = op::v0::Constant::create(element::i64, Shape{3}, {4, 768, 2048});

        auto w2_f16 = std::make_shared<op::v0::Convert>(w2, element::f16);
        auto zp2_f16 = std::make_shared<op::v0::Convert>(zp2, element::f16);
        auto sub2 = std::make_shared<op::v1::Subtract>(w2_f16, zp2_f16);
        auto mul2 = std::make_shared<op::v1::Multiply>(sub2, scale2);
        auto reshape_m2 = std::make_shared<op::v1::Reshape>(mul2, reshape2, false);
        auto convert_m2 = std::make_shared<op::v0::Convert>(reshape_m2, element::f32);

        // Construct MOE node
        ov::op::internal::MOE::Config config;
        config.expert_type = ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU;
        auto moe = std::make_shared<ov::op::internal::MOE>(ov::OutputVector{hidden_states, routing_weights, topk, convert_m0, convert_m1, convert_m2}, config);
        model = std::make_shared<ov::Model>(moe, ov::ParameterVector{hidden_states, routing_weights, topk});
        manager.register_pass<ConvertMOEToMOECompressed>();
    }
    {
        // Construct inputs
        auto hidden_states = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 2048});
        auto routing_weights = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 4});
        auto topk = std::make_shared<ov::op::v0::Parameter>(element::i32, Shape{2, 2});

        // Construct constant weights
        auto w0 = op::v0::Constant::create(element::u4, Shape{4, 768, 16, 128}, {1});
        auto zp0 = op::v0::Constant::create(element::u4, Shape{4, 768, 16, 1}, {0});
        auto scale0 = op::v0::Constant::create(element::f16, Shape{4, 768, 16, 1}, {0.01f});

        // Second projection
        auto w1 = op::v0::Constant::create(element::u4, Shape{4, 768, 16, 128}, {1});
        auto zp1 = op::v0::Constant::create(element::u4, Shape{4, 768, 16, 1}, {0});
        auto scale1 = op::v0::Constant::create(element::f16, Shape{4, 768, 16, 1}, {0.01f});

        // Third projection
        auto w2 = op::v0::Constant::create(element::u4, Shape{4, 768, 16, 128}, {1});
        auto zp2 = op::v0::Constant::create(element::u4, Shape{4, 768, 16, 1}, {0});
        auto scale2 = op::v0::Constant::create(element::f16, Shape{4, 768, 16, 1}, {0.01f});

        ov::intel_gpu::op::MOECompressed::Config config;
        config.hidden_size = 2048;
        config.inter_size = 768;
        config.num_expert = 4;
        config.group_size = 128;
        config.top_k = 2;
        config.out_type = ov::element::f16;
        auto moe_compressed = std::make_shared<ov::intel_gpu::op::MOECompressed>(
            ov::OutputVector{hidden_states, routing_weights, topk, w0, scale0, zp0, w1, scale1, zp1, w2, scale2, zp2},
            config);
        model_ref = std::make_shared<ov::Model>(moe_compressed, ov::ParameterVector{hidden_states, routing_weights, topk});
    }
}
}  // namespace intel_gpu
}  // namespace test
}  // namespace ov