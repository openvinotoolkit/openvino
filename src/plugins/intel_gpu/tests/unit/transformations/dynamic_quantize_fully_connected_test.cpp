// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "intel_gpu/op/fully_connected_compressed.hpp"
#include "intel_gpu/op/placeholder.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "ov_ops/dynamic_quantize.hpp"
#include "plugin/transformations/dynamic_quantize_fully_connected.hpp"

using namespace testing;
using namespace ov::intel_gpu;

namespace ov {
namespace test {
namespace intel_gpu {

using DynamicQuantize = ov::op::internal::DynamicQuantize;

struct DynamicQuantizeFCParams {
    ov::element::Type weights_type;
    ov::element::Type zp_type;
    bool expect_precomputed_reduction;
};

class DynamicQuantizeFullyConnectedTests : public TransformationTestsF, public WithParamInterface<DynamicQuantizeFCParams> {
public:
    static std::string get_test_case_name(const testing::TestParamInfo<DynamicQuantizeFCParams>& obj) {
        const auto& p = obj.param;
        std::ostringstream result;
        result << "weights=" << p.weights_type << "_zp=" << p.zp_type << "_precomputed_reduction=" << p.expect_precomputed_reduction;
        return result.str();
    }

protected:
    // Dynamic quantization group size (128) is larger than the weights group size (64):
    // with precomputed reduction it is adjusted to the weights group size, without it it is kept as is.
    static constexpr size_t K = 1024;
    static constexpr size_t N = 256;
    static constexpr size_t wei_group_size = 64;
    static constexpr uint64_t dyn_quan_group_size = 128;

    void SetUp() override {
        TransformationTestsF::SetUp();
        const auto& p = GetParam();
        const ov::Shape decompression_shape{N, K / wei_group_size};
        auto make_weights = [&]() {
            return ov::OutputVector{std::make_shared<ov::op::v0::Constant>(p.weights_type, ov::Shape{N, K}),
                                    std::make_shared<ov::intel_gpu::op::Placeholder>(),
                                    std::make_shared<ov::op::v0::Constant>(ov::element::f16, decompression_shape),
                                    std::make_shared<ov::op::v0::Constant>(p.zp_type, decompression_shape)};
        };
        {
            auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, -1, K});
            const auto w = make_weights();
            auto fc = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input, w[0], w[1], w[2], w[3]);
            model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input});
            manager.register_pass<DynamicQuantizeFullyConnected>(dyn_quan_group_size, false, true);
        }
        {
            auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, -1, K});
            const auto w = make_weights();
            DynamicQuantize::Attributes attrs;
            attrs.quantization_type = DynamicQuantize::QuantizationType::Symmetric;
            attrs.quantization_dt = ov::element::i8;
            attrs.scale_dt = ov::element::f16;
            attrs.group_sizes = {1, 1, p.expect_precomputed_reduction ? wei_group_size : dyn_quan_group_size};
            if (p.expect_precomputed_reduction) {
                attrs.precomputed_reduction = true;
                attrs.precomputed_reduction_dt = ov::element::i32;
            }
            auto dq = std::make_shared<DynamicQuantize>(input, attrs);
            auto a_zp = std::make_shared<ov::intel_gpu::op::Placeholder>();
            auto precomputed_reduction = p.expect_precomputed_reduction ? dq->output(2) : std::make_shared<ov::intel_gpu::op::Placeholder>()->output(0);
            auto fc = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(dq->output(0),
                                                                                    w[0],
                                                                                    w[1],
                                                                                    w[2],
                                                                                    w[3],
                                                                                    dq->output(1),
                                                                                    a_zp,
                                                                                    precomputed_reduction,
                                                                                    ov::element::f16);
            model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input});
        }
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
};

TEST_P(DynamicQuantizeFullyConnectedTests, CompareFunctions) {}

// Precomputed reduction is attached only to 8-bit weights with static zero point
const std::vector<DynamicQuantizeFCParams> dynamic_quantize_fc_params = {
    {ov::element::u8, ov::element::u8, true},
    {ov::element::i8, ov::element::u8, true},
    {ov::element::u4, ov::element::u4, false},
    {ov::element::i4, ov::element::u4, false},
    {ov::element::u3, ov::element::u8, false},
    {ov::element::u2, ov::element::u8, false},
};

INSTANTIATE_TEST_SUITE_P(smoke_TransformationTests_PrecomputedReduction,
                         DynamicQuantizeFullyConnectedTests,
                         ::testing::ValuesIn(dynamic_quantize_fc_params),
                         DynamicQuantizeFullyConnectedTests::get_test_case_name);

}  // namespace intel_gpu
}  // namespace test
}  // namespace ov
