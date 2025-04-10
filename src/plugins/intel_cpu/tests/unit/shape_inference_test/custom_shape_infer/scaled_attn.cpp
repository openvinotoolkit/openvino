// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "custom_shape_infer.hpp"
#include "openvino/op/ops.hpp"
#include "transformations/cpu_opset/common/op/sdpa.hpp"

namespace ov {
namespace intel_cpu {
namespace unit_test {
namespace cpu_shape_infer {
using namespace ov;
using namespace ov::intel_cpu;
using namespace testing;

using SDPATestParams = std::tuple<unit_test::ShapeVector,  // Input shapes
                                  std::vector<size_t>,     // permute_axes
                                  unit_test::ShapeVector   // Expected output shapes
                                  >;

class SDPACpuShapeInferenceTest
    : public unit_test::OpCpuShapeInferenceTest<ov::intel_cpu::ScaledDotProductAttentionWithKVCache>,
      public WithParamInterface<SDPATestParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<SDPATestParams>& obj) {
        unit_test::ShapeVector tmp_input_shapes;
        std::vector<size_t> tmp_permute_axes;
        unit_test::ShapeVector tmp_exp_shape;
        std::tie(tmp_input_shapes, tmp_permute_axes, tmp_exp_shape) = obj.param;
        std::ostringstream result;
        result << "IS" << ov::test::utils::vec2str(tmp_input_shapes) << "_";
        result << "permute_axes" << ov::test::utils::vec2str(tmp_permute_axes) << "_";
        result << "exp_shape" << ov::test::utils::vec2str(tmp_exp_shape);
        return result.str();
    }

protected:
    void SetUp() override {
        std::tie(input_shapes, permute_axes, output_shapes) = GetParam();

        args.clear();
        for (const auto& ishape : input_shapes) {
            args.push_back(std::make_shared<op::v0::Parameter>(element::f32, ishape.get_shape()));
        }
    }
    OutputVector args;
    std::vector<size_t> permute_axes;
};

TEST_P(SDPACpuShapeInferenceTest, shape_inference) {
    ov::intel_cpu::ScaledDotProductAttentionWithKVCache::Config config;
    config.permute_axes = permute_axes;
    const auto op = make_op(args, config);
    unit_test::cpu_test_shape_infer(op.get(), input_shapes, output_shapes);
}

INSTANTIATE_TEST_SUITE_P(CpuShapeInfer,
                         SDPACpuShapeInferenceTest,
                         Values(
                             // llama
                             make_tuple(unit_test::ShapeVector{{1, 32, 14, 128},
                                                               {1, 32, 14, 128},
                                                               {1, 32, 14, 128},
                                                               {1, 1, 14, 14},
                                                               {1},
                                                               {1, 32, 0, 128},
                                                               {1, 32, 0, 128}},
                                        std::vector<size_t>{},
                                        unit_test::ShapeVector{{1, 32, 14, 128}, {1, 32, 14, 128}, {1, 32, 14, 128}}),
                             make_tuple(unit_test::ShapeVector{{1, 32, 1, 128},
                                                               {1, 32, 1, 128},
                                                               {1, 32, 1, 128},
                                                               {1, 1, 1, 16},
                                                               {1},
                                                               {1, 32, 15, 128},
                                                               {1, 32, 15, 128}},
                                        std::vector<size_t>{},
                                        unit_test::ShapeVector{{1, 32, 1, 128}, {1, 32, 16, 128}, {1, 32, 16, 128}}),
                             // chatglm
                             make_tuple(unit_test::ShapeVector{{1, 1, 32, 128},
                                                               {1, 1, 2, 128},
                                                               {1, 1, 2, 128},
                                                               {1, 1, 1, 8},
                                                               {1},
                                                               {7, 1, 2, 128},
                                                               {7, 1, 2, 128}},
                                        std::vector<size_t>{1, 2, 0, 3},
                                        unit_test::ShapeVector{{1, 32, 1, 128}, {8, 1, 2, 128}, {8, 1, 2, 128}}),
                             make_tuple(unit_test::ShapeVector{{7, 1, 32, 128},
                                                               {7, 1, 2, 128},
                                                               {7, 1, 2, 128},
                                                               {1, 1, 7, 7},
                                                               {1},
                                                               {0, 1, 2, 128},
                                                               {0, 1, 2, 128}},
                                        std::vector<size_t>{1, 2, 0, 3},
                                        unit_test::ShapeVector{{1, 32, 7, 128}, {7, 1, 2, 128}, {7, 1, 2, 128}}),
                             // qwen
                             make_tuple(unit_test::ShapeVector{{1, 1, 32, 128},
                                                               {1, 1, 32, 128},
                                                               {1, 1, 32, 128},
                                                               {1, 1, 1, 5},
                                                               {1},
                                                               {1, 4, 32, 128},
                                                               {1, 4, 32, 128}},
                                        std::vector<size_t>{0, 2, 1, 3},
                                        unit_test::ShapeVector{{1, 32, 1, 128}, {1, 5, 32, 128}, {1, 5, 32, 128}}),

                             make_tuple(unit_test::ShapeVector{{1, 4, 32, 128},
                                                               {1, 4, 32, 128},
                                                               {1, 4, 32, 128},
                                                               {1, 1, 4, 4},
                                                               {1},
                                                               {1, 0, 32, 128},
                                                               {1, 0, 32, 128}},
                                        std::vector<size_t>{0, 2, 1, 3},
                                        unit_test::ShapeVector{{1, 32, 4, 128}, {1, 4, 32, 128}, {1, 4, 32, 128}})),
                         SDPACpuShapeInferenceTest::getTestCaseName);

}  // namespace cpu_shape_infer
}  // namespace unit_test
}  // namespace intel_cpu
}  // namespace ov
