// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "functional_test_utils/skip_tests_config.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/strided_slice.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

class StridedSliceReshapeConcatFusionCPUTest
    : virtual public SubgraphBaseTest,
      public CPUTestsBase {
public:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;

        const InputShape inputShape = {{1, 16}, {{1, 16}}};
        init_input_shapes({inputShape});

        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputDynamicShapes[0]);
        ov::OutputVector concat_inputs;

        for (const int64_t start : {0, 2, 4}) {
            auto begin = ov::op::v0::Constant::create<int64_t>(ov::element::i64, ov::Shape{2}, {0, start});
            auto end = ov::op::v0::Constant::create<int64_t>(ov::element::i64, ov::Shape{2}, {1, start + 4});
            auto strides = ov::op::v0::Constant::create<int64_t>(ov::element::i64, ov::Shape{2}, {1, 1});
            auto strided_slice = std::make_shared<ov::op::v1::StridedSlice>(input,
                                                                            begin,
                                                                            end,
                                                                            strides,
                                                                            std::vector<int64_t>{0, 0},
                                                                            std::vector<int64_t>{0, 0});
            auto shape = ov::op::v0::Constant::create<int64_t>(ov::element::i64, ov::Shape{3}, {1, 1, 4});
            auto reshape = std::make_shared<ov::op::v1::Reshape>(strided_slice, shape, false);
            concat_inputs.push_back(reshape);
        }

        auto concat = std::make_shared<ov::op::v0::Concat>(concat_inputs, 1);
        function = std::make_shared<ov::Model>(ov::OutputVector{concat}, ov::ParameterVector{input});
    }

    void check_results() {
        CheckNumberOfNodesWithType(compiledModel, "Gather", 1);
    }
};

TEST_F(StridedSliceReshapeConcatFusionCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
    check_results();
}

}  // namespace test
}  // namespace ov
