// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/partial_shape.hpp"
#include "openvino/runtime/properties.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "common_test_utils/test_constants.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/op/add.hpp"

using namespace ov::test;

namespace ov {
namespace test {
namespace intel_gpu {

class SharedConstantGPUTest : public SubgraphBaseTest {
protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;

        std::vector<InputShape> input_shapes {
            {ov::PartialShape{1}, {{1}}},
            {ov::PartialShape{1}, {{1}}},
            {ov::PartialShape{2}, {{2}}},
            {ov::PartialShape{1, 2}, {{1, 2}}},
        };
        init_input_shapes(input_shapes);

        auto p0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shapes[0].first);
        auto p1 = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, input_shapes[1].first);
        auto p2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shapes[2].first);
        auto p3 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shapes[3].first);

        std::vector<int32_t> values0{0};
        std::vector<float> values1{0.0f, 1.0f};

        ov::Tensor t0(ov::element::f32, input_shapes[0].first.to_shape(), values0.data());
        ov::Tensor t1(ov::element::i32, input_shapes[1].first.to_shape(), values0.data());
        ov::Tensor t2(ov::element::f32, input_shapes[2].first.to_shape(), values1.data());
        ov::Tensor t3(ov::element::f32, input_shapes[3].first.to_shape(), values1.data());

        auto c0 = std::make_shared<ov::op::v0::Constant>(t0);
        auto c1 = std::make_shared<ov::op::v0::Constant>(t1);
        auto c2 = std::make_shared<ov::op::v0::Constant>(t2);
        auto c3 = std::make_shared<ov::op::v0::Constant>(t3);

        auto add0 = std::make_shared<ov::op::v1::Add>(p0, c0);
        auto add1 = std::make_shared<ov::op::v1::Add>(p1, c1);
        auto add2 = std::make_shared<ov::op::v1::Add>(p2, c2);
        auto add3 = std::make_shared<ov::op::v1::Add>(p3, c3);

        ov::ParameterVector params{p0, p1, p2, p3};
        ov::ResultVector results {
            std::make_shared<ov::op::v0::Result>(add0->output(0)),
            std::make_shared<ov::op::v0::Result>(add1->output(0)),
            std::make_shared<ov::op::v0::Result>(add2->output(0)),
            std::make_shared<ov::op::v0::Result>(add3->output(0))
        };
        function = std::make_shared<ov::Model>(results, params, "SharedConstantGPUTest");

        this->configuration.insert({ov::hint::inference_precision(ov::element::f32)});
    }
};

TEST_F(SharedConstantGPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    run();
}

} // namespace intel_gpu
} // namespace test
} // namespace ov
