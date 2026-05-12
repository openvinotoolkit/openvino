// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
// Regression test for issue #35798: int8 Add/Multiply on the CPU plugin must
// wrap (two's-complement) like the OV reference, not saturate to [-128, 127].

#include <cstdint>
#include <vector>

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

class EltwiseInt8WrapTest : public SubgraphBaseStaticTest {
public:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;

        const ov::Shape shape{3};
        auto pa = std::make_shared<ov::op::v0::Parameter>(ov::element::i8, shape);
        auto pb = std::make_shared<ov::op::v0::Parameter>(ov::element::i8, shape);
        auto add = std::make_shared<ov::op::v1::Add>(pa, pb);
        auto mul = std::make_shared<ov::op::v1::Multiply>(pa, pb);

        ov::ResultVector results{std::make_shared<ov::op::v0::Result>(add),
                                 std::make_shared<ov::op::v0::Result>(mul)};
        function = std::make_shared<ov::Model>(results, ov::ParameterVector{pa, pb}, "EltwiseInt8Wrap");
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const std::vector<int8_t> a{100, -110, 60};
        const std::vector<int8_t> b{80, -50, 90};

        const auto& funcInputs = function->inputs();
        OPENVINO_ASSERT(funcInputs.size() == 2);
        OPENVINO_ASSERT(targetInputStaticShapes[0] == ov::Shape{3});
        OPENVINO_ASSERT(targetInputStaticShapes[1] == ov::Shape{3});

        ov::Tensor ta(ov::element::i8, targetInputStaticShapes[0]);
        ov::Tensor tb(ov::element::i8, targetInputStaticShapes[1]);
        std::copy(a.begin(), a.end(), ta.data<int8_t>());
        std::copy(b.begin(), b.end(), tb.data<int8_t>());

        inputs.insert({funcInputs[0].get_node_shared_ptr(), ta});
        inputs.insert({funcInputs[1].get_node_shared_ptr(), tb});
    }
};

TEST_F(EltwiseInt8WrapTest, smoke_CompareWithRefs) {
    run();
}

}  // namespace test
}  // namespace ov
