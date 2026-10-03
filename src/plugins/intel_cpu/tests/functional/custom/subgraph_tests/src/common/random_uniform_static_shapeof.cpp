// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/constant.hpp"
#include "openvino/op/gather_elements.hpp"
#include "openvino/op/loop.hpp"
#include "openvino/op/random_uniform.hpp"
#include "openvino/op/shape_of.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

// RandomUniform with a static output shape coming from ShapeOf inside a constant Loop.
// GatherElements has no evaluate, so the Loop is not folded and runs as a CPU constant node.
class RandomUniformStaticShapeOf : public SubgraphBaseTest {
protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;

        auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{1, 1});
        auto shape = std::make_shared<ov::op::v3::ShapeOf>(param, ov::element::i32);
        auto min_val = ov::op::v0::Constant::create(ov::element::i32, {}, {0});
        auto max_val = ov::op::v0::Constant::create(ov::element::i32, {}, {1});
        auto indices = std::make_shared<ov::op::v8::RandomUniform>(shape, min_val, max_val, ov::element::i32, 1, 2);
        auto data = ov::op::v0::Constant::create(ov::element::i32, {1, 4}, {5, 6, 7, 8});
        auto gather = std::make_shared<ov::op::v6::GatherElements>(data, indices, 1);
        auto body_cond = ov::op::v0::Constant::create(ov::element::boolean, {}, {true});
        auto body = std::make_shared<ov::Model>(ov::OutputVector{body_cond, gather}, ov::ParameterVector{param});

        auto trip_count = ov::op::v0::Constant::create(ov::element::i32, {}, {2});
        auto exec_cond = ov::op::v0::Constant::create(ov::element::boolean, {}, {true});
        auto loop = std::make_shared<ov::op::v5::Loop>(trip_count, exec_cond);
        loop->set_function(body);
        loop->set_special_body_ports({-1, 0});
        auto init = ov::op::v0::Constant::create(ov::element::i32, {1, 1}, {0});
        loop->set_merged_input(param, init, gather);

        function = std::make_shared<ov::Model>(ov::OutputVector{loop->get_iter_value(gather)},
                                               ov::ParameterVector{},
                                               "RandomUniformStaticShapeOf");
    }
};

TEST_F(RandomUniformStaticShapeOf, smoke_CompareWithRefs) {
    run();
}

}  // namespace test
}  // namespace ov
