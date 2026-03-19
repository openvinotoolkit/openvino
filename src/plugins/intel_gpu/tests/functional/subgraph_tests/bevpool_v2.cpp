// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"

#include "openvino/op/bevpool_v2.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"

namespace {

class BevPoolV2SubgraphTest : virtual public ov::test::SubgraphBaseStaticTest {
protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;

        const auto cf = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 1, 2, 2});
        const auto dw = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 2, 2, 2});
        const auto idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{5});
        const auto itv = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{6});

        const ov::op::v15::Bound x_bound{-10.f, 10.f, 0.5f};
        const ov::op::v15::Bound y_bound{-10.f, 10.f, 0.5f};
        const ov::op::v15::Bound z_bound{-5.f, 3.f, 0.5f};
        const ov::op::v15::Bound d_bound{0.f, 2.f, 1.f};

        const auto bevpool = std::make_shared<ov::op::v15::BevPoolV2>(ov::OutputVector{cf, dw, idx, itv},
                                                                       1,
                                                                       1,
                                                                       2,
                                                                       2,
                                                                       1,
                                                                       2,
                                                                       x_bound,
                                                                       y_bound,
                                                                       z_bound,
                                                                       d_bound);

        function = std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(bevpool)},
                               ov::ParameterVector{cf, dw, idx, itv});
    }
};

TEST_F(BevPoolV2SubgraphTest, Inference) {
    run();
}

}  // namespace
