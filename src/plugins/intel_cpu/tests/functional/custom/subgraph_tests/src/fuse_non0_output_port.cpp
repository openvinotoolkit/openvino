// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

class FuseNon0OuputPort : public SubgraphBaseTest {
    void SetUp() override {
        const ov::Shape x_shape = {1, 10};
        const ov::Shape y_shape = {1};
        const ov::Shape z_shape = {1};
        ov::ParameterVector params(3);
        targetStaticShapes = {{x_shape, y_shape, z_shape}};
        targetDevice = ov::test::utils::DEVICE_CPU;
        params[0] = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, x_shape);
        params[1] = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, y_shape);
        params[2] = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, z_shape);

        // make a sub function
        const auto cond = ov::op::v0::Constant::create(ov::element::boolean, {1}, {true});
        ov::ParameterVector sub_params(3);
        sub_params[0] = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, x_shape);
        sub_params[1] = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, y_shape);
        sub_params[2] = std::make_shared<ov::op::v0::Parameter>(ov::element::boolean, y_shape);
        ov::ResultVector sub_results(3);
        sub_results[0] = std::make_shared<ov::op::v0::Result>(sub_params[0]);
        sub_results[1] = std::make_shared<ov::op::v0::Result>(sub_params[1]);
        sub_results[2] = std::make_shared<ov::op::v0::Result>(sub_params[2]);
        const auto sub_model = std::make_shared<ov::Model>(sub_results, sub_params);

        // loop ops
        const auto trip = ov::op::v0::Constant::create(ov::element::i64, {1}, {2});
        const auto loop = std::make_shared<ov::op::v5::Loop>(trip, cond);
        loop->set_function(sub_model);
        loop->set_invariant_input(sub_params[0], params[0]);
        loop->set_invariant_input(sub_params[1], params[1]);
        loop->set_invariant_input(sub_params[2], cond);
        loop->set_special_body_ports({-1, 2});
        const auto out0 = loop->get_iter_value(sub_results[0]->output(0), -1);
        const auto out1 = loop->get_iter_value(sub_results[1]->output(0), -1);
        const auto out2 = loop->get_iter_value(sub_results[2]->output(0), -1);

        // main function
        const auto c = ov::op::v0::Constant::create(ov::element::i32, {1}, {1});
        const auto z1 = std::make_shared<ov::op::v1::Add>(params[2], c);
        const auto d = std::make_shared<ov::op::v1::Add>(out1, z1);
        function = std::make_shared<ov::Model>(ov::OutputVector{d->output(0), out0, out2}, params, "FuseNon0OuputPort");
    }
};

TEST_F(FuseNon0OuputPort, smoke_FuseNon0OuputPort) {
    run();
}

}  // namespace test
}  // namespace ov
