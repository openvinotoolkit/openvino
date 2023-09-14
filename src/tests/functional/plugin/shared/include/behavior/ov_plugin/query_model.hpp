// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "base/ov_behavior_test_utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/shape_of.hpp"

namespace ov {
namespace test {
namespace behavior {

using OVClassQueryModelTest = OVClassBaseTestP;

TEST_P(OVClassModelTestP, QueryModelActualNoThrow) {
    ov::Core ie = createCoreWithTemplate();
    ie.query_model(actualNetwork, target_device);
}

TEST_P(OVClassModelTestP, QueryModelWithKSO) {
    ov::Core ie = createCoreWithTemplate();

    auto rl_map = ie.query_model(ksoNetwork, target_device);
    auto func = ksoNetwork;
    for (const auto& op : func->get_ops()) {
        if (!rl_map.count(op->get_friendly_name())) {
            FAIL() << "Op " << op->get_friendly_name() << " is not supported by " << target_device;
        }
    }
}

TEST_P(OVClassQueryModelTest, QueryModelWithMatMul) {
    ov::Core ie = createCoreWithTemplate();

    std::shared_ptr<ov::Model> func;
    {
        ov::PartialShape shape({1, 84});
        ov::element::Type type(ov::element::Type_t::f32);
        auto param = std::make_shared<ov::op::v0::Parameter>(type, shape);
        auto matMulWeights = ov::op::v0::Constant::create(ov::element::Type_t::f32, {10, 84}, {1});
        auto shapeOf = std::make_shared<ov::op::v3::ShapeOf>(matMulWeights);
        auto gConst1 = ov::op::v0::Constant::create(ov::element::Type_t::i32, {1}, {1});
        auto gConst2 = ov::op::v0::Constant::create(ov::element::Type_t::i64, {}, {0});
        auto gather = std::make_shared<ov::op::v1::Gather>(shapeOf, gConst1, gConst2);
        auto concatConst = ov::op::v0::Constant::create(ov::element::Type_t::i64, {1}, {1});
        auto concat = std::make_shared<ov::op::v0::Concat>(ov::NodeVector{concatConst, gather}, 0);
        auto relu = std::make_shared<ov::op::v0::Relu>(param);
        auto reshape = std::make_shared<ov::op::v1::Reshape>(relu, concat, false);
        auto matMul = std::make_shared<ov::op::v0::MatMul>(reshape, matMulWeights, false, true);
        auto matMulBias = ov::op::v0::Constant::create(ov::element::Type_t::f32, {1, 10}, {1});
        auto addBias = std::make_shared<ov::op::v1::Add>(matMul, matMulBias);
        auto result = std::make_shared<ov::op::v0::Result>(addBias);

        ov::ParameterVector params = {param};
        ov::ResultVector results = {result};

        func = std::make_shared<ov::Model>(results, params);
    }

    auto rl_map = ie.query_model(func, target_device);
    for (const auto& op : func->get_ops()) {
        if (!rl_map.count(op->get_friendly_name())) {
            FAIL() << "Op " << op->get_friendly_name() << " is not supported by " << target_device;
        }
    }
}

TEST_P(OVClassQueryModelTest, QueryModelHETEROWithDeviceIDNoThrow) {
    ov::Core ie = createCoreWithTemplate();

    auto deviceIDs = ie.get_property(target_device, ov::available_devices);
    if (deviceIDs.empty())
        GTEST_FAIL();
    OV_ASSERT_NO_THROW(ie.query_model(actualNetwork,
                                      ov::test::utils::DEVICE_HETERO,
                                      ov::device::priorities(target_device + "." + deviceIDs[0], target_device)));
}

TEST_P(OVClassQueryModelTest, QueryModelWithBigDeviceIDThrows) {
    ov::Core ie = createCoreWithTemplate();
    ASSERT_THROW(ie.query_model(actualNetwork, target_device + ".110"), ov::Exception);
}

TEST_P(OVClassQueryModelTest, QueryModelWithInvalidDeviceIDThrows) {
    ov::Core ie = createCoreWithTemplate();
    ASSERT_THROW(ie.query_model(actualNetwork, target_device + ".l0"), ov::Exception);
}

}  // namespace behavior
}  // namespace test
}  // namespace ov
