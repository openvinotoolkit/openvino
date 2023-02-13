// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "base/ov_behavior_test_utils.hpp"
#include <openvino/runtime/properties.hpp>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/unicode_utils.hpp"
#include "openvino/util/file_util.hpp"

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
#    include <iostream>
#    define GTEST_COUT std::cerr << "[          ] [ INFO ] "
#    include <codecvt>
#    include <functional_test_utils/skip_tests_config.hpp>
#endif

namespace ov {
namespace test {
namespace behavior {

using OVClassNetworkTestP = OVClassBaseTestP;
using OVClassQueryNetworkTest = OVClassBaseTestP;

//
// QueryNetwork
//

TEST_P(OVClassNetworkTestP, QueryNetworkActualNoThrow) {
    ov::Core ie = createCoreWithTemplate();
    ie.query_model(actualNetwork, target_device);
}

TEST_P(OVClassNetworkTestP, QueryNetworkWithKSO) {
    ov::Core ie = createCoreWithTemplate();

    auto rl_map = ie.query_model(ksoNetwork, target_device);
    auto func = ksoNetwork;
    for (const auto& op : func->get_ops()) {
        if (!rl_map.count(op->get_friendly_name())) {
            FAIL() << "Op " << op->get_friendly_name() << " is not supported by " << target_device;
        }
    }
}

TEST_P(OVClassNetworkTestP, SetAffinityWithConstantBranches) {
    ov::Core ie = createCoreWithTemplate();

    std::shared_ptr<ngraph::Function> func;
    {
        ngraph::PartialShape shape({1, 84});
        ngraph::element::Type type(ngraph::element::Type_t::f32);
        auto param = std::make_shared<ngraph::opset6::Parameter>(type, shape);
        auto matMulWeights = ngraph::opset6::Constant::create(ngraph::element::Type_t::f32, {10, 84}, {1});
        auto shapeOf = std::make_shared<ngraph::opset6::ShapeOf>(matMulWeights);
        auto gConst1 = ngraph::opset6::Constant::create(ngraph::element::Type_t::i32, {1}, {1});
        auto gConst2 = ngraph::opset6::Constant::create(ngraph::element::Type_t::i64, {}, {0});
        auto gather = std::make_shared<ngraph::opset6::Gather>(shapeOf, gConst1, gConst2);
        auto concatConst = ngraph::opset6::Constant::create(ngraph::element::Type_t::i64, {1}, {1});
        auto concat = std::make_shared<ngraph::opset6::Concat>(ngraph::NodeVector{concatConst, gather}, 0);
        auto relu = std::make_shared<ngraph::opset6::Relu>(param);
        auto reshape = std::make_shared<ngraph::opset6::Reshape>(relu, concat, false);
        auto matMul = std::make_shared<ngraph::opset6::MatMul>(reshape, matMulWeights, false, true);
        auto matMulBias = ngraph::opset6::Constant::create(ngraph::element::Type_t::f32, {1, 10}, {1});
        auto addBias = std::make_shared<ngraph::opset6::Add>(matMul, matMulBias);
        auto result = std::make_shared<ngraph::opset6::Result>(addBias);

        ngraph::ParameterVector params = {param};
        ngraph::ResultVector results = {result};

        func = std::make_shared<ngraph::Function>(results, params);
    }

    auto rl_map = ie.query_model(func, target_device);
    for (const auto& op : func->get_ops()) {
        if (!rl_map.count(op->get_friendly_name())) {
            FAIL() << "Op " << op->get_friendly_name() << " is not supported by " << target_device;
        }
    }
    for (const auto& op : func->get_ops()) {
        std::string affinity = rl_map[op->get_friendly_name()];
        op->get_rt_info()["affinity"] = affinity;
    }
    auto exeNetwork = ie.compile_model(ksoNetwork, target_device);
}

TEST_P(OVClassNetworkTestP, SetAffinityWithKSO) {
    ov::Core ie = createCoreWithTemplate();

    auto rl_map = ie.query_model(ksoNetwork, target_device);
    auto func = ksoNetwork;
    for (const auto& op : func->get_ops()) {
        if (!rl_map.count(op->get_friendly_name())) {
            FAIL() << "Op " << op->get_friendly_name() << " is not supported by " << target_device;
        }
    }
    for (const auto& op : func->get_ops()) {
        std::string affinity = rl_map[op->get_friendly_name()];
        op->get_rt_info()["affinity"] = affinity;
    }
    auto exeNetwork = ie.compile_model(ksoNetwork, target_device);
}

TEST_P(OVClassQueryNetworkTest, QueryNetworkWithDeviceID) {
    ov::Core ie = createCoreWithTemplate();

    if (supportsDeviceID(ie, target_device)) {
        ie.query_model(simpleNetwork, target_device + ".0");
    } else {
        GTEST_FAIL() << "Device does not support DeviceID property" << std::endl;
    }
}

TEST_P(OVClassQueryNetworkTest, QueryNetworkWithBigDeviceIDThrows) {
    ov::Core ie = createCoreWithTemplate();

    if (supportsDeviceID(ie, target_device)) {
        ASSERT_THROW(ie.query_model(actualNetwork, target_device + ".110"), ov::Exception);
    } else {
        GTEST_FAIL() << "Device does not support DeviceID property" << std::endl;
    }
}

TEST_P(OVClassQueryNetworkTest, QueryNetworkWithInvalidDeviceIDThrows) {
    ov::Core ie = createCoreWithTemplate();

    if (supportsDeviceID(ie, target_device)) {
        ASSERT_THROW(ie.query_model(actualNetwork, target_device + ".l0"), ov::Exception);
    } else {
        GTEST_FAIL() << "Device does not support DeviceID property" << std::endl;
    }
}

}  // namespace behavior
}  // namespace test
}  // namespace ov
