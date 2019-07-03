// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <string.h>
#include <transform/transform_network.hpp>
#include <transform/transformations/sub.hpp>
#include <ie_builders.hpp>

#include "tranformations_test.hpp"

using namespace testing;
using namespace InferenceEngine;

class TransformNetworkTest: public TransformationTestCommon {};

TEST_F(TransformationTestCommon, Sub) {
    Builder::Network builder("sub");

    idx_t firstInputId = builder.addLayer(Builder::InputLayer("FirstInput").setPort(Port({1,3, 227, 227})));
    idx_t secondInputId = builder.addLayer(Builder::InputLayer("SecondInput").setPort(Port({1,3, 227, 227})));
    idx_t eltwiseSubId = builder.addLayer({firstInputId, secondInputId}, Builder::EltwiseLayer("Sub").setEltwiseType(Builder::EltwiseLayer::EltwiseType::SUB));
    idx_t clampId = builder.addLayer({eltwiseSubId}, Builder::ClampLayer("clamp"));
    auto network = Transform::Network(builder);

    Transform::TransformationSub transformationSub;
    transformationSub.execute(network);
    ASSERT_THROW(network.getLayer("Sub"), InferenceEngine::details::InferenceEngineException);
    auto sumLayer = network.getLayer(firstInputId).getOutPort().getConnection().getDestination().getLayer();
    auto powerLayer = network.getLayer(secondInputId).getOutPort().getConnection().getDestination().getLayer();
    ASSERT_EQ(sumLayer.getType(), "Eltwise");
    ASSERT_EQ(sumLayer.getParameter("operation").as<std::string>(), "sum");
    ASSERT_EQ(powerLayer.getType(), "Power");
    ASSERT_EQ(powerLayer.getParameter("power").as<float>(), 1.0f);
    ASSERT_EQ(powerLayer.getParameter("scale").as<float>(), -1.0f);
    ASSERT_EQ(powerLayer.getParameter("shift").as<float>(), 0.0f);
    ASSERT_EQ(sumLayer.getOutPort().getConnection().getDestination().getLayer().getId(), clampId);
}