// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <string.h>
#include <transform/transform_network.hpp>
#include <transform/transformations/eltwise_broadcast.hpp>
#include <ie_builders.hpp>

#include "tranformations_test.hpp"

using namespace testing;
using namespace InferenceEngine;

class TransformNetworkTest: public TransformationTestCommon {};

TEST_F(TransformationTestCommon, EltwiseBroadcastOneDimension) {
    Builder::Network builder("eltwiseBroadcast");

    idx_t firstInputId = builder.addLayer(Builder::InputLayer("FirstInput").setPort(Port({1, 3, 227, 1})));
    idx_t secondInputId = builder.addLayer(Builder::InputLayer("SecondInput").setPort(Port({1, 3, 227, 227})));
    idx_t eltwiseSumId = builder.addLayer({firstInputId, secondInputId}, Builder::EltwiseLayer("Sum").
                                                                         setEltwiseType(Builder::EltwiseLayer::EltwiseType::SUM).
                                                                         setOutputPort(Port({1, 3, 227, 227})));
    auto network = Transform::Network(builder);

    Transform::TransformationEltwiseBroadcast transformationEltwiseBroadcast;
    transformationEltwiseBroadcast.execute(network);
    auto firstInputLayer = network.getLayer(firstInputId);
    auto tileLayer = network.getLayer(firstInputId).getOutPort().getConnection().getDestination().getLayer();
    ASSERT_EQ(tileLayer.getType(), "Tile");
    ASSERT_EQ(tileLayer.getParameter("axis").as<size_t>(), 3);
    ASSERT_EQ(tileLayer.getParameter("tiles").as<size_t>(), 227);
    ASSERT_EQ(firstInputLayer.getOutPort().getConnection().getDestination().getLayer().getId(), tileLayer.getId());
    ASSERT_EQ(tileLayer.getOutPort().getConnection().getDestination().getLayer().getId(), eltwiseSumId);
}

TEST_F(TransformationTestCommon, EltwiseBroadcastTwoDimensions) {
    Builder::Network builder("eltwiseBroadcast");

    idx_t firstInputId = builder.addLayer(Builder::InputLayer("FirstInput").setPort(Port({1, 1, 227, 1})));
    idx_t secondInputId = builder.addLayer(Builder::InputLayer("SecondInput").setPort(Port({1, 3, 227, 227})));
    idx_t eltwiseSumId = builder.addLayer({firstInputId, secondInputId}, Builder::EltwiseLayer("Sum").
                                                                         setEltwiseType(Builder::EltwiseLayer::EltwiseType::SUM).
                                                                         setOutputPort(Port({1, 3, 227, 227})));
    auto network = Transform::Network(builder);

    Transform::TransformationEltwiseBroadcast transformationEltwiseBroadcast;
    transformationEltwiseBroadcast.execute(network);
    auto firstInputLayer = network.getLayer(firstInputId);
    auto tile1Layer = network.getLayer(firstInputId).getOutPort().getConnection().getDestination().getLayer();
    auto tile2Layer = tile1Layer.getOutPort().getConnection().getDestination().getLayer();
    ASSERT_EQ(tile1Layer.getType(), "Tile");
    ASSERT_EQ(tile1Layer.getParameter("axis").as<size_t>(), 1);
    ASSERT_EQ(tile1Layer.getParameter("tiles").as<size_t>(), 3);
    ASSERT_EQ(tile2Layer.getType(), "Tile");
    ASSERT_EQ(tile2Layer.getParameter("axis").as<size_t>(), 3);
    ASSERT_EQ(tile2Layer.getParameter("tiles").as<size_t>(), 227);
    ASSERT_EQ(firstInputLayer.getOutPort().getConnection().getDestination().getLayer().getId(), tile1Layer.getId());
    ASSERT_EQ(tile1Layer.getOutPort().getConnection().getDestination().getLayer().getId(), tile2Layer.getId());
    ASSERT_EQ(tile2Layer.getOutPort().getConnection().getDestination().getLayer().getId(), eltwiseSumId);
}