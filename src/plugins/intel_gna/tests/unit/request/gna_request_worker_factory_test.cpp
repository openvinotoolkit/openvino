// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "mock_gna_device.hpp"
#include "mock_subrequest.hpp"
#include "request/model_wrapper_factory.hpp"
#include "request/worker_factory.hpp"

using namespace GNAPluginNS;
using namespace request;
using namespace testing;

class GNA_Request_WorkerFactoryTest : public ::testing::Test {};

TEST_F(GNA_Request_WorkerFactoryTest, createWorker_without_splitting) {
    const Gna2AccelerationMode accMode = Gna2AccelerationModeAuto;
    auto deviceMock = std::make_shared<MockGNADevice>();

    std::vector<std::shared_ptr<Subrequest>> subrequests;
    // expect throw when model nullptr
    EXPECT_THROW(subrequests = WorkerFactory::createModelSubrequests(nullptr, deviceMock, accMode), std::exception);

    // kMaxLayersBigger value bigger than model layers number
    const constexpr uint32_t kObjectLayers = 2;
    const constexpr uint32_t kMaxLayersBigger = kObjectLayers + 1;
    const constexpr uint32_t numberOfPieces = 1;
    auto modelWrapper = ModelWrapperFactory::createWithNumberOfEmptyOperations(kObjectLayers);

    // expect throw when model nullptr
    EXPECT_THROW(subrequests = WorkerFactory::createModelSubrequests(modelWrapper, nullptr, accMode), std::exception);

    EXPECT_CALL(*deviceMock.get(), maxLayersCount()).Times(1).WillOnce(Return(kMaxLayersBigger));
    EXPECT_CALL(*deviceMock.get(), createModel(_)).Times(numberOfPieces).WillOnce(Return(0));
    EXPECT_CALL(*deviceMock.get(), createRequestConfig(_)).Times(numberOfPieces).WillOnce(Return(0));

    EXPECT_NO_THROW((subrequests = WorkerFactory::createModelSubrequests(modelWrapper, deviceMock, accMode)));
    EXPECT_EQ(subrequests.size(), numberOfPieces);
    subrequests.clear();

    // max layers count value equal to layers number
    EXPECT_CALL(*deviceMock.get(), maxLayersCount()).Times(1).WillOnce(Return(kObjectLayers));
    EXPECT_CALL(*deviceMock.get(), createModel(_)).Times(numberOfPieces).WillOnce(Return(0));
    EXPECT_CALL(*deviceMock.get(), createRequestConfig(_)).Times(numberOfPieces).WillOnce(Return(0));

    EXPECT_NO_THROW(subrequests = WorkerFactory::createModelSubrequests(modelWrapper, deviceMock, accMode));
    EXPECT_EQ(subrequests.size(), numberOfPieces);
}

TEST_F(GNA_Request_WorkerFactoryTest, createWorker_with_splitting) {
    auto deviceMock = std::make_shared<MockGNADevice>();

    const constexpr uint32_t kObjectLayers = 5;
    const Gna2AccelerationMode accMode = Gna2AccelerationModeAuto;

    auto modelWrapper = ModelWrapperFactory::createWithNumberOfEmptyOperations(kObjectLayers);

    // ensure that kObjectLayers == 0 value is handled
    const constexpr uint32_t kMaxLayersZero = 0;
    EXPECT_CALL(*deviceMock.get(), maxLayersCount()).Times(1).WillOnce(Return(kMaxLayersZero));
    std::vector<std::shared_ptr<Subrequest>> subrequests;
    EXPECT_THROW(subrequests = WorkerFactory::createModelSubrequests(modelWrapper, deviceMock, accMode),
                 std::exception);
    subrequests.clear();

    // check layers bigger than kMaxLayersSmaller - split into two pieces
    const constexpr uint32_t kMaxLayersSmaller = kObjectLayers - 1;
    size_t numberOfPieces = 2;
    EXPECT_CALL(*deviceMock.get(), maxLayersCount()).Times(1).WillOnce(Return(kMaxLayersSmaller));
    subrequests.clear();

    Gna2Model tempModel1;
    Gna2Model tempModel2;
    EXPECT_CALL(*deviceMock.get(), createModel(_))
        .Times(numberOfPieces)
        .WillOnce(DoAll(SaveArg<0>(&tempModel1), Return(0)))
        .WillOnce(DoAll(SaveArg<0>(&tempModel2), Return(0)));
    EXPECT_CALL(*deviceMock.get(), createRequestConfig(_)).Times(numberOfPieces).WillRepeatedly(Return(0));

    EXPECT_NO_THROW(subrequests = WorkerFactory::createModelSubrequests(modelWrapper, deviceMock, accMode));

    EXPECT_EQ(tempModel1.NumberOfOperations, kMaxLayersSmaller);          // 4
    EXPECT_EQ(tempModel1.Operations, modelWrapper->object().Operations);  // Operations[0]

    EXPECT_EQ(tempModel2.NumberOfOperations, kObjectLayers % kMaxLayersSmaller);              // 1
    EXPECT_EQ(tempModel2.Operations, &modelWrapper->object().Operations[kMaxLayersSmaller]);  // Operations[4]

    EXPECT_EQ(subrequests.size(), numberOfPieces);
}
