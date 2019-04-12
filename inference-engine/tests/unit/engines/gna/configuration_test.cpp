//
// Copyright 2016-2018 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#include <vector>

#include <gtest/gtest.h>
#include <mock_icnn_network.hpp>
#include "gna_plugin/gna_plugin_config.hpp"
#include "gna_matcher.hpp"
#include "test_irs.hpp"

using namespace InferenceEngine;
using namespace GNAPluginNS;
using namespace ::testing;

class GNAConfigTest : public GNATest {

 protected:
    MockICNNNetwork net;

    void SetUp() override  {
    }
};

TEST_F(GNAConfigTest, reportAnErrorIfConfigNotFound) {

    Config c ({{TargetDevice :: eGNA, Precision::I16},
               {TargetDevice :: eCPU, Precision::FP32}});

    EXPECT_CALL(net, getPrecision()).WillRepeatedly(Return(Precision::FP32));
    EXPECT_CALL(net, getTargetDevice()).WillRepeatedly(Return(TargetDevice::eGNA));

    ASSERT_ANY_THROW(c.find_configuration(net));
}

TEST_F(GNAConfigTest, canFindConfiguration) {

    Config c ({{TargetDevice :: eGNA, Precision::I16},
               {TargetDevice :: eCPU, Precision::FP32}});

    EXPECT_CALL(net, getPrecision()).WillRepeatedly(Return(Precision::FP32));
    EXPECT_CALL(net, getTargetDevice()).WillRepeatedly(Return(TargetDevice::eCPU));

    auto match = c.find_configuration(net);

    EXPECT_EQ(match.device, TargetDevice::eCPU);
    EXPECT_EQ(match.networkPrec, Precision::FP32);
}

TEST_F(GNAConfigTest, canPassTroughNetworkAfterFindConfiguration) {

    Config c ({{TargetDevice :: eGNA, Precision::I16},
               {TargetDevice :: eCPU, Precision::FP32}});

    EXPECT_CALL(net, getPrecision()).WillRepeatedly(Return(Precision::FP32));
    EXPECT_CALL(net, getTargetDevice()).WillRepeatedly(Return(TargetDevice::eCPU));

    auto match = c.find_configuration(net);

    auto net2 = match.convert(net);

    EXPECT_EQ(net2->getTargetDevice(), TargetDevice::eCPU);
    EXPECT_EQ(net2->getPrecision(), Precision::FP32);
}

TEST_F(GNAConfigTest, canNotMatchWithDefaultDevice) {

    Config c ({{TargetDevice :: eGNA, Precision::I16},
               {TargetDevice :: eCPU, Precision::FP32}});

    c.setDefaultDevice(TargetDevice::eGNA);

    EXPECT_CALL(net, getPrecision()).WillRepeatedly(Return(Precision::FP32));
    EXPECT_CALL(net, getTargetDevice()).WillRepeatedly(Return(TargetDevice::eDefault));

    EXPECT_ANY_THROW(c.find_configuration(net).convert(net));
}

TEST_F(GNAConfigTest, canMatchWithDefaultDevice) {

    Config c ({{TargetDevice :: eGNA, Precision::I16},
               {TargetDevice :: eCPU, Precision::FP32}});

    c.setDefaultDevice(TargetDevice::eGNA);

    EXPECT_CALL(net, getPrecision()).WillRepeatedly(Return(Precision::I16));
    EXPECT_CALL(net, getTargetDevice()).WillRepeatedly(Return(TargetDevice::eDefault));

    auto net2 = c.find_configuration(net).convert(net);

    EXPECT_EQ(net2->getTargetDevice(), TargetDevice::eDefault);
    EXPECT_EQ(net2->getPrecision(), Precision::I16);
}

TEST_F(GNAConfigTest, canMatchWith1AsyncThread) {
    assert_that()
        .onInferModel(GNATestIRs::Fc2DOutputModel())
        .inNotCompactMode()
        .withAcceleratorThreadsNumber("1")
        .gna().propagate_forward().called_without().pwl_inserted_into_nnet();
}

TEST_F(GNAConfigTest, canMatchWith4AsyncThreads) {
    assert_that()
        .onInferModel(GNATestIRs::Fc2DOutputModel())
        .inNotCompactMode()
        .withAcceleratorThreadsNumber("4")
        .gna().propagate_forward().called_without().pwl_inserted_into_nnet();
}

TEST_F(GNAConfigTest, canNOTMatchWith0AsyncThreads) {
    assert_that()
        .onInferModel(GNATestIRs::Fc2DOutputModel())
        .inNotCompactMode()
        .withAcceleratorThreadsNumber("0")
        .gna().propagate_forward().called_without().pwl_inserted_into_nnet()
        .throws();
}

TEST_F(GNAConfigTest, canNOTMatchWith128AsyncThreads) {
    assert_that()
        .onInferModel(GNATestIRs::Fc2DOutputModel())
        .inNotCompactMode()
        .withAcceleratorThreadsNumber("128")
        .gna().propagate_forward().called_without().pwl_inserted_into_nnet()
        .throws();
}

TEST_F(GNAConfigTest, canMatchWithSingleMultipleOMPThreads) {
    assert_that()
        .onInferModel(GNATestIRs::Fc2DOutputModel())
        .inNotCompactMode()
        .enable_omp_multithreading()
        .gna().propagate_forward().called_without().pwl_inserted_into_nnet();
}

TEST_F(GNAConfigTest, failToCreatePluginWithDifferentInputScaleFactors) {
    assert_that().creating().gna_plugin()
        .withGNAConfig(std::string(GNA_CONFIG_KEY(SCALE_FACTOR))+"_1", 1000)
        .withGNAConfig(std::string(GNA_CONFIG_KEY(SCALE_FACTOR))+"_2", 2000).throws();
}