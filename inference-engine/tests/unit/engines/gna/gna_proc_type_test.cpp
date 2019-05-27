// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include <gtest/gtest.h>
#include <mock_icnn_network.hpp>
#include <gmock/gmock-generated-actions.h>
#include <gna/gna_config.hpp>
#include "gna_plugin.hpp"
#include "gna_mock_api.hpp"
#include "gna_matcher.hpp"

using namespace std;
using namespace InferenceEngine;
using namespace GNAPluginNS;
using namespace ::testing;

class GNAProcTypeTest : public GNATest {

 protected:
};

TEST_F(GNAProcTypeTest, defaultProcTypeIsSWEXACT) {
    assert_that().onInfer1AFModel().gna().propagate_forward().called_with().proc_type(GNA_SOFTWARE & GNA_HARDWARE);
}

TEST_F(GNAProcTypeTest, canPassHWProcTypeToGNA) {
    assert_that().onInfer1AFModel().withGNADeviceMode("GNA_HW").gna().propagate_forward().called_with().proc_type(GNA_HARDWARE);
}

TEST_F(GNAProcTypeTest, canPassSWProcTypeToGNA) {
    assert_that().onInfer1AFModel().withGNADeviceMode("GNA_SW").gna().propagate_forward().called_with().proc_type(GNA_SOFTWARE);
}

TEST_F(GNAProcTypeTest, canPassSWEXACTProcTypeToGNA) {
    assert_that().onInfer1AFModel().withGNADeviceMode("GNA_SW_EXACT").gna().
        propagate_forward().called_with().proc_type(GNA_SOFTWARE & GNA_HARDWARE);
}