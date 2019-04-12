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