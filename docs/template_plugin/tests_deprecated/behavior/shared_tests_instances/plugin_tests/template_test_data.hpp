// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "behavior_test_plugin.h"

// correct params
#define BEH_HETERO BehTestParams("HETERO", \
                                 FuncTestUtils::TestModel::convReluNormPoolFcModelFP32.model_xml_str, \
                                 FuncTestUtils::TestModel::convReluNormPoolFcModelFP32.weights_blob, \
                                 Precision::FP32)

#define BEH_TEMPLATE BehTestParams("TEMPLATE", \
                               FuncTestUtils::TestModel::convReluNormPoolFcModelFP16.model_xml_str, \
                               FuncTestUtils::TestModel::convReluNormPoolFcModelFP16.weights_blob, \
                               Precision::FP32)

// all parameters are unsupported - reversed
#define BEH_US_ALL_TEMPLATE    BehTestParams("TEMPLATE", \
                                         FuncTestUtils::TestModel::convReluNormPoolFcModelQ78.model_xml_str, \
                                         FuncTestUtils::TestModel::convReluNormPoolFcModelQ78.weights_blob, \
                                         Precision::Q78)

const BehTestParams supportedValues[] = {
    BEH_TEMPLATE,
};

const BehTestParams requestsSupportedValues[] = {
    BEH_TEMPLATE,
};

const BehTestParams allInputSupportedValues[] = {
    BEH_TEMPLATE,
    BEH_TEMPLATE.withIn(Precision::FP16),
    BEH_TEMPLATE.withIn(Precision::U8),
    BEH_TEMPLATE.withIn(Precision::I16),
};

const BehTestParams allOutputSupportedValues[] = {
    BEH_TEMPLATE,
    BEH_TEMPLATE.withOut(Precision::FP16),
};

const BehTestParams typeUnSupportedValues[] = {
    BEH_TEMPLATE.withIn(Precision::Q78),
    BEH_TEMPLATE.withIn(Precision::U16),
    BEH_TEMPLATE.withIn(Precision::I8),
    BEH_TEMPLATE.withIn(Precision::I32),
};

const BehTestParams batchUnSupportedValues[] = {
    BEH_TEMPLATE.withBatchSize(0),
};

const BehTestParams allUnSupportedValues[] = {
    BEH_US_ALL_TEMPLATE,
};

const std::vector<BehTestParams> withCorrectConfValuesNetworkOnly = {
    BEH_TEMPLATE.withConfig({ { KEY_DEVICE_ID, "0" } }),
};

const BehTestParams withIncorrectConfValues[] = {
    BEH_TEMPLATE.withConfig({ { KEY_CPU_BIND_THREAD, "ON" } }),
};

const std::vector<BehTestParams> withCorrectConfValues = {
    BEH_TEMPLATE.withConfig({}),
};
