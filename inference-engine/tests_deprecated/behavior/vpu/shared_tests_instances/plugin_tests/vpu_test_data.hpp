// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior_test_plugin.h"

// correct params
#define BEH_MYRIAD BehTestParams("MYRIAD", \
                                 FuncTestUtils::TestModel::convReluNormPoolFcModelFP16.model_xml_str, \
                                 FuncTestUtils::TestModel::convReluNormPoolFcModelFP16.weights_blob, \
                                 Precision::FP32)
#define BEH_HETERO BehTestParams("HETERO", \
                                 FuncTestUtils::TestModel::convReluNormPoolFcModelFP32.model_xml_str, \
                                 FuncTestUtils::TestModel::convReluNormPoolFcModelFP32.weights_blob, \
                                 Precision::FP32)

// all parameters are unsupported - reversed
#define BEH_US_ALL_MYRIAD  BehTestParams("MYRIAD", \
                                         FuncTestUtils::TestModel::convReluNormPoolFcModelQ78.model_xml_str, \
                                         FuncTestUtils::TestModel::convReluNormPoolFcModelQ78.weights_blob, \
                                         Precision::Q78)
const BehTestParams supportedValues[] = {
        BEH_MYRIAD,
};

const BehTestParams requestsSupportedValues[] = {
        BEH_MYRIAD,
};

const BehTestParams allInputSupportedValues[] = {
        BEH_MYRIAD, BEH_MYRIAD.withIn(Precision::U8), BEH_MYRIAD.withIn(Precision::FP16),
        // I16 not supported yet
        // (ISSUE-7979) [IE myriad] The plugin should support I16 format for Input
        //BEH_MYRIAD.withIn(Precision::I16),
};

const BehTestParams allOutputSupportedValues[] = {
        BEH_MYRIAD, BEH_MYRIAD.withOut(Precision::FP16),
};

const BehTestParams typeUnSupportedValues[] = {
        BEH_MYRIAD.withIn(Precision::Q78), BEH_MYRIAD.withIn(Precision::U16), BEH_MYRIAD.withIn(Precision::I8),
        BEH_MYRIAD.withIn(Precision::I16), BEH_MYRIAD.withIn(Precision::I32),
};

const BehTestParams batchUnSupportedValues[] = {
        BEH_MYRIAD.withBatchSize(0),
};

const BehTestParams allUnSupportedValues[] = {
        BEH_US_ALL_MYRIAD,
};

const std::vector<BehTestParams> deviceSpecificConfigurations = {
    BEH_MYRIAD.withConfig({{VPU_MYRIAD_CONFIG_KEY(PROTOCOL), VPU_MYRIAD_CONFIG_VALUE(USB)}}),
    BEH_MYRIAD.withConfig({{VPU_MYRIAD_CONFIG_KEY(PROTOCOL), VPU_MYRIAD_CONFIG_VALUE(PCIE)}}),

    BEH_MYRIAD.withConfig({{VPU_MYRIAD_CONFIG_KEY(PLATFORM), VPU_MYRIAD_CONFIG_VALUE(2450)}}),
    BEH_MYRIAD.withConfig({{VPU_MYRIAD_CONFIG_KEY(PLATFORM), VPU_MYRIAD_CONFIG_VALUE(2480)}}),
};

const std::vector<BehTestParams> deviceAgnosticConfigurations = {
    BEH_MYRIAD.withConfig({{VPU_CONFIG_KEY(IGNORE_IR_STATISTIC), CONFIG_VALUE(YES)}}),
    BEH_MYRIAD.withConfig({{VPU_CONFIG_KEY(IGNORE_IR_STATISTIC), CONFIG_VALUE(NO)}}),

    BEH_MYRIAD.withConfig({{VPU_MYRIAD_CONFIG_KEY(FORCE_RESET), CONFIG_VALUE(YES)}}),
    BEH_MYRIAD.withConfig({{VPU_MYRIAD_CONFIG_KEY(FORCE_RESET), CONFIG_VALUE(NO)}}),

    BEH_MYRIAD.withConfig({{CONFIG_KEY(LOG_LEVEL), CONFIG_VALUE(LOG_NONE)}}),
    BEH_MYRIAD.withConfig({{CONFIG_KEY(LOG_LEVEL), CONFIG_VALUE(LOG_ERROR)}}),
    BEH_MYRIAD.withConfig({{CONFIG_KEY(LOG_LEVEL), CONFIG_VALUE(LOG_WARNING)}}),
    BEH_MYRIAD.withConfig({{CONFIG_KEY(LOG_LEVEL), CONFIG_VALUE(LOG_INFO)}}),
    BEH_MYRIAD.withConfig({{CONFIG_KEY(LOG_LEVEL), CONFIG_VALUE(LOG_DEBUG)}}),
    BEH_MYRIAD.withConfig({{CONFIG_KEY(LOG_LEVEL), CONFIG_VALUE(LOG_TRACE)}}),

    BEH_MYRIAD.withConfig({{VPU_CONFIG_KEY(HW_STAGES_OPTIMIZATION), CONFIG_VALUE(YES)}}),
    BEH_MYRIAD.withConfig({{VPU_CONFIG_KEY(HW_STAGES_OPTIMIZATION), CONFIG_VALUE(NO)}}),

    BEH_MYRIAD.withConfig({{VPU_CONFIG_KEY(PRINT_RECEIVE_TENSOR_TIME), CONFIG_VALUE(YES)}}),
    BEH_MYRIAD.withConfig({{VPU_CONFIG_KEY(PRINT_RECEIVE_TENSOR_TIME), CONFIG_VALUE(NO)}}),
};

const std::vector<BehTestParams> withCorrectConfValuesPluginOnly = {
};

const std::vector<BehTestParams> withCorrectConfValuesNetworkOnly = {
};

const BehTestParams withIncorrectConfValues[] = {
    BEH_MYRIAD.withConfig({{VPU_MYRIAD_CONFIG_KEY(PROTOCOL), "BLUETOOTH"}}),
    BEH_MYRIAD.withConfig({{VPU_MYRIAD_CONFIG_KEY(PROTOCOL), "LAN"}}),

    BEH_MYRIAD.withConfig({{VPU_CONFIG_KEY(IGNORE_IR_STATISTIC), "ON"}}),
    BEH_MYRIAD.withConfig({{VPU_CONFIG_KEY(IGNORE_IR_STATISTIC), "OFF"}}),

    BEH_MYRIAD.withConfig({{VPU_CONFIG_KEY(HW_STAGES_OPTIMIZATION), "ON"}}),
    BEH_MYRIAD.withConfig({{VPU_CONFIG_KEY(HW_STAGES_OPTIMIZATION), "OFF"}}),

    BEH_MYRIAD.withConfig({{VPU_MYRIAD_CONFIG_KEY(FORCE_RESET), "ON"}}),
    BEH_MYRIAD.withConfig({{VPU_MYRIAD_CONFIG_KEY(FORCE_RESET), "OFF"}}),

    BEH_MYRIAD.withConfig({{CONFIG_KEY(LOG_LEVEL), "VERBOSE"}}),

    BEH_MYRIAD.withConfig({{VPU_MYRIAD_CONFIG_KEY(PLATFORM), "-1"}}),
    BEH_MYRIAD.withConfig({{VPU_MYRIAD_CONFIG_KEY(PLATFORM), "0"}}),
    BEH_MYRIAD.withConfig({{VPU_MYRIAD_CONFIG_KEY(PLATFORM), "1"}}),

    BEH_MYRIAD.withConfig({{VPU_CONFIG_KEY(PRINT_RECEIVE_TENSOR_TIME), "ON"}}),
    BEH_MYRIAD.withConfig({{VPU_CONFIG_KEY(PRINT_RECEIVE_TENSOR_TIME), "OFF"}}),
};

const BehTestParams withIncorrectConfKeys[] = {
        BEH_MYRIAD.withIncorrectConfigItem(),
};
