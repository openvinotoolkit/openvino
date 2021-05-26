// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior_test_plugin.h"

// correct params
#define BEH_MYRIAD BehTestParams("MYRIAD", \
                                 convReluNormPoolFcModelFP16.model_xml_str, \
                                 convReluNormPoolFcModelFP16.weights_blob, \
                                 Precision::FP32)
#define BEH_HETERO BehTestParams("HETERO", \
                                 convReluNormPoolFcModelFP32.model_xml_str, \
                                 convReluNormPoolFcModelFP32.weights_blob, \
                                 Precision::FP32)
// for multi-device we are testing the fp16 (as it is supported by all device combos we are considering for testing
// e.g. GPU and VPU, for CPU the network is automatically (internally) converted to fp32.
#define BEH_MULTI(device) BehTestParams("MULTI", \
                                        convReluNormPoolFcModelFP16.model_xml_str, \
                                        convReluNormPoolFcModelFP16.weights_blob, \
                                        Precision::FP32, \
                                        {{MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, #device}})
#define BEH_MULTI_CONFIG  BehTestParams("MULTI", \
                                        convReluNormPoolFcModelFP16.model_xml_str, \
                                        convReluNormPoolFcModelFP16.weights_blob, \
                                        Precision::FP32)

// all parameters are unsupported - reversed
#define BEH_US_ALL_MYRIAD  BehTestParams("MYRIAD", \
                                         convReluNormPoolFcModelQ78.model_xml_str, \
                                         convReluNormPoolFcModelQ78.weights_blob, \
                                         Precision::Q78)
#define BEH_US_ALL_MULTI(device) BehTestParams("MULTI", \
                                               convReluNormPoolFcModelQ78.model_xml_str, \
                                               convReluNormPoolFcModelQ78.weights_blob, \
                                               Precision::Q78, \
                                               {{MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, #device}})
const BehTestParams supportedValues[] = {
        BEH_MYRIAD,
        BEH_MULTI(MYRIAD),
};

const BehTestParams requestsSupportedValues[] = {
        BEH_MYRIAD,
        BEH_MULTI(MYRIAD),
};

const BehTestParams allInputSupportedValues[] = {
        BEH_MYRIAD, BEH_MYRIAD.withIn(Precision::U8), BEH_MYRIAD.withIn(Precision::FP16),
        BEH_MULTI(MYRIAD), BEH_MULTI(MYRIAD).withIn(Precision::U8), BEH_MULTI(MYRIAD).withIn(Precision::FP16),
        // I16 not supported yet
        // (Issue-7979) [IE myriad] The plugin should support I16 format for Input
        //BEH_MYRIAD.withIn(Precision::I16),
};

const BehTestParams allOutputSupportedValues[] = {
        BEH_MYRIAD, BEH_MYRIAD.withOut(Precision::FP16),
        BEH_MULTI(MYRIAD), BEH_MULTI(MYRIAD).withOut(Precision::FP16),
};

const BehTestParams typeUnSupportedValues[] = {
        BEH_MYRIAD.withIn(Precision::Q78), BEH_MYRIAD.withIn(Precision::U16), BEH_MYRIAD.withIn(Precision::I8),
        BEH_MYRIAD.withIn(Precision::I16), BEH_MYRIAD.withIn(Precision::I32),
        BEH_MULTI(MYRIAD).withIn(Precision::Q78), BEH_MULTI(MYRIAD).withIn(Precision::U16),
        BEH_MULTI(MYRIAD).withIn(Precision::I8),
        BEH_MULTI(MYRIAD).withIn(Precision::I16), BEH_MULTI(MYRIAD).withIn(Precision::I32),
};

const BehTestParams batchUnSupportedValues[] = {
        BEH_MYRIAD.withBatchSize(0),
        BEH_MULTI(MYRIAD).withBatchSize(0),
};

const BehTestParams allUnSupportedValues[] = {
        BEH_US_ALL_MYRIAD,
        BEH_US_ALL_MULTI(MYRIAD),
};

const std::vector<BehTestParams> deviceSpecificConfigurations = {
    BEH_MYRIAD.withConfig({{InferenceEngine::MYRIAD_PROTOCOL, InferenceEngine::MYRIAD_USB}}),
    BEH_MYRIAD.withConfig({{InferenceEngine::MYRIAD_PROTOCOL, InferenceEngine::MYRIAD_PCIE}}),

    // Deprecated
    BEH_MYRIAD.withConfig({{VPU_MYRIAD_CONFIG_KEY(PLATFORM), VPU_MYRIAD_CONFIG_VALUE(2450)}}),
    BEH_MYRIAD.withConfig({{VPU_MYRIAD_CONFIG_KEY(PLATFORM), VPU_MYRIAD_CONFIG_VALUE(2480)}}),

    BEH_MYRIAD.withConfig({{VPU_MYRIAD_CONFIG_KEY(PROTOCOL), VPU_MYRIAD_CONFIG_VALUE(USB)}}),
    BEH_MYRIAD.withConfig({{VPU_MYRIAD_CONFIG_KEY(PROTOCOL), VPU_MYRIAD_CONFIG_VALUE(PCIE)}}),
};

const std::vector<BehTestParams> deviceAgnosticConfigurations = {
    BEH_MYRIAD.withConfig({{InferenceEngine::MYRIAD_ENABLE_FORCE_RESET, CONFIG_VALUE(YES)}}),
    BEH_MYRIAD.withConfig({{InferenceEngine::MYRIAD_ENABLE_FORCE_RESET, CONFIG_VALUE(NO)}}),

    BEH_MYRIAD.withConfig({{CONFIG_KEY(LOG_LEVEL), CONFIG_VALUE(LOG_NONE)}}),
    BEH_MYRIAD.withConfig({{CONFIG_KEY(LOG_LEVEL), CONFIG_VALUE(LOG_ERROR)}}),
    BEH_MYRIAD.withConfig({{CONFIG_KEY(LOG_LEVEL), CONFIG_VALUE(LOG_WARNING)}}),
    BEH_MYRIAD.withConfig({{CONFIG_KEY(LOG_LEVEL), CONFIG_VALUE(LOG_INFO)}}),
    BEH_MYRIAD.withConfig({{CONFIG_KEY(LOG_LEVEL), CONFIG_VALUE(LOG_DEBUG)}}),
    BEH_MYRIAD.withConfig({{CONFIG_KEY(LOG_LEVEL), CONFIG_VALUE(LOG_TRACE)}}),

    BEH_MYRIAD.withConfig({{InferenceEngine::MYRIAD_ENABLE_HW_ACCELERATION, CONFIG_VALUE(YES)}}),
    BEH_MYRIAD.withConfig({{InferenceEngine::MYRIAD_ENABLE_HW_ACCELERATION, CONFIG_VALUE(NO)}}),

    BEH_MYRIAD.withConfig({{InferenceEngine::MYRIAD_ENABLE_RECEIVING_TENSOR_TIME, CONFIG_VALUE(YES)}}),
    BEH_MYRIAD.withConfig({{InferenceEngine::MYRIAD_ENABLE_RECEIVING_TENSOR_TIME, CONFIG_VALUE(NO)}}),

    BEH_MYRIAD.withConfig({{InferenceEngine::MYRIAD_THROUGHPUT_STREAMS, "1"}}),
    BEH_MYRIAD.withConfig({{InferenceEngine::MYRIAD_THROUGHPUT_STREAMS, "2"}}),
    BEH_MYRIAD.withConfig({{InferenceEngine::MYRIAD_THROUGHPUT_STREAMS, "3"}}),


    BEH_MULTI_CONFIG.withConfig({
        {MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, "MYRIAD"},
        {CONFIG_KEY(LOG_LEVEL), CONFIG_VALUE(LOG_DEBUG)}
    }),
    BEH_MULTI_CONFIG.withConfig({
        {MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, "MYRIAD"},
        {InferenceEngine::MYRIAD_ENABLE_HW_ACCELERATION, CONFIG_VALUE(YES)}
    }),

    // Please do not use other types of DDR in tests with a real device, because it may hang.
    BEH_MYRIAD.withConfig({{InferenceEngine::MYRIAD_DDR_TYPE, InferenceEngine::MYRIAD_DDR_AUTO}}),

    // Deprecated
    BEH_MYRIAD.withConfig({{VPU_MYRIAD_CONFIG_KEY(FORCE_RESET), CONFIG_VALUE(YES)}}),
    BEH_MYRIAD.withConfig({{VPU_MYRIAD_CONFIG_KEY(FORCE_RESET), CONFIG_VALUE(NO)}}),

    BEH_MYRIAD.withConfig({{VPU_CONFIG_KEY(HW_STAGES_OPTIMIZATION), CONFIG_VALUE(YES)}}),
    BEH_MYRIAD.withConfig({{VPU_CONFIG_KEY(HW_STAGES_OPTIMIZATION), CONFIG_VALUE(NO)}}),

    BEH_MYRIAD.withConfig({{VPU_CONFIG_KEY(PRINT_RECEIVE_TENSOR_TIME), CONFIG_VALUE(YES)}}),
    BEH_MYRIAD.withConfig({{VPU_CONFIG_KEY(PRINT_RECEIVE_TENSOR_TIME), CONFIG_VALUE(NO)}}),

    BEH_MULTI_CONFIG.withConfig({
        {MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, "MYRIAD"},
        {VPU_CONFIG_KEY(HW_STAGES_OPTIMIZATION), CONFIG_VALUE(YES)}
    }),

    // Please do not use other types of DDR in tests with a real device, because it may hang.
    BEH_MYRIAD.withConfig({{VPU_MYRIAD_CONFIG_KEY(MOVIDIUS_DDR_TYPE), VPU_MYRIAD_CONFIG_VALUE(DDR_AUTO)}}),
};

const std::vector<BehTestParams> withCorrectConfValuesPluginOnly = {
};

const std::vector<BehTestParams> withCorrectConfValuesNetworkOnly = {
};

const BehTestParams withIncorrectConfValues[] = {
    BEH_MYRIAD.withConfig({{InferenceEngine::MYRIAD_PROTOCOL, "BLUETOOTH"}}),
    BEH_MYRIAD.withConfig({{InferenceEngine::MYRIAD_PROTOCOL, "LAN"}}),

    BEH_MYRIAD.withConfig({{InferenceEngine::MYRIAD_ENABLE_HW_ACCELERATION, "ON"}}),
    BEH_MYRIAD.withConfig({{InferenceEngine::MYRIAD_ENABLE_HW_ACCELERATION, "OFF"}}),

    BEH_MYRIAD.withConfig({{InferenceEngine::MYRIAD_ENABLE_FORCE_RESET, "ON"}}),
    BEH_MYRIAD.withConfig({{InferenceEngine::MYRIAD_ENABLE_FORCE_RESET, "OFF"}}),

    BEH_MYRIAD.withConfig({{CONFIG_KEY(LOG_LEVEL), "VERBOSE"}}),

    BEH_MYRIAD.withConfig({{InferenceEngine::MYRIAD_ENABLE_RECEIVING_TENSOR_TIME, "ON"}}),
    BEH_MYRIAD.withConfig({{InferenceEngine::MYRIAD_ENABLE_RECEIVING_TENSOR_TIME, "OFF"}}),

    BEH_MYRIAD.withConfig({{InferenceEngine::MYRIAD_THROUGHPUT_STREAMS, "Single"}}),
    BEH_MYRIAD.withConfig({{InferenceEngine::MYRIAD_THROUGHPUT_STREAMS, "TWO"}}),

    BEH_MULTI_CONFIG.withConfig({{MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, "MYRIAD"},
                                 {InferenceEngine::MYRIAD_ENABLE_HW_ACCELERATION,"ON"}}),
    BEH_MULTI_CONFIG.withConfig({{MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, "MYRIAD"},
                                 {CONFIG_KEY(LOG_LEVEL), "VERBOSE"}}),

    BEH_MYRIAD.withConfig({{InferenceEngine::MYRIAD_DDR_TYPE, "-1"}}),
    BEH_MYRIAD.withConfig({{InferenceEngine::MYRIAD_DDR_TYPE, "MICRON"}}),

    // Deprecated
    BEH_MYRIAD.withConfig({{VPU_MYRIAD_CONFIG_KEY(PROTOCOL), "BLUETOOTH"}}),
    BEH_MYRIAD.withConfig({{VPU_MYRIAD_CONFIG_KEY(PROTOCOL), "LAN"}}),

    BEH_MYRIAD.withConfig({{VPU_CONFIG_KEY(HW_STAGES_OPTIMIZATION), "ON"}}),
    BEH_MYRIAD.withConfig({{VPU_CONFIG_KEY(HW_STAGES_OPTIMIZATION), "OFF"}}),

    BEH_MYRIAD.withConfig({{VPU_MYRIAD_CONFIG_KEY(FORCE_RESET), "ON"}}),
    BEH_MYRIAD.withConfig({{VPU_MYRIAD_CONFIG_KEY(FORCE_RESET), "OFF"}}),

    BEH_MYRIAD.withConfig({{VPU_CONFIG_KEY(PRINT_RECEIVE_TENSOR_TIME), "ON"}}),
    BEH_MYRIAD.withConfig({{VPU_CONFIG_KEY(PRINT_RECEIVE_TENSOR_TIME), "OFF"}}),

    BEH_MULTI_CONFIG.withConfig({{MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, "MYRIAD"},
                                 {VPU_CONFIG_KEY(HW_STAGES_OPTIMIZATION),"ON"}}),

    BEH_MYRIAD.withConfig({{VPU_MYRIAD_CONFIG_KEY(MOVIDIUS_DDR_TYPE), "-1"}}),
    BEH_MYRIAD.withConfig({{VPU_MYRIAD_CONFIG_KEY(MOVIDIUS_DDR_TYPE), "MICRON"}}),

    BEH_MYRIAD.withConfig({{VPU_MYRIAD_CONFIG_KEY(PLATFORM), "-1"}}),
    BEH_MYRIAD.withConfig({{VPU_MYRIAD_CONFIG_KEY(PLATFORM), "0"}}),
    BEH_MYRIAD.withConfig({{VPU_MYRIAD_CONFIG_KEY(PLATFORM), "1"}}),

    BEH_MULTI_CONFIG.withConfig({{MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, "MYRIAD"},
                                 {VPU_MYRIAD_CONFIG_KEY(PLATFORM), "-1"}}),
    BEH_MULTI_CONFIG.withConfig({{MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, "MYRIAD"},
                                 {VPU_MYRIAD_CONFIG_KEY(PLATFORM), "0"}}),
    BEH_MULTI_CONFIG.withConfig({{MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, "MYRIAD"},
                                 {VPU_MYRIAD_CONFIG_KEY(PLATFORM), "1"}}),
};

const BehTestParams withIncorrectConfKeys[] = {
        BEH_MYRIAD.withIncorrectConfigItem(),
        BEH_MULTI(MYRIAD).withIncorrectConfigItem(),
};
