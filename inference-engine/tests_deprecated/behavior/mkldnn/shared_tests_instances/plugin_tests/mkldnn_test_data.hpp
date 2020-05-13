// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior_test_plugin.h"

// correct params
#define BEH_MKLDNN BehTestParams("CPU", \
                                 FuncTestUtils::TestModel::convReluNormPoolFcModelFP32.model_xml_str, \
                                 FuncTestUtils::TestModel::convReluNormPoolFcModelFP32.weights_blob, \
                                 Precision::FP32)
#define BEH_MKLDNN_FP16 BehTestParams("CPU", \
                                 FuncTestUtils::TestModel::convReluNormPoolFcModelFP16.model_xml_str, \
                                 FuncTestUtils::TestModel::convReluNormPoolFcModelFP16.weights_blob, \
                                 Precision::FP16)
#define BEH_HETERO BehTestParams("HETERO", \
                                 FuncTestUtils::TestModel::convReluNormPoolFcModelFP32.model_xml_str, \
                                 FuncTestUtils::TestModel::convReluNormPoolFcModelFP32.weights_blob, \
                                 Precision::FP32)
// for multi-device we are testing the fp16 (as it is supported by all device combos we are considering for testing
// e.g. GPU and VPU, for CPU the network is automatically (internally) converted to fp32.
// Yet the input precision FP16 is not supported by the CPU yet
const std::map<std::string, std::string> multi_device_conf = {{MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, "CPU"}};
#define BEH_MULTI BehTestParams("MULTI", \
                                FuncTestUtils::TestModel::convReluNormPoolFcModelFP16.model_xml_str, \
                                FuncTestUtils::TestModel::convReluNormPoolFcModelFP16.weights_blob, \
                                Precision::FP32, \
                                multi_device_conf)

// all parameters are unsupported - reversed
#define BEH_US_ALL_MKLDNN  BehTestParams("CPU", \
                                         FuncTestUtils::TestModel::convReluNormPoolFcModelQ78.model_xml_str, \
                                         FuncTestUtils::TestModel::convReluNormPoolFcModelQ78.weights_blob, \
                                         Precision::Q78)
#define BEH_US_ALL_MULTI   BehTestParams("MULTI", \
                                         FuncTestUtils::TestModel::convReluNormPoolFcModelQ78.model_xml_str, \
                                         FuncTestUtils::TestModel::convReluNormPoolFcModelQ78.weights_blob, \
                                         Precision::Q78, \
                                         multi_device_conf)

const BehTestParams supportedValues[] = {
        BEH_MKLDNN,
        BEH_MULTI,
};

const BehTestParams requestsSupportedValues[] = {
        BEH_MKLDNN,
        // the following adds additional test the MKLDNNGraphlessInferRequest (explicitly created for streams)
        BEH_MKLDNN.withConfig({{KEY_CPU_THROUGHPUT_STREAMS, CPU_THROUGHPUT_AUTO}}),
        BEH_MKLDNN.withConfig({{CONFIG_KEY(CPU_THROUGHPUT_STREAMS),"0"},
                               {CONFIG_KEY(CPU_THREADS_NUM), "1"}}),
        BEH_MULTI,
};

const BehTestParams allInputSupportedValues[] = {
        BEH_MKLDNN, BEH_MKLDNN.withIn(Precision::U8), BEH_MKLDNN.withIn(Precision::U16),
        BEH_MKLDNN.withIn(Precision::I16),
        // the following list withConfig tests the MKLDNNGraphlessInferRequest (explicitly created for streams)
        BEH_MKLDNN.withIn(Precision::U8).withConfig({{KEY_CPU_THROUGHPUT_STREAMS, CPU_THROUGHPUT_AUTO}}),
        BEH_MKLDNN.withIn(Precision::U16).withConfig({{KEY_CPU_THROUGHPUT_STREAMS, CPU_THROUGHPUT_AUTO}}),
        BEH_MKLDNN.withIn(Precision::I16).withConfig({{KEY_CPU_THROUGHPUT_STREAMS, CPU_THROUGHPUT_AUTO}}),
        BEH_MKLDNN_FP16.withIn(Precision::FP32),
        BEH_MKLDNN_FP16.withIn(Precision::U8),
        BEH_MKLDNN_FP16.withIn(Precision::U16),
        BEH_MKLDNN_FP16.withIn(Precision::I16),
        // the following list withConfig tests the MKLDNNGraphlessInferRequest (explicitly created for streams)
        BEH_MKLDNN_FP16.withIn(Precision::FP32).withConfig({{KEY_CPU_THROUGHPUT_STREAMS, CPU_THROUGHPUT_AUTO}}),
        BEH_MKLDNN_FP16.withIn(Precision::U8).withConfig({{KEY_CPU_THROUGHPUT_STREAMS, CPU_THROUGHPUT_AUTO}}),
        BEH_MKLDNN_FP16.withIn(Precision::U16).withConfig({{KEY_CPU_THROUGHPUT_STREAMS, CPU_THROUGHPUT_AUTO}}),
        BEH_MKLDNN_FP16.withIn(Precision::I16).withConfig({{KEY_CPU_THROUGHPUT_STREAMS, CPU_THROUGHPUT_AUTO}}),
        BEH_MULTI,
        BEH_MULTI.withIn(Precision::U8),
        BEH_MULTI.withIn(Precision::U16),
        BEH_MULTI.withIn(Precision::I16),
        BEH_MULTI.withIn(Precision::U8).withConfig({{MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, "CPU"},
                                                    {KEY_CPU_THROUGHPUT_STREAMS,                           CPU_THROUGHPUT_AUTO}}),
        BEH_MULTI.withIn(Precision::U16).withConfig({{MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, "CPU"},
                                                     {KEY_CPU_THROUGHPUT_STREAMS,                           CPU_THROUGHPUT_AUTO}}),
        BEH_MULTI.withIn(Precision::I16).withConfig({{MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, "CPU"},
                                                     {KEY_CPU_THROUGHPUT_STREAMS,                           CPU_THROUGHPUT_AUTO}}),
};

const BehTestParams allOutputSupportedValues[] = {
        BEH_MKLDNN,
        // the following withConfig test checks the MKLDNNGraphlessInferRequest (explicitly created for streams)
        BEH_MKLDNN.withConfig({{KEY_CPU_THROUGHPUT_STREAMS, CPU_THROUGHPUT_AUTO}}),
        BEH_MULTI.withOut(Precision::FP32),
        BEH_MULTI.withConfig({{MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, "CPU"},
                              {KEY_CPU_THROUGHPUT_STREAMS,                           CPU_THROUGHPUT_AUTO}}),
};

const BehTestParams typeUnSupportedValues[] = {
        BEH_MKLDNN.withIn(Precision::Q78),
        BEH_MKLDNN_FP16,
        BEH_MULTI.withIn(Precision::Q78),
};

const BehTestParams allUnSupportedValues[] = {
        BEH_US_ALL_MKLDNN,
        BEH_US_ALL_MULTI,
};

const std::vector<BehTestParams> withCorrectConfValues = {
        BEH_MKLDNN.withConfig({{KEY_CPU_THROUGHPUT_STREAMS, CPU_THROUGHPUT_NUMA}}),
        BEH_MKLDNN.withConfig({{KEY_CPU_THROUGHPUT_STREAMS, CPU_THROUGHPUT_AUTO}}),
        BEH_MKLDNN.withConfig({{KEY_CPU_THROUGHPUT_STREAMS, "8"}}),
        BEH_MKLDNN.withConfig({{KEY_CPU_BIND_THREAD, NO}}),
        BEH_MKLDNN.withConfig({{KEY_CPU_BIND_THREAD, YES}}),
        BEH_MKLDNN.withConfig({{KEY_DYN_BATCH_LIMIT, "10"}}),
        BEH_MULTI.withConfig({{MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, "CPU"},
                              {KEY_CPU_THROUGHPUT_STREAMS,                           CPU_THROUGHPUT_NUMA}}),
        BEH_MULTI.withConfig({{MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, "CPU"},
                              {KEY_CPU_THROUGHPUT_STREAMS,                           CPU_THROUGHPUT_AUTO}}),
        BEH_MULTI.withConfig({{MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, "CPU"},
                              {KEY_CPU_THROUGHPUT_STREAMS,                           "8"}}),
        BEH_MULTI.withConfig({{MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, "CPU"},
                              {KEY_CPU_BIND_THREAD,                                  NO}}),
        BEH_MULTI.withConfig({{MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, "CPU"},
                              {KEY_CPU_BIND_THREAD,                                  YES}}),
        BEH_MULTI.withConfig({{MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, "CPU"},
                              {KEY_DYN_BATCH_LIMIT,                                  "10"}}),
};

const BehTestParams withIncorrectConfValues[] = {
        BEH_MKLDNN.withConfig({{KEY_CPU_THROUGHPUT_STREAMS, "OFF"}}),
        BEH_MKLDNN.withConfig({{KEY_CPU_BIND_THREAD, "OFF"}}),
        BEH_MKLDNN.withConfig({{KEY_DYN_BATCH_LIMIT, "NAN"}}),
        BEH_MULTI.withConfig({{MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, "CPU"},
                              {KEY_CPU_THROUGHPUT_STREAMS,                           "OFF"}}),
        BEH_MULTI.withConfig({{MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, "CPU"},
                              {KEY_CPU_BIND_THREAD,                                  "OFF"}}),
        BEH_MULTI.withConfig({{MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, "CPU"},
                              {KEY_DYN_BATCH_LIMIT,                                  "NAN"}}),
};

const std::vector<BehTestParams> withCorrectConfValuesPluginOnly;

const std::vector<BehTestParams> withCorrectConfValuesNetworkOnly = {
        BEH_MKLDNN.withConfig({}),
        BEH_MULTI
};

const BehTestParams withIncorrectConfKeys[] = {
        BEH_MKLDNN.withIncorrectConfigItem(),
        BEH_MULTI.withIncorrectConfigItem(),
};
