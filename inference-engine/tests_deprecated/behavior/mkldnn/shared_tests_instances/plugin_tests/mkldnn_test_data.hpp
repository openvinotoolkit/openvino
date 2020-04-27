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

// all parameters are unsupported - reversed
#define BEH_US_ALL_MKLDNN  BehTestParams("CPU", \
                                         FuncTestUtils::TestModel::convReluNormPoolFcModelQ78.model_xml_str, \
                                         FuncTestUtils::TestModel::convReluNormPoolFcModelQ78.weights_blob, \
                                         Precision::Q78)

const BehTestParams supportedValues[] = {
        BEH_MKLDNN,
};

const BehTestParams requestsSupportedValues[] = {
        BEH_MKLDNN,
        // the following adds additional test the MKLDNNGraphlessInferRequest (explicitly created for streams)
        BEH_MKLDNN.withConfig({{KEY_CPU_THROUGHPUT_STREAMS, CPU_THROUGHPUT_AUTO}}),
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
};

const BehTestParams allOutputSupportedValues[] = {
        BEH_MKLDNN,
        // the following withConfig test checks the MKLDNNGraphlessInferRequest (explicitly created for streams)
        BEH_MKLDNN.withConfig({{KEY_CPU_THROUGHPUT_STREAMS, CPU_THROUGHPUT_AUTO}}),
};

const BehTestParams typeUnSupportedValues[] = {
        BEH_MKLDNN.withIn(Precision::Q78),
        BEH_MKLDNN_FP16,
};

const BehTestParams allUnSupportedValues[] = {
        BEH_US_ALL_MKLDNN,
};

const std::vector<BehTestParams> withCorrectConfValues = {
        BEH_MKLDNN.withConfig({{KEY_CPU_THROUGHPUT_STREAMS, CPU_THROUGHPUT_NUMA}}),
        BEH_MKLDNN.withConfig({{KEY_CPU_THROUGHPUT_STREAMS, CPU_THROUGHPUT_AUTO}}),
        BEH_MKLDNN.withConfig({{KEY_CPU_THROUGHPUT_STREAMS, "8"}}),
        BEH_MKLDNN.withConfig({{KEY_CPU_BIND_THREAD, NO}}),
        BEH_MKLDNN.withConfig({{KEY_CPU_BIND_THREAD, YES}}),
        BEH_MKLDNN.withConfig({{KEY_DYN_BATCH_LIMIT, "10"}}),
};

const BehTestParams withIncorrectConfValues[] = {
        BEH_MKLDNN.withConfig({{KEY_CPU_THROUGHPUT_STREAMS, "OFF"}}),
        BEH_MKLDNN.withConfig({{KEY_CPU_BIND_THREAD, "OFF"}}),
        BEH_MKLDNN.withConfig({{KEY_DYN_BATCH_LIMIT, "NAN"}}),
};

const std::vector<BehTestParams> withCorrectConfValuesPluginOnly;

const std::vector<BehTestParams> withCorrectConfValuesNetworkOnly = {
        BEH_MKLDNN.withConfig({}),
};

const BehTestParams withIncorrectConfKeys[] = {
        BEH_MKLDNN.withIncorrectConfigItem(),
};
