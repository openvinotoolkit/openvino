// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior_test_plugin.h"
#include <cldnn/cldnn_config.hpp>

// correct params
#define BEH_CLDNN BehTestParams("GPU", \
                                FuncTestUtils::TestModel::convReluNormPoolFcModelFP32.model_xml_str, \
                                FuncTestUtils::TestModel::convReluNormPoolFcModelFP32.weights_blob, \
                                Precision::FP32)
#define BEH_HETERO BehTestParams("HETERO", \
                                 FuncTestUtils::TestModel::convReluNormPoolFcModelFP32.model_xml_str, \
                                 FuncTestUtils::TestModel::convReluNormPoolFcModelFP32.weights_blob, \
                                 Precision::FP32)

// all parameters are unsupported - reversed
#define BEH_US_ALL_CLDNN   BehTestParams("GPU", \
                                         FuncTestUtils::TestModel::convReluNormPoolFcModelQ78.model_xml_str, \
                                         FuncTestUtils::TestModel::convReluNormPoolFcModelQ78.weights_blob, \
                                         Precision::Q78)

const BehTestParams supportedValues[] = {
        BEH_CLDNN,
};

const BehTestParams requestsSupportedValues[] = {
        BEH_CLDNN,
};

const BehTestParams allInputSupportedValues[] = {
        BEH_CLDNN, BEH_CLDNN.withIn(Precision::FP16), BEH_CLDNN.withIn(Precision::U8), BEH_CLDNN.withIn(Precision::I16),
        BEH_CLDNN.withIn(Precision::I32),
        BEH_CLDNN.withIn(Precision::U8).withConfig({{KEY_GPU_THROUGHPUT_STREAMS, GPU_THROUGHPUT_AUTO}}),
        BEH_CLDNN.withIn(Precision::FP16).withConfig({{KEY_GPU_THROUGHPUT_STREAMS, GPU_THROUGHPUT_AUTO}}),
        BEH_CLDNN.withIn(Precision::I16).withConfig({{KEY_GPU_THROUGHPUT_STREAMS, GPU_THROUGHPUT_AUTO}}),
        BEH_CLDNN.withIn(Precision::I32).withConfig({{KEY_GPU_THROUGHPUT_STREAMS, GPU_THROUGHPUT_AUTO}}),
};

const BehTestParams allOutputSupportedValues[] = {
        BEH_CLDNN, BEH_CLDNN.withOut(Precision::FP16),
        BEH_CLDNN.withIn(Precision::FP16).withConfig({{KEY_GPU_THROUGHPUT_STREAMS, GPU_THROUGHPUT_AUTO}}),
};

const BehTestParams typeUnSupportedValues[] = {
        BEH_CLDNN.withIn(Precision::Q78), BEH_CLDNN.withIn(Precision::I8),
};

const BehTestParams allUnSupportedValues[] = {
        BEH_US_ALL_CLDNN,
};

const std::vector<BehTestParams> withCorrectConfValues = {
        BEH_CLDNN.withConfig({{KEY_GPU_THROUGHPUT_STREAMS, GPU_THROUGHPUT_AUTO}}),
        BEH_CLDNN.withConfig({{KEY_GPU_THROUGHPUT_STREAMS, "2"}}),
        BEH_CLDNN.withConfig({{KEY_PERF_COUNT, NO}}),
        /*BEH_CLDNN.withConfig({ { KEY_PERF_COUNT, YES } }),*/
        BEH_CLDNN.withConfig({{KEY_DUMP_KERNELS, NO}}),
        BEH_CLDNN.withConfig({{KEY_DUMP_KERNELS, YES}}),
        BEH_CLDNN.withConfig({{KEY_TUNING_MODE, TUNING_DISABLED}}),
//         Too long inference of AlexNet (980 secs)
        BEH_CLDNN.withConfig({{KEY_TUNING_MODE, TUNING_CREATE},
                              {KEY_TUNING_FILE, "tfile"}}),
        BEH_CLDNN.withConfig({{KEY_DEVICE_ID, "0"}}),
};

const BehTestParams withIncorrectConfValues[] = {
        BEH_CLDNN.withConfig({{KEY_GPU_THROUGHPUT_STREAMS, "OFF"}}),
        BEH_CLDNN.withConfig({{KEY_PERF_COUNT, "ON"}}),
        BEH_CLDNN.withConfig({{KEY_CONFIG_FILE, "unknown_file"}}),
        BEH_CLDNN.withConfig({{KEY_DUMP_KERNELS, "ON"}}),
        BEH_CLDNN.withConfig({{KEY_TUNING_MODE, "TUNING_UNKNOWN_MODE"}}),
        BEH_CLDNN.withConfig({{KEY_DEVICE_ID, "DEVICE_UNKNOWN"}}),
        // FIXME: [IE clDNN] The plugin doesn't throw GENERAL_ERROR if use non-exist tuning file. CVS-8593
        //BEH_CLDNN.withConfig({ { KEY_TUNING_MODE, TUNING_USE_EXISTING },
        //                       { KEY_TUNING_FILE, "unknown_file" } }),
};

const std::vector<BehTestParams> withCorrectConfValuesNetworkOnly = {
        BEH_CLDNN.withConfig({{}}),
};

const BehTestParams withIncorrectConfKeys[] = {
        BEH_CLDNN.withIncorrectConfigItem(),
};
