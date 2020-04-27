// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior_test_plugin.h"

using TestModel = FuncTestUtils::TestModel::TestModel;

TestModel getGnaCnnModel(InferenceEngine::Precision netPrc);

TestModel getGnaMemoryModel(InferenceEngine::Precision netPrc);

const TestModel gnaCnnModelFP32 = getGnaCnnModel(InferenceEngine::Precision::FP32);
const TestModel gnaMemoryModelFP32 = getGnaMemoryModel(InferenceEngine::Precision::FP32);

// correct params
#define BEH_HETERO BehTestParams("HETERO", \
                                 FuncTestUtils::TestModel::convReluNormPoolFcModelFP32.model_xml_str, \
                                 FuncTestUtils::TestModel::convReluNormPoolFcModelFP32.weights_blob, \
                                 Precision::FP32)
#define BEH_CNN_GNA BehTestParams("GNA", \
                                  gnaCnnModelFP32.model_xml_str, \
                                  gnaCnnModelFP32.weights_blob, \
                                  Precision::FP32)
#define BEH_RNN_GNA BehTestParams("GNA", \
                                  gnaMemoryModelFP32.model_xml_str, \
                                  gnaMemoryModelFP32.weights_blob, \
                                  Precision::FP32)
#define BEH_GNA BEH_RNN_GNA

// all parameters are unsupported - reversed
#define BEH_US_ALL_GNA     BehTestParams("GNA", \
                                         FuncTestUtils::TestModel::convReluNormPoolFcModelFP16.model_xml_str, \
                                         FuncTestUtils::TestModel::convReluNormPoolFcModelFP16.weights_blob, \
                                         Precision::FP16)

const BehTestParams supportedValues[] = {
        BEH_GNA,
};

const BehTestParams requestsSupportedValues[] = {
        BEH_GNA,
};

const BehTestParams allInputSupportedValues[] = {
        BEH_GNA, BEH_GNA.withIn(Precision::U8), BEH_GNA.withIn(Precision::I16),
};

const BehTestParams allOutputSupportedValues[] = {
        BEH_GNA,
};

const BehTestParams typeUnSupportedValues[] = {
        BEH_RNN_GNA.withIn(Precision::FP16), BEH_RNN_GNA.withIn(Precision::Q78), BEH_RNN_GNA.withIn(Precision::I8),
        BEH_RNN_GNA.withIn(Precision::I32),
        BEH_CNN_GNA.withIn(Precision::FP16), BEH_CNN_GNA.withIn(Precision::Q78), BEH_CNN_GNA.withIn(Precision::I8),
        BEH_CNN_GNA.withIn(Precision::I32),
};

const BehTestParams batchUnSupportedValues[] = {
        BEH_RNN_GNA.withBatchSize(2),
        BEH_CNN_GNA.withBatchSize(2),
};

const BehTestParams allUnSupportedValues[] = {
        BEH_US_ALL_GNA,
};

const std::vector<BehTestParams> withCorrectConfValues = {
        BEH_GNA.withConfig({{KEY_GNA_SCALE_FACTOR, "1.0"}}),
        BEH_GNA.withConfig({{KEY_GNA_PRECISION, "I8"}}),
        BEH_GNA.withConfig({{KEY_GNA_FIRMWARE_MODEL_IMAGE, "gfile"}}),
        BEH_GNA.withConfig({{KEY_GNA_DEVICE_MODE, GNA_AUTO}}),
        BEH_GNA.withConfig({{KEY_GNA_DEVICE_MODE, GNA_SW_FP32}}),
        BEH_GNA.withConfig({{KEY_GNA_DEVICE_MODE, GNA_SW}}),
        BEH_GNA.withConfig({{KEY_GNA_DEVICE_MODE, GNA_SW_EXACT}}),
        BEH_GNA.withConfig({{KEY_GNA_COMPACT_MODE, NO}}),
};

const std::vector<BehTestParams> withGnaHwConfValue = {
        BEH_GNA.withConfig({{KEY_GNA_DEVICE_MODE, GNA_HW}}),
};

const BehTestParams withIncorrectConfValues[] = {
        BEH_GNA.withConfig({{KEY_GNA_DEVICE_MODE,   GNA_SW_FP32},
                            {KEY_GNA_LIB_N_THREADS, "2"}}),
        BEH_GNA.withConfig({{KEY_GNA_SCALE_FACTOR, "NAN"}}),
        BEH_GNA.withConfig({{KEY_GNA_PRECISION, "FP8"}}),
        BEH_GNA.withConfig({{KEY_GNA_DEVICE_MODE, "AUTO"}}),
        BEH_GNA.withConfig({{KEY_GNA_COMPACT_MODE, "ON"}}),
};

const BehTestParams withIncorrectConfKeys[] = {
        BEH_GNA.withIncorrectConfigItem(),
        BEH_GNA.withConfigItem({"KEY_KEY_GNA_DEVICE_MODE", GNA_SW}),
        BEH_GNA.withConfigItem({"GNA_DEVICE_MODE_XYZ", GNA_SW}),
        BEH_GNA.withConfigItem({"KEY_GNA_DEVICE_MODE_XYZ", GNA_SW}),
        BEH_GNA.withConfigItem({"KEY_GNA_SCALE_FACTOR_1", GNA_SW}),
};
