// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu_classification_case.hpp"

//------------------------------------------------------------------------------
// Implementation of methods of class VpuNoClassificationRegression
//------------------------------------------------------------------------------

std::string VpuNoClassificationRegression::getTestCaseName(
        TestParamInfo<ClassificationTestVpuParam::ParamType> param) {
    return VpuNoRegressionBase::getTestCaseName(get<0>(param.param),
                                                get<1>(param.param),
                                                get<2>(param.param),
                                                get<3>(param.param)) +
            "_SHAVES=" + (get<4>(param.param) == -1 ? "AUTO" : std::to_string(get<4>(param.param))) +
           "_IsIgnoreStatistic=" + std::to_string(get<5>(param.param)) +
           "_" + get<6>(param.param).name();
}

void  VpuNoClassificationRegression::SetUp() {
    TestsCommon::SetUp();

    plugin_name_ = get<0>(ClassificationTestVpuParam::GetParam()).first;
    device_name_ = get<0>(ClassificationTestVpuParam::GetParam()).second;
    in_precision_= get<1>(ClassificationTestVpuParam::GetParam());
    batch_= get<2>(ClassificationTestVpuParam::GetParam());
    do_reshape_= get<3>(ClassificationTestVpuParam::GetParam());
    resources_= get<4>(ClassificationTestVpuParam::GetParam());
    is_ignore_statistic_ = get<5>(ClassificationTestVpuParam::GetParam());
    source_param_= get<6>(ClassificationTestVpuParam::GetParam());

    InitConfig();
}

void VpuNoClassificationRegression::InitConfig() {
    VpuNoRegressionBase::InitConfig();

    if (resources_ != -1) {
        config_["VPU_NUMBER_OF_CMX_SLICES"] = std::to_string(resources_);
        config_["VPU_NUMBER_OF_SHAVES"] = std::to_string(resources_);
    }

    if (is_ignore_statistic_) {
        config_["VPU_IGNORE_IR_STATISTIC"] = CONFIG_VALUE(YES);
    } else {
        config_["VPU_IGNORE_IR_STATISTIC"] = CONFIG_VALUE(NO);
    }
}

//------------------------------------------------------------------------------
// Implementation of methods of class VpuNoClassificationRegressionSpecific
//------------------------------------------------------------------------------

std::string VpuNoClassificationRegressionSpecific::getTestCaseName(
        TestParamInfo<ClassificationSpecificTestVpuParam::ParamType> param) {
    return VpuNoRegressionBase::getTestCaseName(get<0>(param.param),
                                                get<1>(param.param),
                                                get<2>(param.param),
                                                get<3>(param.param));
}

void  VpuNoClassificationRegressionSpecific::SetUp() {
    TestsCommon::SetUp();

    plugin_name_ = get<0>(ClassificationSpecificTestVpuParam::GetParam()).first;
    device_name_ = get<0>(ClassificationSpecificTestVpuParam::GetParam()).second;
    in_precision_= get<1>(ClassificationSpecificTestVpuParam::GetParam());
    batch_= get<2>(ClassificationSpecificTestVpuParam::GetParam());
    do_reshape_= get<3>(ClassificationSpecificTestVpuParam::GetParam());

    InitConfig();
}

void VpuNoClassificationRegressionSpecific::InitConfig() {
    VpuNoRegressionBase::InitConfig();
}

//------------------------------------------------------------------------------
// Implementation of methods of class VpuNoRegressionWithCompilation
//------------------------------------------------------------------------------

std::string VpuNoRegressionWithCompilation::getTestCaseName(
        TestParamInfo <CompilationTestParam::ParamType> param) {
    return "plugin=" + get<0>(param.param).first +
           std::string("_") + "device=" + get<0>(param.param).second +
           std::string("_") + get<1>(param.param).name();
}

void VpuNoRegressionWithCompilation::SetUp() {
    plugin_name_ = get<0>(CompilationTestParam::GetParam()).first;
    device_name_ = get<0>(CompilationTestParam::GetParam()).second;
    compilation_param_ = get<1>(CompilationTestParam::GetParam());

    PluginCache::get().reset();
}
