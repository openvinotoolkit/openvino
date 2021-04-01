// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu_case_common.hpp"

bool CheckMyriadX() {
    if (auto envVar = std::getenv("IE_VPU_MYRIADX")) {
        return std::stoi(envVar) != 0;
    }
    return false;
}

bool CheckMA2085() {
    if (auto envVar = std::getenv("IE_VPU_MA2085")) {
        return std::stoi(envVar) != 0;
    }
    return false;
}

//------------------------------------------------------------------------------
// Implementation of methods of class VpuNoRegressionBase
//------------------------------------------------------------------------------

std::string VpuNoRegressionBase::getTestCaseName(PluginDevicePair plugin_device_names,
                                                 Precision precision,
                                                 int batch,
                                                 bool do_reshape) {
    return "plugin=" + plugin_device_names.first +
           "_device=" + plugin_device_names.second +
           "_InPrecision=" + precision.name() +
           "_Batch=" + std::to_string(batch) +
           "_DoReshape=" + std::to_string(do_reshape);
}

std::string VpuNoRegressionBase::getDeviceName() const {
    return device_name_;
}

void VpuNoRegressionBase::InitConfig() {
    config_[CONFIG_KEY(LOG_LEVEL)] = CONFIG_VALUE(LOG_INFO);
}
