// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <map>
#include <list>
#include <unordered_map>
#include <string>
#include "backend/dnn_types.h"

namespace GNAPluginNS {
struct InputDesc {
    std::unordered_map<std::string, intel_dnn_orientation_t> orientation_in;
    /// order of scale factors matches inputs order in original topology
    std::vector<float> inputScaleFactors;
    std::map<std::string, int> bytes_allocated_for_input;
    std::unordered_map<std::string, std::list<std::vector<void *>>::iterator> ptr_inputs_global_id;
    std::list<std::vector<void *>> ptr_inputs_global_storage;

    std::vector<void *>& getPtrInputsGlobal(const std::string& name);
    intel_dnn_orientation_t getOrientation(const std::string& name);
    float getScaleFactor(std::size_t index);
};
}  // namespace GNAPluginNS
