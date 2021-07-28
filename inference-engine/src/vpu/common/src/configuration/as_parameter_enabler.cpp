// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/configuration/as_parameter_enabler.hpp>

InferenceEngine::Parameter AsParameterEnabler::asParameter(const std::string& value) {
    return {value};
}

