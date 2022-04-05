// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "ie_parameter.hpp"

template<class OptionConcept>
struct AsParsedParameterEnabler {
    static InferenceEngine::Parameter asParameter(const std::string& value) { return {OptionConcept::parse(value)}; }
};

struct AsParameterEnabler {
    static InferenceEngine::Parameter asParameter(const std::string& value);
};
