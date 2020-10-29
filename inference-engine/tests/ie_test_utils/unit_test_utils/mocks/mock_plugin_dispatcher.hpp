// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gmock/gmock.h>

#include <string>
#include <vector>

#include "ie_plugin_dispatcher.hpp"

IE_SUPPRESS_DEPRECATED_START
class MockDispatcher : public InferenceEngine::PluginDispatcher {
public:
    explicit MockDispatcher(const std::vector<std::string>& pp) : PluginDispatcher(pp) {}
    MOCK_CONST_METHOD1(getPluginByName, InferenceEngine::InferencePlugin(const std::string& name));
};
IE_SUPPRESS_DEPRECATED_END