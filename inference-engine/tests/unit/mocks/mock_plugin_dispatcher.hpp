// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ie_plugin_dispatcher.hpp"
#include "ie_plugin_ptr.hpp"
#include <gmock/gmock.h>
#include <string>
#include <vector>

class MockDispatcher : public InferenceEngine::PluginDispatcher {
public:
    MockDispatcher(const std::vector<std::string>& pp) : PluginDispatcher(pp) {}
    MOCK_CONST_METHOD1(getPluginByName, InferenceEngine::InferencePlugin(const std::string& name));
};