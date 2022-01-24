// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpu_case_params.hpp"
#include "vpu_tests_config.hpp"
using PluginNamesVector = std::vector<std::pair<std::string, std::string>>;

//------------------------------------------------------------------------------
// class VpuTestParamsContainer
//------------------------------------------------------------------------------

class VpuTestParamsContainer {
public:
//    inline static const std::vector<CompilationParameter>& compilationParameters();
//    inline static const std::vector<DetectionSrcParam>& detectionSrcParams();
//    inline static const std::vector<DetectionSrcParam>& detectionSrcParamsSmoke();
    inline static const PluginNamesVector& testingPlugin() {
        return testing_plugin_;
    };
private:
//    static std::vector<CompilationParameter> compilation_parameters_;
    static PluginNamesVector testing_plugin_;
//    static std::vector<DetectionSrcParam> detection_src_params_;
//    static std::vector<DetectionSrcParam> detection_src_params_smoke_;
};

//------------------------------------------------------------------------------
// Implementation of inline methods of class VpuTestParamsContainer
//------------------------------------------------------------------------------


