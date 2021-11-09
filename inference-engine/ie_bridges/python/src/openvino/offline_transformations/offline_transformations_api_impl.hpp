// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <string>

#include "Python.h"
#include "ie_api_impl.hpp"

namespace InferenceEnginePython {

void ApplyMOCTransformations(InferenceEnginePython::IENetwork network, bool cf);

void ApplyPOTTransformations(InferenceEnginePython::IENetwork network, std::string device);

void ApplyLowLatencyTransformation(InferenceEnginePython::IENetwork network, bool use_const_initializer = true);

void ApplyMakeStatefulTransformation(InferenceEnginePython::IENetwork network,
                                     std::map<std::string, std::string>& param_res_names);

void ApplyPruningTransformation(InferenceEnginePython::IENetwork network);

void GenerateMappingFile(InferenceEnginePython::IENetwork network, std::string path, bool extract_names);

void Serialize(InferenceEnginePython::IENetwork network, std::string path_to_xml, std::string path_to_bin);

void CheckAPI();

};  // namespace InferenceEnginePython
