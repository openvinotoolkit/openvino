// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <string>
#include <vector>

#include "Python.h"
#include "ie_api_impl.hpp"

namespace InferenceEnginePython {

void ApplyMOCTransformations(InferenceEnginePython::IENetwork network, bool cf);

void ApplyPOTTransformations(InferenceEnginePython::IENetwork network, std::string device);

void ApplyLowLatencyTransformation(InferenceEnginePython::IENetwork network, bool use_const_initializer = true);

void ApplyPruningTransformation(InferenceEnginePython::IENetwork network);

void GenerateMappingFile(InferenceEnginePython::IENetwork network, std::string path, bool extract_names);

/// TODO: this is helper class to create ngraph::Constant as it is not exposed to python via 'cython'
/// This class shall not be used when migrated to 'pybind11' (task 33021)
struct ConstantInfo {
    ConstantInfo(const std::vector<float>& data_, int axis_, int shape_size_): data(data_), axis(axis_), shape_size(shape_size_) {}
    ConstantInfo() = default;
    std::vector<float> data = {};
    int axis = 0;
    int shape_size = 0;  // for {1,3,1,1} shape shape_size shall be 4
};

using ConstantInfoPtr = std::shared_ptr<ConstantInfo>;

ConstantInfoPtr CreateConstantInfo(const std::vector<float>& data_, int axis_, int shape_size_);
ConstantInfoPtr CreateEmptyConstantInfo();

void ApplyScaleInputs(InferenceEnginePython::IENetwork network, const std::map<std::string, ConstantInfoPtr>& values);

void ApplySubtractMeanInputs(InferenceEnginePython::IENetwork network, const std::map<std::string, ConstantInfoPtr>& values);

void CheckAPI();

};  // namespace InferenceEnginePython
