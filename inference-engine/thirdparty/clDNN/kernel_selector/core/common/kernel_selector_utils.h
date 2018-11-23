/*
// Copyright (c) 2016 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#pragma once

#include "jitter.h"
#include "tensor_type.h"

namespace kernel_selector {

    struct weight_bias_params;
    struct convolution_params;

    bool CheckConvolutionPaddedInputDesc(const convolution_params& params, const DataTensor& reqDesc);
    DataTensor GetConvolutionBFYXPaddedTensor(const convolution_params& cp);
    bool CovolutionCheckInput(const Params& p, const optional_params& o);
    bool CovolutionUpdateInputParams(convolution_params& params);
    WeightsType DataTypeToWeightsType(Datatype t);
    bool CheckWeights(const WeightsTensor& tensor, WeightsType reqType, std::vector<WeightsLayout> reqLayouts);
    std::vector<size_t> GetImageSizes(const kernel_selector::WeightsTensor& dimensions, const WeightsLayout layout);
    bool CheckImageSize(const weight_bias_params& newParams, const WeightsLayout layout);
    bool UpdateWeightsParams(weight_bias_params& newParams, const optional_params& options, std::vector<WeightsLayout> layouts, WeightsReorderParams& weightsReorderParams);
    JitConstants GetTensorFriendlyWorkGroupsJit(const DataTensor& t);
    std::vector<size_t> GetTensorFriendlyWorkGroups(const DataTensor& t);
    std::vector<size_t> GetOptimalLocalWorkGroupSizes(std::vector<size_t> gws);
    bool CheckInputsOutputNoPitchSameDims(const base_params& params);
}