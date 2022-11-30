// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "jitter.h"
#include "intel_gpu/runtime/utils.hpp"
#include <vector>

using namespace cldnn;

namespace kernel_selector {
struct weight_bias_params;
struct optional_params;
struct WeightsReorderParams;

std::vector<size_t> GetImageSizes(const kernel_selector::WeightsTensor& dimensions, const WeightsLayout layout);
bool CheckImageSize(const weight_bias_params& newParams, const WeightsLayout layout);
bool UpdateWeightsParams(weight_bias_params& newParams,
                         const optional_params& options,
                         WeightsLayout layout,
                         WeightsReorderParams& weightsReorderParams,
                         const ParamsKey& paramsKey = ParamsKey(),
                         size_t groups = 1,
                         bool rotate = false);
JitConstants GetTensorFriendlyWorkGroupsJit(const DataTensor& t);
std::vector<size_t> GetTensorFriendlyWorkGroups(const DataTensor& t);
std::vector<size_t> GetOptimalLocalWorkGroupSizes(std::vector<size_t> gws, const EngineInfo& info,
                                                  DataLayout input_layout = DataLayout::bfyx, DataLayout output_layout = DataLayout::bfyx,
                                                  std::vector<std::vector<Tensor::DataChannelName>> dims_by_gws =
                                                      {{ Tensor::DataChannelName::X, Tensor::DataChannelName::Y },
                                                       { Tensor::DataChannelName::FEATURE },
                                                       { Tensor::DataChannelName::BATCH }});
bool CheckInputsOutputNoPitchSameDims(const base_params& params);

template<typename T = std::uint32_t>
size_t hash_combine_dim_tensor(size_t seed, const DimTensor<T>& dt);
size_t hash_combine_dim(size_t seed, const Tensor::Dim& dim);
size_t hash_combine_dt(size_t seed, const DataTensor& dt);
size_t hash_combine_wt(size_t seed, const WeightsTensor& wt);
size_t hash_combine_usize(size_t s, const kernel_selector::uSize& u_size);
template <typename T>
size_t hash_combine_vec(size_t seed, const std::vector<T> vec) {
    for (auto v : vec) {
        seed = hash_combine(seed, v);
    }
    return seed;
}
}  // namespace kernel_selector
