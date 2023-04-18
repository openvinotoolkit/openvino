// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "jitter.h"
#include <vector>

namespace kernel_selector {
struct weight_bias_params;
struct optional_params;
struct WeightsReorderParams;

struct DimensionAccessHelper {
    static size_t dynamic_dimension_offset(Tensor::DataChannelName c, size_t tensor_id = 0) {
        const auto shape_info_layout = DataLayout::bfvuwzyx;
        auto idx = DataTensor::max_rank() - DataTensor::Channelndex(shape_info_layout, c) - 1;
        return DataTensor::max_rank() * tensor_id + idx;
    }

    explicit DimensionAccessHelper(const DataTensor& t, size_t dyn_tensor_id = 0, bool padded = false)
    : x(toCodeString(t.X(),       dynamic_dimension_offset(Tensor::DataChannelName::X, dyn_tensor_id), padded))
    , y(toCodeString(t.Y(),       dynamic_dimension_offset(Tensor::DataChannelName::Y, dyn_tensor_id), padded))
    , z(toCodeString(t.Z(),       dynamic_dimension_offset(Tensor::DataChannelName::Z, dyn_tensor_id), padded))
    , w(toCodeString(t.W(),       dynamic_dimension_offset(Tensor::DataChannelName::W, dyn_tensor_id), padded))
    , u(toCodeString(t.U(),       dynamic_dimension_offset(Tensor::DataChannelName::U, dyn_tensor_id), padded))
    , v(toCodeString(t.V(),       dynamic_dimension_offset(Tensor::DataChannelName::V, dyn_tensor_id), padded))
    , f(toCodeString(t.Feature(), dynamic_dimension_offset(Tensor::DataChannelName::FEATURE, dyn_tensor_id), padded))
    , b(toCodeString(t.Batch(),   dynamic_dimension_offset(Tensor::DataChannelName::BATCH, dyn_tensor_id), padded)) { }

    std::string x, y, z, w, u, v, f, b;
};

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
}  // namespace kernel_selector
