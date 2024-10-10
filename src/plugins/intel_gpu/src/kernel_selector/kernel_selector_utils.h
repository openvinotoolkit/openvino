// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "jitter.h"
#include <vector>

namespace kernel_selector {
struct weight_bias_params;
struct WeightsReorderParams;

struct DimensionAccessHelperBase {
    explicit DimensionAccessHelperBase(const DataTensor& t) {
        dims = {
            t.Batch(),
            t.Feature(),
            t.U(),
            t.V(),
            t.W(),
            t.Z(),
            t.Y(),
            t.X(),
        };
    }

    Tensor::Dim& x_dim() { return dims[7]; }
    Tensor::Dim& y_dim() { return dims[6]; }
    Tensor::Dim& z_dim() { return dims[5]; }
    Tensor::Dim& w_dim() { return dims[4]; }
    Tensor::Dim& v_dim() { return dims[3]; }
    Tensor::Dim& u_dim() { return dims[2]; }
    Tensor::Dim& f_dim() { return dims[1]; }
    Tensor::Dim& b_dim() { return dims[0]; }

    std::vector<Tensor::Dim> dims;
};

struct DimensionAccessHelperJit : virtual DimensionAccessHelperBase {
    explicit DimensionAccessHelperJit(const DataTensor& t, bool padded = false)
    : DimensionAccessHelperBase(t) {
        size_t dyn_shape_offset = t.get_dynamic_shape_offset();
        size_t dyn_pad_offset = dyn_shape_offset + DataTensor::max_rank();
        has_dynamic_pad = false;
        for (auto d : dims) {
            dims_sizes.push_back(toCodeString(d, dyn_shape_offset, padded, d.pad.is_dynamic, dyn_pad_offset));
            dyn_shape_offset++;
            if (padded) {
                if (d.pad.is_dynamic) {
                    pad_before_after_sizes.push_back("(shape_info[" + std::to_string(dyn_pad_offset++) + "])");
                    pad_before_after_sizes.push_back("(shape_info[" + std::to_string(dyn_pad_offset++) + "])");
                    has_dynamic_pad = true;
                } else {
                    pad_before_after_sizes.push_back(toCodeString(d.pad.before));
                    pad_before_after_sizes.push_back(toCodeString(d.pad.after));
                }
            }
        }
    }

    std::string x() { return dims_sizes[7]; }
    std::string y() { return dims_sizes[6]; }
    std::string z() { return dims_sizes[5]; }
    std::string w() { return dims_sizes[4]; }
    std::string v() { return dims_sizes[3]; }
    std::string u() { return dims_sizes[2]; }
    std::string f() { return dims_sizes[1]; }
    std::string b() { return dims_sizes[0]; }
    std::pair<std::string, std::string> x_pad() { return {pad_before_after_sizes[14], pad_before_after_sizes[15]}; }
    std::pair<std::string, std::string> y_pad() { return {pad_before_after_sizes[12], pad_before_after_sizes[13]}; }
    std::pair<std::string, std::string> z_pad() { return {pad_before_after_sizes[10], pad_before_after_sizes[11]}; }
    std::pair<std::string, std::string> w_pad() { return {pad_before_after_sizes[8], pad_before_after_sizes[9]}; }
    std::pair<std::string, std::string> v_pad() { return {pad_before_after_sizes[6], pad_before_after_sizes[7]}; }
    std::pair<std::string, std::string> u_pad() { return {pad_before_after_sizes[4], pad_before_after_sizes[5]}; }
    std::pair<std::string, std::string> f_pad() { return {pad_before_after_sizes[2], pad_before_after_sizes[3]}; }
    std::pair<std::string, std::string> b_pad() { return {pad_before_after_sizes[0], pad_before_after_sizes[1]}; }

    std::vector<std::string> dims_sizes;
    std::vector<std::string> pad_before_after_sizes;
    bool has_dynamic_pad;
};

std::vector<size_t> GetImageSizes(const kernel_selector::WeightsTensor& dimensions, const WeightsLayout layout);
bool CheckImageSize(const weight_bias_params& newParams, const WeightsLayout layout);
bool UpdateWeightsParams(weight_bias_params& newParams,
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
