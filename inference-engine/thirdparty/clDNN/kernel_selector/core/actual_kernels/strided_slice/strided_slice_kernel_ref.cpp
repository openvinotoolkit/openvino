// Copyright (c) 2019-2020 Intel Corporation
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

#include "strided_slice_kernel_ref.h"
#include "kernel_selector_utils.h"
#include <string>
#include <vector>

namespace kernel_selector {

template<typename T>
static void makeJitConstForParam(JitConstants& jit, const std::string name, const T& vec) {
    jit.AddConstant(MakeJitConstant(name + "_SIZES", vec));
    jit.AddConstant(MakeJitConstant(name + "_BATCH", vec[0]));
    jit.AddConstant(MakeJitConstant(name + "_FEATURE", vec[1]));
    if (vec.size() == 5) {  // BFZYX
        jit.AddConstant(MakeJitConstant(name + "_Z", vec[2]));
        jit.AddConstant(MakeJitConstant(name + "_Y", vec[3]));
        jit.AddConstant(MakeJitConstant(name + "_X", vec[4]));
    } else {  // BFYX
        jit.AddConstant(MakeJitConstant(name + "_Z", 0));
        jit.AddConstant(MakeJitConstant(name + "_Y", vec[2]));
        jit.AddConstant(MakeJitConstant(name + "_X", vec[3]));
    }
}

static size_t GetUsedOutDimsCount(const strided_slice_params& params) {
    auto dims = params.output.GetDims();
    size_t first_non_unit_dim = 0; // order is xy(z)fb, so by default consider that we use all dims
    for (size_t i = 0; i < dims.size(); i++) {
        if (dims[i].v != 1) {
            break;
        }
        first_non_unit_dim = i;
    }
    return dims.size() - first_non_unit_dim;
}

ParamsKey StridedSliceKernelRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::INT32);
    k.EnableInputDataType(Datatype::INT64);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT32);
    k.EnableOutputDataType(Datatype::INT64);
    k.EnableAllInputLayout();
    k.EnableAllOutputLayout();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    return k;
}

bool StridedSliceKernelRef::Validate(const Params& p, const optional_params& o) const {
    if (p.GetType() != KernelType::STRIDED_SLICE || o.GetType() != KernelType::STRIDED_SLICE) {
        return false;
    }

    const strided_slice_params& params = static_cast<const strided_slice_params&>(p);
    if (params.inputs.empty())
        return false;

    if (params.output.Dimentions() > 5 || params.inputs[0].Dimentions() > 5)
        return false;

    bool shrink_mode = std::find(params.shrink_axis_mask.begin(), params.shrink_axis_mask.end(), 1) != params.shrink_axis_mask.end();
    if (shrink_mode) {
        size_t shrinked_axes = std::count_if(params.shrink_axis_mask.begin(), params.shrink_axis_mask.end(), [](const uint8_t& v) {
            return v == 1;
        });
        size_t used_out_dims = GetUsedOutDimsCount(params);

        // Count of actual output dims + count of shrinked axes shouldn't exceed 5 to be able to find input index correctly
        if (used_out_dims + shrinked_axes > 5) {
            return false;
        }
    }
    return true;
}

CommonDispatchData StridedSliceKernelRef::SetDefault(const strided_slice_params& params, const optional_params&) const {
    CommonDispatchData dispatchData;

    // If the new_axis_mask is set, then begin, end, and stride are ignored
    // and a new length 1 dimension is adding. Input data just copying to output
    // TODO: remove data copying in case where only shape size changing
    dispatchData.gws = { params.output.Batch().v,
                         params.output.Feature().v,
                         params.output.Z().v * params.output.Y().v * params.output.X().v };

    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);

    return dispatchData;
}

JitConstants StridedSliceKernelRef::GetJitConstants(const strided_slice_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    makeJitConstForParam(jit, "SLICE_BEGIN", params.striding_params[0]);
    makeJitConstForParam(jit, "SLICE_END", params.striding_params[1]);
    makeJitConstForParam(jit, "SLICE_STEPS", params.striding_params[2]);

    jit.AddConstant(MakeJitConstant(
        "NEW_AXIS_MODE",
        std::find(params.new_axis_mask.begin(), params.new_axis_mask.end(), 1) != params.new_axis_mask.end()));

    bool shrink_mode = std::find(params.shrink_axis_mask.begin(), params.shrink_axis_mask.end(), 1) != params.shrink_axis_mask.end();
    if (shrink_mode) {
        jit.AddConstant(MakeJitConstant("SHRINK_MODE", true));
        makeJitConstForParam(jit, "SHRINK", params.shrink_axis_mask);
        std::vector<std::string> bfzyx_in_order;
        if (params.output.Dimentions() == 5)
            bfzyx_in_order = {"batch", "feature", "z", "y", "x"};
        else
            bfzyx_in_order = {"batch", "feature", "y", "x"};

        // Insert zeroes to indices order for shinked axes
        for (size_t i = 0; i < params.shrink_axis_mask.size(); i++) {
            if (params.shrink_axis_mask[i] == 1) {
                bfzyx_in_order.insert(bfzyx_in_order.begin() + i, "0");
            }
        }

        auto get_input_idx_order = [&](std::vector<std::string> bfzyx_in_order) -> std::string {
            return bfzyx_in_order[0] + "," +
                   bfzyx_in_order[1] + "," +
                   bfzyx_in_order[2] + "," +
                   bfzyx_in_order[3] + "," +
                   bfzyx_in_order[4];
        };
        // Erase indices that exceeds 5d tensor. It should be safe, because we check in Validate method that
        // shrinked axes don't result in too big dims count
        while (bfzyx_in_order.size() > 5) {
            bfzyx_in_order.pop_back();
        }

        jit.AddConstant(MakeJitConstant("INPUT_INDICES_ORDER", get_input_idx_order(bfzyx_in_order)));
    }

    return jit;
}

KernelsData StridedSliceKernelRef::GetKernelsData(const Params& params, const optional_params& options) const {
    if (!Validate(params, options)) {
        return {};
    }

    KernelData kd = KernelData::Default<strided_slice_params>(params);
    strided_slice_params& newParams = *static_cast<strided_slice_params*>(kd.params.get());

    assert(params.GetType() == KernelType::STRIDED_SLICE);

    auto dispatchData = SetDefault(newParams, options);
    auto entry_point = GetEntryPoint(kernelName, newParams.layerID, options);
    auto cldnn_jit = GetJitConstants(newParams);
    std::string jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];

    FillCLKernelData(kernel, dispatchData, params.engineInfo, kernelName, jit, entry_point);

    kd.estimatedTime = DONT_USE_IF_HAVE_SOMETHING_ELSE;

    return {kd};
}
}  // namespace kernel_selector
