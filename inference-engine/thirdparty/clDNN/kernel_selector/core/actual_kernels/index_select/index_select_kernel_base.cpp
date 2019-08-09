// Copyright (c) 2018-2019 Intel Corporation
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

#include "index_select_kernel_base.h"

#include "kernel_selector_utils.h"
#include <string>
#include <vector>

namespace kernel_selector {
JitConstants IndexSelectKernelBase::GetJitConstants(const index_select_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    jit.AddConstant(MakeJitConstant("AXES_NUMBER", params.axes.size()));

    if (params.reverse) {
        jit.AddConstant(MakeJitConstant("REVERSE", 1));
    }

    for (size_t i = 0; i < params.axes.size(); i++) {
        std::string size_name = "REVERSE_AXIS_SIZE";
        size_t size_value = 0;
        if (params.axes.size() > 1) {
            std::stringstream ss;
            ss << "REVERSE_" << toString(params.axes[i]) << "_SIZE";
            size_name = ss.str();
        }
        jit.AddConstant(MakeJitConstant(toString(params.axes[i]), ""));
        if (params.reverse) {
            if (params.axes[i] == IndexSelectAxis::BATCH) {
                size_value = params.inputs.at(0).Batch().v;
            } else if (params.axes[i] == IndexSelectAxis::X) {
                size_value = params.inputs.at(0).X().v;
            } else if (params.axes[i] == IndexSelectAxis::Y) {
                size_value = params.inputs.at(0).Y().v;
            } else if (params.axes[i] == IndexSelectAxis::FEATURE) {
                size_value = params.inputs.at(0).Feature().v;
            }
        }
        jit.AddConstant(MakeJitConstant(size_name, size_value));
    }

    return jit;
}

IndexSelectKernelBase::DispatchData IndexSelectKernelBase::SetDefault(const index_select_params& params) {
    const auto& output = params.output;
    DispatchData kd;

    kd.fp16UnitUsed = params.inputs[0].GetDType() == Datatype::F16;

    std::vector<size_t> global;

    if (params.axes.size() == 1) {
        if (params.reverse) {
            if (params.axes[0] == IndexSelectAxis::BATCH) {
                global = {1, params.inputs.at(0).Batch().v, output.Feature().v};
            } else if (params.axes[0] == IndexSelectAxis::X) {
                global = {output.Batch().v, params.inputs.at(0).X().v, output.Feature().v};
            } else if (params.axes[0] == IndexSelectAxis::Y) {
                global = {output.Batch().v, params.inputs.at(0).Y().v, output.Feature().v};
            } else if (params.axes[0] == IndexSelectAxis::FEATURE) {
                global = {output.Batch().v, params.inputs.at(0).Feature().v, output.Y().v};
            }
        } else {
            const auto indices = params.inputs.at(1).X().v;

            if (params.axes[0] == IndexSelectAxis::BATCH) {
                global = {1, indices, output.Feature().v};
            } else if (params.axes[0] == IndexSelectAxis::X || params.axes[0] == IndexSelectAxis::Y) {
                global = {output.Batch().v, indices, output.Feature().v};
            } else if (params.axes[0] == IndexSelectAxis::FEATURE) {
                global = {output.Batch().v, indices, output.Y().v};
            }
        }
    } else {
        if (params.reverse) {
            global = {output.Batch().v, output.Y().v, output.Feature().v};
        }
    }

    const auto& local = GetOptimalLocalWorkGroupSizes(global);

    kd.gws0 = global[0];
    kd.gws1 = global[1];
    kd.gws2 = global[2];

    kd.lws0 = local[0];
    kd.lws1 = local[1];
    kd.lws2 = local[2];

    return kd;
}

KernelsData IndexSelectKernelBase::GetCommonKernelsData(const Params& params,
                                                        const optional_params& options,
                                                        float estimated_time) const {
    assert(params.GetType() == KernelType::INDEX_SELECT);

    const auto& prim_params =
        static_cast<const index_select_params&>(params);

    auto run_info = SetDefault(prim_params);
    KernelData k_data = KernelData::Default<index_select_params>(params);

    auto cldnn_jit = GetJitConstants(prim_params);
    auto entry_point = GetEntryPoint(kernelName, prim_params.layerID, options);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = k_data.kernels[0];
    FillCLKernelData(kernel,
                     run_info,
                     params.engineInfo,
                     kernelName,
                     jit,
                     entry_point,
                     DEFAULT,
                     false,
                     false,
                     (uint32_t)prim_params.inputs.size());

    k_data.estimatedTime = estimated_time;

    return {k_data};
}
}  // namespace kernel_selector
