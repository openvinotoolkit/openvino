/*
// Copyright (c) 2019 Intel Corporation
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

#include "reduce_kernel_ref.h"
#include "kernel_selector_utils.h"
#include <vector>
#include <string>

namespace kernel_selector {
ParamsKey ReduceKernelRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::INT32);
    k.EnableInputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT32);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::bfzyx);
    k.EnableOutputLayout(DataLayout::bfzyx);
    k.EnableInputLayout(DataLayout::bfwzyx);
    k.EnableOutputLayout(DataLayout::bfwzyx);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableDifferentTypes();
    return k;
}

CommonDispatchData ReduceKernelRef::SetDefault(const reduce_params& params, const optional_params&) const {
    CommonDispatchData runInfo;

    std::vector<size_t> global = {params.output.LogicalSize(), 1, 1};

    auto local = GetOptimalLocalWorkGroupSizes(global);

    runInfo.gws0 = global[0];
    runInfo.gws1 = global[1];
    runInfo.gws2 = global[2];

    runInfo.lws0 = local[0];
    runInfo.lws1 = local[1];
    runInfo.lws2 = local[2];

    return runInfo;
}

JitConstants ReduceKernelRef::GetJitConstants(const reduce_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    jit.AddConstant(MakeJitConstant("COMPUTATIONAL_OPERATIONS_NUMBER", params.output.LogicalSize()));
    jit.AddConstant(MakeJitConstant("REDUCE_" + toString(params.reduceMode) + "_MODE", 1));
    jit.AddConstant(MakeJitConstant("KEEP_DIMS", params.keepDims));

    auto inputDims = params.inputs[0].LogicalDims();
    std::reverse(inputDims.begin(), inputDims.end());

    auto convertAxesToIE = [&]() -> std::vector<int32_t> {
        std::vector<int32_t> res;
        auto sz = inputDims.size();

        for (size_t i = 0; i < params.reduceAxes.size(); ++i) {
            switch (params.reduceAxes[i]) {
                case 0: res.push_back(0); break;
                case 1: res.push_back(1); break;
                case 2: res.push_back(sz == 6 ? 5 : sz == 5 ? 4 : 3); break;
                case 3: res.push_back(sz == 6 ? 4 : sz == 5 ? 3 : 2); break;
                case 4: res.push_back(sz == 6 ? 3 : 2); break;
                case 5: res.push_back(2); break;
            }
        }
        return res;
    };

    auto getDimSizeNameByNum = [&](int dim) -> std::string {
        if (params.inputs[0].GetLayout() == DataLayout::bfwzyx) {
            switch (dim) {
                case 0: return "BATCH_NUM";
                case 1: return "FEATURE_NUM";
                case 2: return "SIZE_W";
                case 3: return "SIZE_Z";
                case 4: return "SIZE_Y";
                case 5: return "SIZE_X";
            }
        } else if (params.inputs[0].GetLayout() == DataLayout::bfzyx) {
            switch (dim) {
                case 0: return "BATCH_NUM";
                case 1: return "FEATURE_NUM";
                case 2: return "SIZE_Z";
                case 3: return "SIZE_Y";
                case 4: return "SIZE_X";
            }
        } else if (params.inputs[0].GetLayout() == DataLayout::bfyx) {
            switch (dim) {
                case 0: return "BATCH_NUM";
                case 1: return "FEATURE_NUM";
                case 2: return "SIZE_Y";
                case 3: return "SIZE_X";
            }
        }
        return "";
    };

    auto convertedAxes = convertAxesToIE();

    const size_t kept_dims = inputDims.size() - params.reduceAxes.size();
    if (kept_dims == 1) {
        for (size_t i = 0; i < inputDims.size(); ++i)
            if (std::find(convertedAxes.begin(), convertedAxes.end(), i) == convertedAxes.end())
                jit.AddConstant(MakeJitConstant(getDimSizeNameByNum(i) + "_IDX_COMP(index)", "index"));
    } else {
        size_t kept_cnt = 0;
        for (size_t i = 0; i < inputDims.size(); ++i) {
            if (std::find(convertedAxes.begin(), convertedAxes.end(), i) == convertedAxes.end()) {
                if (kept_cnt == 0) {
                    std::string str = "(index ";
                    for (size_t j = i + 1; j < inputDims.size(); ++j) {
                        if (std::find(convertedAxes.begin(), convertedAxes.end(), j) == convertedAxes.end()) {
                            str += "/ INPUT0_" + getDimSizeNameByNum(j);
                        }
                    }
                    str += ")";
                    jit.AddConstant(MakeJitConstant(getDimSizeNameByNum(i) + "_IDX_COMP(index)", str));
                } else if (kept_cnt == kept_dims - 1) {
                    jit.AddConstant(MakeJitConstant(getDimSizeNameByNum(i) + "_IDX_COMP(index)", "(index % INPUT0_" + getDimSizeNameByNum(i) + ")"));
                } else {
                    std::string str = "(index ";
                    for (size_t j = i + 1; j < inputDims.size(); ++j) {
                        if (std::find(convertedAxes.begin(), convertedAxes.end(), j) == convertedAxes.end()) {
                            str += "/ INPUT0_" + getDimSizeNameByNum(j);
                        }
                    }
                    str += " % INPUT0_" + getDimSizeNameByNum(i) + ")";
                    jit.AddConstant(MakeJitConstant(getDimSizeNameByNum(i) + "_IDX_COMP(index)", str));
                }
                kept_cnt += 1;
            }
        }
    }

    for (size_t a = 0; a < params.reduceAxes.size(); a++) {
        switch (params.reduceAxes[a]) {
            case 0: jit.AddConstant(MakeJitConstant("REDUCE_BATCH", 1)); break;
            case 1: jit.AddConstant(MakeJitConstant("REDUCE_FEATURE", 1)); break;
            case 2: jit.AddConstant(MakeJitConstant("REDUCE_X", 1)); break;
            case 3: jit.AddConstant(MakeJitConstant("REDUCE_Y", 1)); break;
            case 4: jit.AddConstant(MakeJitConstant("REDUCE_Z", 1)); break;
            case 5: jit.AddConstant(MakeJitConstant("REDUCE_W", 1)); break;
        }
    }

    return jit;
}

KernelsData ReduceKernelRef::GetKernelsData(const Params& params, const optional_params& options) const {
    KernelData kd = KernelData::Default<reduce_params>(params);
    reduce_params& newParams = *static_cast<reduce_params*>(kd.params.get());

    assert(params.GetType() == KernelType::REDUCE);

    auto runInfo = SetDefault(newParams, options);
    auto entry_point = GetEntryPoint(kernelName, newParams.layerID, options);
    auto cldnn_jit = GetJitConstants(newParams);
    std::string jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];

    FillCLKernelData(kernel, runInfo, params.engineInfo, kernelName, jit, entry_point);

    kd.estimatedTime = DONT_USE_IF_HAVE_SOMETHING_ELSE;

    return {kd};
}
}  // namespace kernel_selector
