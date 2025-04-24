// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reduce_kernel_base.h"
#include "kernel_selector_utils.h"
#include <vector>
#include <string>
#include "common_tools.h"

namespace kernel_selector {

bool ReduceKernelBase::Validate(const Params& p) const {
    auto& params = dynamic_cast<const reduce_params&>(p);

    if (params.GetType() != KernelType::REDUCE) {
        return false;
    }

    for (auto& fused_op : params.fused_ops) {
        if (!IsFusedPrimitiveSupported(fused_op))
            return false;
    }

    return true;
}

JitConstants ReduceKernelBase::GetJitConstants(const reduce_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    const auto& output = params.outputs[0];
    if (output.is_dynamic()) {
        DimensionAccessHelperJit dims(output);
        jit.AddConstant(MakeJitConstant("COMPUTATIONAL_OPERATIONS_NUMBER", toVectorMulString({dims.x(),
                                                                                              dims.y(),
                                                                                              dims.z(),
                                                                                              dims.w(),
                                                                                              dims.u(),
                                                                                              dims.v(),
                                                                                              dims.f(),
                                                                                              dims.b()})));
    } else {
        jit.AddConstant(MakeJitConstant("COMPUTATIONAL_OPERATIONS_NUMBER", params.outputs[0].LogicalSize()));
    }

    jit.AddConstant(MakeJitConstant("REDUCE_" + toString(params.reduceMode) + "_MODE", 1));
    jit.AddConstant(MakeJitConstant("KEEP_DIMS", params.keepDims));

    auto inputDims = params.inputs[0].LogicalDims();
    std::reverse(inputDims.begin(), inputDims.end());

    auto convertAxesToIE = [&]() -> std::vector<int32_t> {
        std::vector<int32_t> res;
        auto sz = inputDims.size();

        for (size_t i = 0; i < params.reduceAxes.size(); ++i) {
            auto axis = params.reduceAxes[i];
            if (axis < 2) {
                res.push_back(axis);
            } else {
                res.push_back(static_cast<int32_t>(sz + 1 - axis));
            }
        }
        return res;
    };

    auto getDimSizeNameByNum = [&](size_t dim) -> std::string {
        if (params.inputs[0].Dimentions() == 8) {
            switch (dim) {
                case 0: return "BATCH_NUM";
                case 1: return "FEATURE_NUM";
                case 2: return "SIZE_V";
                case 3: return "SIZE_U";
                case 4: return "SIZE_W";
                case 5: return "SIZE_Z";
                case 6: return "SIZE_Y";
                case 7: return "SIZE_X";
            }
        } else if (params.inputs[0].Dimentions() == 7) {
            switch (dim) {
                case 0: return "BATCH_NUM";
                case 1: return "FEATURE_NUM";
                case 2: return "SIZE_U";
                case 3: return "SIZE_W";
                case 4: return "SIZE_Z";
                case 5: return "SIZE_Y";
                case 6: return "SIZE_X";
            }
        } else  if (params.inputs[0].Dimentions() == 6) {
            switch (dim) {
                case 0: return "BATCH_NUM";
                case 1: return "FEATURE_NUM";
                case 2: return "SIZE_W";
                case 3: return "SIZE_Z";
                case 4: return "SIZE_Y";
                case 5: return "SIZE_X";
            }
        } else if (params.inputs[0].Dimentions() == 5) {
            switch (dim) {
                case 0: return "BATCH_NUM";
                case 1: return "FEATURE_NUM";
                case 2: return "SIZE_Z";
                case 3: return "SIZE_Y";
                case 4: return "SIZE_X";
            }
        } else if (params.inputs[0].Dimentions() == 4) {
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

    std::string divider;
    for (size_t i = 0; i < params.reduceAxes.size(); ++i) {
        divider += "INPUT0_" + getDimSizeNameByNum(convertedAxes[i]);
        size_t range_check = i;
        if (++range_check < params.reduceAxes.size())
            divider += "*";
    }
    jit.AddConstant(MakeJitConstant("DIVIDER", divider));

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
                    jit.AddConstant(MakeJitConstant(getDimSizeNameByNum(i) + "_IDX_COMP(index)",
                                                    "(index % INPUT0_" + getDimSizeNameByNum(i) + ")"));
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
            case 0:
                jit.AddConstant(MakeJitConstant("REDUCE_BATCH", 1));
                break;
            case 1:
                jit.AddConstant(MakeJitConstant("REDUCE_FEATURE", 1));
                break;
            case 2:
                jit.AddConstant(MakeJitConstant("REDUCE_X", 1));
                break;
            case 3:
                jit.AddConstant(MakeJitConstant("REDUCE_Y", 1));
                break;
            case 4:
                jit.AddConstant(MakeJitConstant("REDUCE_Z", 1));
                break;
            case 5:
                jit.AddConstant(MakeJitConstant("REDUCE_W", 1));
                break;
            case 6:
                jit.AddConstant(MakeJitConstant("REDUCE_U", 1));
                break;
            case 7:
                jit.AddConstant(MakeJitConstant("REDUCE_V", 1));
                break;
        }
    }

    return jit;
}

Datatype ReduceKernelBase::GetAccumulatorType(const reduce_params& params) const {
    const auto& input_dt = params.inputs[0].GetDType();
    const auto& reduce_mode = params.reduceMode;

    if (reduce_mode == ReduceMode::MAX || reduce_mode == ReduceMode::MIN) {
        return input_dt;
    } else {
        switch (input_dt) {
            case Datatype::F32: return Datatype::F32;
            case Datatype::F16: return Datatype::F32;
            case Datatype::INT8: return Datatype::INT32;
            case Datatype::UINT8: return Datatype::INT32;
            default: return Datatype::F32;
        }
    }
}

Datatype ReduceKernelBase::GetFinalAccumulatorType(const reduce_params& params) const {
    const auto& reduce_mode = params.reduceMode;

    if (reduce_mode == ReduceMode::MEAN ||
        reduce_mode == ReduceMode::LOG_SUM_EXP ||
        reduce_mode == ReduceMode::LOG_SUM ||
        reduce_mode == ReduceMode::L2 ||
        reduce_mode == ReduceMode::L1) {
        return Datatype::F32;
    } else {
        return GetAccumulatorType(params);
    }
}

Datatype ReduceKernelBase::GetActivationType(const reduce_params& params) const {
    if (params.outputs[0].GetDType() == Datatype::F16)
        return Datatype::F16;
    else
        return Datatype::F32;
}

void ReduceKernelBase::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [this](const Params& params, KernelData& kd) {
        const auto& prim_params = static_cast<const reduce_params&>(params);
        auto dispatchData = SetDefault(prim_params);
        OPENVINO_ASSERT(kd.kernels.size() == 1, "[GPU] Invalid kernels size for update dispatch data func");
        kd.kernels[0].params.workGroups.global = dispatchData.gws;
        kd.kernels[0].params.workGroups.local = dispatchData.lws;
        kd.kernels[0].skip_execution = KernelData::SkipKernelExecution(prim_params);
    };
}

KernelsData ReduceKernelBase::GetCommonKernelsData(const Params& p) const {
    if (!Validate(p)) {
        return {};
    }

    const reduce_params& params = static_cast<const reduce_params&>(p);
    DispatchData dispatchData = SetDefault(params);

    KernelData kd = KernelData::Default<reduce_params>(params);

    auto cldnn_jit = GetJitConstants(params);
    auto entry_point = GetEntryPoint(kernelName, params.layerID, params);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    GetUpdateDispatchDataFunc(kd);

    auto& kernel = kd.kernels[0];
    FillCLKernelData(kernel,
                     dispatchData,
                     params.engineInfo,
                     kernelName,
                     jit,
                     entry_point,
                     EXE_MODE_DEFAULT,
                     false,
                     false,
                     1,
                     GetFusedPrimitiveInputsCount(params),
                     1,
                     params.is_shape_agnostic);

    return {kd};
}
}  // namespace kernel_selector
