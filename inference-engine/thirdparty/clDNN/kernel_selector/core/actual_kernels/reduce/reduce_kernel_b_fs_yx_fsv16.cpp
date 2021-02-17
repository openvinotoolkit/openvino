/*
// Copyright (c) 2020 Intel Corporation
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

#include "reduce_kernel_b_fs_yx_fsv16.h"
#include "kernel_selector_utils.h"
#include <vector>
#include <string>
#include "common_tools.h"

namespace kernel_selector {

static const size_t SIMD = 16;
using NDims = std::vector<kernel_selector::Tensor::Dim>;

static size_t calc_read_offset(const reduce_params& params) {
    auto read_offset = 1;
    if (BytesPerElement(params.inputs[0].GetDType()) == 4)
        read_offset = 4;
    else if (BytesPerElement(params.inputs[0].GetDType()) == 2)
        read_offset = 8;
    else if (BytesPerElement(params.inputs[0].GetDType()) == 1)
        read_offset = 16;
    return read_offset;
}

static NDims calc_in_dims(const reduce_params& params) {
    auto input = params.inputs[0];
    auto in_dims = input.GetDims();
    auto reduce_axes = params.reduceAxes;

    std::vector<size_t> ordered_axes = {0, 1, 3, 2};
    std::reverse(in_dims.begin(), in_dims.end());
    for (size_t a = 0; a < params.reduceAxes.size(); a++) {
        in_dims[ordered_axes[params.reduceAxes[a]]].v = 1;
    }

    return in_dims;
}

ParamsKey ReduceKernel_b_fs_yx_fsv16::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::INT32);
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT32);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableDifferentTypes();
    return k;
}

CommonDispatchData ReduceKernel_b_fs_yx_fsv16::SetDefault(const reduce_params& params, const optional_params&) const {
    CommonDispatchData dispatchData;

    auto in_dims = calc_in_dims(params);
    dispatchData.gws = { 16,
                         CeilDiv(in_dims[3].v, calc_read_offset(params)) * in_dims[2].v,  // X, Y
                         CeilDiv(in_dims[1].v, SIMD) * in_dims[0].v };                    // F, B
    dispatchData.lws = { SIMD, 1, 1 };

    return dispatchData;
}

JitConstants ReduceKernel_b_fs_yx_fsv16::GetJitConstants(const reduce_params& params) const {
    auto jit = ReduceKernelBase::GetJitConstants(params);
    auto in_dims = calc_in_dims(params);
    auto read_offset = calc_read_offset(params);

    // Universal output sizes for keep dims = true/false cases
    jit.AddConstant(MakeJitConstant("COMMON_OUTPUT_SIZE_X", in_dims[3].v));
    jit.AddConstant(MakeJitConstant("COMMON_OUTPUT_SIZE_Y", in_dims[2].v));
    jit.AddConstant(MakeJitConstant("COMMON_OUTPUT_FEATURE_NUM", in_dims[1].v));
    jit.AddConstant(MakeJitConstant("COMMON_OUTPUT_BATCH_NUM", in_dims[0].v));
    jit.AddConstant(MakeJitConstant("READ_OFFSET", read_offset));
    jit.AddConstant(MakeJitConstant("BLOCK_READ(ptr,offset)", "DT_INPUT_BLOCK_READ" + std::to_string(read_offset) + "(ptr,offset)"));
    jit.Merge(MakeTypeJitConstants(GetActivationType(params), "ACTIVATION"));
    jit.Merge(MakeTypeJitConstants(GetAccumulatorType(params), "ACCUMULATOR"));
    jit.Merge(MakeTypeJitConstants(GetFinalAccumulatorType(params), "FINAL_ACCUMULATOR"));

    if (!params.fused_ops.empty()) {
        auto input_dt = GetActivationType(params);
        std::vector<std::string> idx_order = {"b", "f", "y", "x"};
        std::string var_name = "reduce_result";

        bool cant_handle_vec16 = read_offset > 8 ? true : false;
        size_t vec_size = cant_handle_vec16 ? 8 : read_offset;

        FusedOpsConfiguration conf_scalar = {"_SCALAR",
                                             idx_order,
                                             var_name,
                                             input_dt,
                                             1,
                                             LoadType::LT_ALIGNED_READ,
                                             BoundaryCheck::DISABLED,
                                             IndexType::TENSOR_COORD,
                                             Tensor::DataChannelName::X};

        if (cant_handle_vec16) {
            FusedOpsConfiguration conf_vector_1 = {"_VECTOR_1",
                                                   idx_order,
                                                   var_name+".lo",
                                                   input_dt,
                                                   vec_size,
                                                   LoadType::LT_ALIGNED_READ,
                                                   BoundaryCheck::DISABLED,
                                                   IndexType::TENSOR_COORD,
                                                   Tensor::DataChannelName::X};

            std::vector<std::string> idx_order_vec_2 = {"b", "f", "y", "x + 8"};
            FusedOpsConfiguration conf_vector_2 = {"_VECTOR_2",
                                                   idx_order_vec_2,
                                                   var_name+".hi",
                                                   input_dt,
                                                   vec_size,
                                                   LoadType::LT_ALIGNED_READ,
                                                   BoundaryCheck::DISABLED,
                                                   IndexType::TENSOR_COORD,
                                                   Tensor::DataChannelName::X};

            jit.AddConstant(MakeJitConstant("FUSED_OPS_VECTOR", "{FUSED_OPS_VECTOR_1;final_result.lo=FUSED_OPS_RESULT_VECTOR_1;} {FUSED_OPS_VECTOR_2;final_result.hi=FUSED_OPS_RESULT_VECTOR_2;}"));
            jit.AddConstant(MakeJitConstant("FUSED_OPS_RESULT_VECTOR", "final_result"));
            jit.Merge(MakeFusedOpsJitConstants(params, {conf_scalar, conf_vector_1, conf_vector_2}));
        } else {
            FusedOpsConfiguration conf_vector = {"_VECTOR",
                                                 idx_order,
                                                 var_name,
                                                 input_dt,
                                                 vec_size,
                                                 LoadType::LT_ALIGNED_READ,
                                                 BoundaryCheck::DISABLED,
                                                 IndexType::TENSOR_COORD,
                                                 Tensor::DataChannelName::X};

            jit.Merge(MakeFusedOpsJitConstants(params, {conf_vector, conf_scalar}));
        }
    }

    return jit;
}

KernelsData ReduceKernel_b_fs_yx_fsv16::GetKernelsData(const Params& params, const optional_params& options) const {
    return GetCommonKernelsData(params, options);
}

KernelsPriority ReduceKernel_b_fs_yx_fsv16::GetKernelsPriority(const Params& /*params*/, const optional_params& /*options*/) const {
    return FORCE_PRIORITY_6;
}
}  // namespace kernel_selector
