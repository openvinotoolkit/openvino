// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gemm_kernel_base.h"
#include <vector>
#include "kernel_selector_utils.h"

namespace kernel_selector {

size_t GemmKernelBase::GetOuputSize(const std::vector<int64_t>& output_order, const kernel_selector::DataTensor &output,
                                    char target_dim) const {
    OPENVINO_ASSERT(target_dim == 'X' || target_dim == 'Y');

    auto output_dims_order = GetDimsOrder(output_order);
    int dim_idx = (target_dim == 'X') ? 10 : 8;
    switch (output_dims_order[dim_idx]) {
        case 'b':
            return output.Batch().v;
        case 'f':
            return output.Feature().v;
        case 'w':
            return output.W().v;
        case 'z':
            return output.Z().v;
        case 'y':
            return output.Y().v;
        case 'x':
            return output.X().v;
        default:
            OPENVINO_THROW("Unsupported dimension: ", output_dims_order[dim_idx]);
    }
}

std::vector<int64_t> GemmKernelBase::ConvTo8dims(const std::vector<int64_t>& order_idx) const {
    std::vector<int64_t> dims_order;
    if (order_idx.size() == 2) {
        dims_order = {0, 1, 2, 3, 4, 5};
        dims_order.push_back(order_idx[0] + 6);
        dims_order.push_back(order_idx[1] + 6);
    } else if (order_idx.size() == 3) {
        dims_order.push_back(order_idx[0] == 0 ? 0 : order_idx[0] + 5);
        dims_order.push_back(1);
        dims_order.push_back(2);
        dims_order.push_back(3);
        dims_order.push_back(4);
        dims_order.push_back(5);
        dims_order.push_back(order_idx[1] == 0 ? 0 : order_idx[1] + 5);
        dims_order.push_back(order_idx[2] == 0 ? 0 : order_idx[2] + 5);
    } else if (order_idx.size() == 4) {
        dims_order.push_back(order_idx[0] < 2 ? order_idx[0] : order_idx[0] + 4);
        dims_order.push_back(order_idx[1] < 2 ? order_idx[1] : order_idx[1] + 4);
        dims_order.push_back(2);
        dims_order.push_back(3);
        dims_order.push_back(4);
        dims_order.push_back(5);
        dims_order.push_back(order_idx[2] < 2 ? order_idx[2] : order_idx[2] + 4);
        dims_order.push_back(order_idx[3] < 2 ? order_idx[3] : order_idx[3] + 4);
    } else if (order_idx.size() == 5) {
        dims_order.push_back(order_idx[0] < 2 ? order_idx[0] : order_idx[0] + 3);
        dims_order.push_back(order_idx[1] < 2 ? order_idx[1] : order_idx[1] + 3);
        dims_order.push_back(2);
        dims_order.push_back(3);
        dims_order.push_back(4);
        dims_order.push_back(order_idx[2] < 2 ? order_idx[2] : order_idx[2] + 3);
        dims_order.push_back(order_idx[3] < 2 ? order_idx[3] : order_idx[3] + 3);
        dims_order.push_back(order_idx[4] < 2 ? order_idx[4] : order_idx[4] + 3);
    } else if (order_idx.size() == 6) {
        dims_order.push_back(order_idx[0] < 2 ? order_idx[0] : order_idx[0] + 2);
        dims_order.push_back(order_idx[1] < 2 ? order_idx[1] : order_idx[1] + 2);
        dims_order.push_back(2);
        dims_order.push_back(3);
        dims_order.push_back(order_idx[2] < 2 ? order_idx[2] : order_idx[2] + 2);
        dims_order.push_back(order_idx[3] < 2 ? order_idx[3] : order_idx[3] + 2);
        dims_order.push_back(order_idx[4] < 2 ? order_idx[4] : order_idx[4] + 2);
        dims_order.push_back(order_idx[5] < 2 ? order_idx[5] : order_idx[5] + 2);
    } else if (order_idx.size() == 7) {
        dims_order.push_back(order_idx[0] < 2 ? order_idx[0] : order_idx[0] + 1);
        dims_order.push_back(order_idx[1] < 2 ? order_idx[1] : order_idx[1] + 1);
        dims_order.push_back(2);
        dims_order.push_back(order_idx[2] < 2 ? order_idx[2] : order_idx[2] + 1);
        dims_order.push_back(order_idx[3] < 2 ? order_idx[3] : order_idx[3] + 1);
        dims_order.push_back(order_idx[4] < 2 ? order_idx[4] : order_idx[4] + 1);
        dims_order.push_back(order_idx[5] < 2 ? order_idx[5] : order_idx[5] + 1);
        dims_order.push_back(order_idx[6] < 2 ? order_idx[6] : order_idx[6] + 1);
    } else {
        dims_order = {0, 1, 2, 3, 4, 5, 6, 7};
    }
    return dims_order;
}

std::vector<std::string> GemmKernelBase::GetTransposedDims(const std::vector<int64_t>& order_idx, bool is_tiled_opt) const {
    auto converted_dims = ConvTo8dims(order_idx);
    std::vector<std::string> dim_ids;
    for (auto dim : converted_dims) {
        switch (dim) {
        case 0:
            dim_ids.push_back("b");
            break;
        case 1:
            dim_ids.push_back("f");
            break;
        case 2:
            dim_ids.push_back("u");
            break;
        case 3:
            dim_ids.push_back("v");
            break;
        case 4:
            dim_ids.push_back("w");
            break;
        case 5:
            dim_ids.push_back("z");
            break;
        case 6:
            if (is_tiled_opt) {
                dim_ids.push_back("(y+write_id)");
            } else {
                dim_ids.push_back("y");
            }
            break;
        case 7:
            dim_ids.push_back("x");
            break;
        default:
            break;
        }
    }
    return dim_ids;
}

std::string GemmKernelBase::GetDimsOrder(const std::vector<int64_t>& order_idx) const {
    auto get_order_idx = [](std::vector<int64_t> order_idx, int64_t dim_idx) {
        int loc = 0;
        for (auto idx : order_idx) {
            if (idx == dim_idx)
                break;
            loc += 1;
        }
        return loc;
    };

    std::string dims_order = "";
    if (order_idx.size() == 2) {
        const std::vector<std::string> dims2 = {"y", "x"};
        dims_order = "b,f,w,z,"
                    + dims2[get_order_idx(order_idx, 0)] + "," + dims2[get_order_idx(order_idx, 1)];
    } else if (order_idx.size() == 3) {
        const std::vector<std::string> dims3 = {"b", "y", "x"};
        dims_order = dims3[get_order_idx(order_idx, 0)] + ",f,w,z,"
                    + dims3[get_order_idx(order_idx, 1)] + "," + dims3[get_order_idx(order_idx, 2)];
    } else if (order_idx.size() == 4) {
        const std::vector<std::string> dims4 = {"b", "f", "y", "x"};
        dims_order = dims4[get_order_idx(order_idx, 0)] + "," + dims4[get_order_idx(order_idx, 1)] + ",w,z,"
                    + dims4[get_order_idx(order_idx, 2)] + "," + dims4[get_order_idx(order_idx, 3)];
    } else if (order_idx.size() == 5) {
        const std::vector<std::string> dims5 = {"b", "f", "z", "y", "x"};
        dims_order = dims5[get_order_idx(order_idx, 0)] + "," + dims5[get_order_idx(order_idx, 1)] + ",w,"
                    + dims5[get_order_idx(order_idx, 2)] + "," + dims5[get_order_idx(order_idx, 3)] + ","
                    + dims5[get_order_idx(order_idx, 4)];
    } else if (order_idx.size() == 6) {
        const std::vector<std::string> dims6 = {"b", "f", "w", "z", "y", "x"};
        dims_order = dims6[get_order_idx(order_idx, 0)] + "," + dims6[get_order_idx(order_idx, 1)] + ","
                    + dims6[get_order_idx(order_idx, 2)] + "," + dims6[get_order_idx(order_idx, 3)] + ","
                    + dims6[get_order_idx(order_idx, 4)] + "," + dims6[get_order_idx(order_idx, 5)];
    } else {
        dims_order = "b,f,w,z,y,x";
    }
    return dims_order;
}

JitConstants GemmKernelBase::GetJitConstants(const gemm_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    auto get_transpose_mode = [](const std::vector<int64_t>& order_idx) {
        int64_t rank = order_idx.size() - 1;

        if (rank == order_idx[rank]) {
            // normal
            return 0;
        } else if (rank == order_idx[rank - 1]) {
            // the second last dim is moved to the last
            return 1;
        } else {
            // randomly transposed
            return 2;
        }
    };

    jit.AddConstants({
        MakeJitConstant("ALPHA", params.alpha),
        MakeJitConstant("BETA", params.beta),
        MakeJitConstant("TRANSPOSE_INPUT0", get_transpose_mode(params.input0_order)),
        MakeJitConstant("TRANSPOSE_INPUT1", get_transpose_mode(params.input1_order)),
        MakeJitConstant("QUANTIZATION_TERM", params.quantization != QuantizationType::NONE),
    });

    jit.AddConstants({
        MakeJitConstant("INPUT0_DIMS_ORDER", GetDimsOrder(params.input0_order)),
        MakeJitConstant("INPUT1_DIMS_ORDER", GetDimsOrder(params.input1_order)),
        MakeJitConstant("MATMUL_AXIS", static_cast<char>(std::toupper(GetDimsOrder(params.input0_order).at(10)))),
    });

    return jit;
}

GemmKernelBase::DispatchData GemmKernelBase::SetDefault(const gemm_params& params) const {
    const auto& output = params.outputs[0];

    DispatchData dispatchData;

    if (!output.is_dynamic()) {
        auto total_batches = output.LogicalSize() / (output.X().v * output.Y().v);
        dispatchData.gws = { output.X().v, output.Y().v, total_batches };
        dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);
    }

    return dispatchData;
}

void GemmKernelBase::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [this](const Params& params, KernelData& kd) {
        const auto& prim_params = static_cast<const gemm_params&>(params);
            auto dispatchData = SetDefault(prim_params);
            OPENVINO_ASSERT(kd.kernels.size() == 1, "[GPU] Invalid kernels size for update dispatch data func");
            kd.kernels[0].params.workGroups.global = dispatchData.gws;
            kd.kernels[0].params.workGroups.local = dispatchData.lws;
            kd.kernels[0].skip_execution = KernelData::SkipKernelExecution(prim_params);
    };
}

KernelsData GemmKernelBase::GetCommonKernelsData(const Params& params,
                                                 const optional_params& options) const {
    if (!Validate(params, options)) {
        return KernelsData();
    }

    const auto& prim_params = static_cast<const gemm_params&>(params);

    auto dispatchData = SetDefault(prim_params);
    KernelData k_data = KernelData::Default<gemm_params>(params);
    GetUpdateDispatchDataFunc(k_data);
    auto cldnn_jit = GetJitConstants(prim_params);
    auto entry_point = GetEntryPoint(kernelName, prim_params.layerID, params, options);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = k_data.kernels[0];
    FillCLKernelData(kernel,
                     dispatchData,
                     params.engineInfo,
                     kernelName,
                     jit,
                     entry_point,
                     EXE_MODE_DEFAULT,
                     false,
                     false,
                     (uint32_t)prim_params.inputs.size(),
                     GetFusedPrimitiveInputsCount(params),
                     1,
                     prim_params.has_dynamic_tensors());

    return {k_data};
}

JitConstants GemmKernelBase::GetFusedPrimitivesJitConstants(const gemm_params&, const DispatchData&) const {
    return {};
}

bool GemmKernelBase::Validate(const Params& p, const optional_params&) const {
    const gemm_params& params = static_cast<const gemm_params&>(p);

    if (params.GetType() != KernelType::GEMM) {
        return false;
    }

    for (auto& fused_op : params.fused_ops) {
        if (!IsFusedPrimitiveSupported(fused_op))
            return false;
    }

    return true;
}

Datatype GemmKernelBase::GetActivationType(const gemm_params& params) const {
    if (params.quantization != QuantizationType::NONE)
        return Datatype::F32;

    return GetUnitType(params);
}

}  // namespace kernel_selector
