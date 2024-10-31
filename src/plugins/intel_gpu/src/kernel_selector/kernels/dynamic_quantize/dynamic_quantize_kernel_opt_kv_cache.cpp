// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dynamic_quantize_kernel_kv_cache.h"
#include "kernel_selector_utils.h"
#include <string>


static constexpr size_t subgroup_size = 16;

namespace kernel_selector {
static Tensor::NDims get_normalized_dims(const DataTensor& tensor) {
    auto dims = tensor.GetDims();
    std::reverse(dims.begin(), dims.end());

    return dims;
}

static size_t get_elements_number_per_batch(const dynamic_quantize_params& params) {
    const auto& group_sizes = params.group_sizes;
    const auto& input_dims = get_normalized_dims(params.inputs[0]);

    size_t total_elements_number = 1;
    for (size_t i = 0; i < group_sizes.size(); i++) {
        if (group_sizes[i] != UINT64_MAX) {
            total_elements_number *= input_dims[i].v;
        }
    }

    return total_elements_number;
}

static size_t get_elements_number_per_group(const dynamic_quantize_params& params) {
    const auto& group_sizes = params.group_sizes;
    const auto& input_dims = get_normalized_dims(params.inputs[0]);

    size_t total_elements_number = 1;
    for (size_t i = 0; i < group_sizes.size(); i++) {
        if (group_sizes[i] == UINT64_MAX) {
            total_elements_number *= input_dims[i].v;
        } else {
            total_elements_number *= group_sizes[i];
        }
    }

    return total_elements_number;
}

static std::string generate_dims_indexes_calculation(std::vector<std::pair<std::string, std::string>> dims) {
    std::reverse(dims.begin(), dims.end()); // reorder dims in order from innermost to outermost dimensions

    auto generate_calc_function = [&](std::string data_type, std::string index_var, size_t dim_idx) {
        std::string index_calc_str;
        index_calc_str += "" + data_type + " " + dims[dim_idx].first + " = ";
        index_calc_str += "(" + index_var + " / ";
        index_calc_str += "(1";
        for (size_t i = 0; i < dim_idx; i++) {
            index_calc_str += " * " + dims[i].second;
        }
        index_calc_str += ")) % " + dims[dim_idx].second + ";";

        return index_calc_str;
    };

    std::stringstream indexes_calc_str;
    for (size_t i = 0; i < dims.size(); i++) {
        indexes_calc_str << generate_calc_function("uint", "data_idx", i);
    }

    return indexes_calc_str.str();
}

static size_t get_per_iter_elements_number(const dynamic_quantize_params& params) {
    const auto maxWorkGroupSize = params.engineInfo.maxWorkGroupSize;
    const auto total_grouped_elements = get_elements_number_per_group(params);

    if (total_grouped_elements % maxWorkGroupSize == 0)
        return maxWorkGroupSize;

    if (total_grouped_elements < maxWorkGroupSize)
        return total_grouped_elements;

    return 0;
}

ParamsKey DynamicQuantizeKernelKVCache::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableDifferentTypes();
    k.EnableAllInputLayout();
    k.EnableAllOutputLayout();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableDynamicShapesSupport();
    return k;
}

JitConstants DynamicQuantizeKernelKVCache::GetJitConstants(const dynamic_quantize_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    const std::vector<std::pair<std::string, std::string>> default_dims = {{"b", "INPUT0_BATCH_NUM"},
                                                                           {"f", "INPUT0_FEATURE_NUM"},
                                                                           {"y", "INPUT0_SIZE_Y"},
                                                                           {"x", "INPUT0_SIZE_X"}};

    const auto& group_sizes = params.group_sizes;
    std::vector<std::pair<std::string, std::string>> batch_dims, grouped_dims;
    for (size_t i = 0; i < group_sizes.size(); i++) {
        if (group_sizes[i] == 1) {
            batch_dims.push_back(default_dims[i]);
        } else {
            grouped_dims.push_back(default_dims[i]);
        }
    }

    const auto& input_dims = get_normalized_dims(params.inputs[0]);
    const auto total_grouped_elements = get_elements_number_per_group(params);
    const auto per_iter_elements_number = get_per_iter_elements_number(params);
    const auto total_subgroups_number = total_grouped_elements / input_dims.back().v;

    // Drop the last dimensions, since it will be processed in the kernel's loop
    grouped_dims.pop_back();

    const bool append_mode = params.append_axis != -1;
    std::pair<std::string, std::string> append_axis_info = {};
    if (append_mode) {
        jit.AddConstant(MakeJitConstant("APPEND_MODE", append_mode));
        jit.AddConstant(MakeJitConstant("APPEND_AXIS_NAME", default_dims[params.append_axis].first));
    }

    jit.AddConstant(MakeJitConstant("DECLARE_BATCHED_DIMS_INDEXES(data_idx)", generate_dims_indexes_calculation(batch_dims)));
    jit.AddConstant(MakeJitConstant("DECLARE_GROUPED_DIMS_INDEXES(data_idx)", generate_dims_indexes_calculation(grouped_dims)));
    jit.AddConstant(MakeJitConstant("SUBGROUPS_NUMBER", total_subgroups_number));

    const auto iterations_number = total_grouped_elements / per_iter_elements_number;

    jit.AddConstant(MakeJitConstant("ITERATIONS_NUMBER", iterations_number));
    jit.AddConstant(MakeJitConstant("ASYMMETRIC_QUANTIZATION", params.use_asymmetric_quantization));
    jit.AddConstant(MakeJitConstant("GROUP_SCALES_WITH_ZP", params.combine_scales_and_zp));

    bool rearrange_scales_order = false;
    const auto& scales_output_order = params.scales_output_order;
    if (!scales_output_order.empty()) {
        for (size_t i = 0; i < scales_output_order.size(); i++) {
            if (i != scales_output_order[i]) {
                rearrange_scales_order = true;
                break;
            }
        }
    }

    if (rearrange_scales_order) {
        const std::array<char, 4> default_dim_order = {'b', 'f', 'y', 'x'};
        std::stringstream ss;
        for (size_t i = 0; i < scales_output_order.size(); i++) {
            ss << default_dim_order[scales_output_order[i]];

            if (i + 1 != scales_output_order.size())
                ss << ", ";
        }

        jit.AddConstant(MakeJitConstant("SCALES_OUTPUT_ORDER", ss.str()));
    }

    for (size_t i = 0; i < group_sizes.size(); i++) {
        jit.AddConstant(MakeJitConstant("GROUP_SIZE_DIM" + std::to_string(i), group_sizes[i]));
    }

    return jit;
}

CommonDispatchData DynamicQuantizeKernelKVCache::SetDefault(const dynamic_quantize_params& params) const {
    CommonDispatchData dispatchData;

    const auto& input_dims = get_normalized_dims(params.inputs[0]);
    const auto total_batched_elements = get_elements_number_per_batch(params);
    const auto total_grouped_elements = get_elements_number_per_group(params);
    const auto total_subgroups_number = total_grouped_elements / input_dims.back().v;

    dispatchData.gws = {subgroup_size, total_subgroups_number, total_batched_elements};
    dispatchData.lws = {subgroup_size, total_subgroups_number, 1};

    return dispatchData;
}

void DynamicQuantizeKernelKVCache::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [this](const Params& params, KernelData& kd) {
        const auto& prim_params = static_cast<const dynamic_quantize_params&>(params);
        auto dispatchData = SetDefault(prim_params);
        OPENVINO_ASSERT(kd.kernels.size() == 1, "[GPU] Invalid kernels size for update dispatch data func");
        kd.kernels[0].params.workGroups.global = dispatchData.gws;
        kd.kernels[0].params.workGroups.local = dispatchData.lws;
        kd.kernels[0].skip_execution = false;

        if (prim_params.append_axis != -1) {
            kd.kernels[0].params.scalars.clear();

            ScalarDescriptor axis_offset;
            axis_offset.t = ScalarDescriptor::Types::UINT32;
            axis_offset.v.u32 = static_cast<uint32_t>(prim_params.axis_offset);
            kd.kernels[0].params.scalars.push_back(axis_offset);
        }
    };
}

KernelsData DynamicQuantizeKernelKVCache::GetKernelsData(const Params& params) const {
    assert(params.GetType() == KernelType::DYNAMIC_QUANTIZE);

    if (!Validate(params))
        return {};

    const dynamic_quantize_params& prim_params = static_cast<const dynamic_quantize_params&>(params);
    auto dispatchData = SetDefault(prim_params);

    KernelData kd = KernelData::Default<dynamic_quantize_params>(params);

    auto cldnn_jit = GetJitConstants(prim_params);
    auto entry_point = GetEntryPoint(kernelName, prim_params.layerID, params);
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
                     static_cast<int>(prim_params.outputs.size()),
                     prim_params.is_shape_agnostic);

    if (prim_params.append_axis != -1)
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::SCALAR, 0});

    return {kd};
}

KernelsPriority DynamicQuantizeKernelKVCache::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_3;
}

bool DynamicQuantizeKernelKVCache::Validate(const Params& params) const {
    if (!KernelBaseOpenCL::Validate(params))
        return false;

    const auto& dq_params = static_cast<const dynamic_quantize_params&>(params);

    const auto& group_sizes = dq_params.group_sizes;
    const auto& input_dims = get_normalized_dims(dq_params.inputs[0]);
    const size_t non_compressed_dims_number = std::count(group_sizes.begin(), group_sizes.end(), 1);

    if (non_compressed_dims_number == group_sizes.size())
        return false;

    for (size_t i = 0; i < group_sizes.size(); i++) {
        if (group_sizes[i] != 1 && input_dims[i].is_dynamic) {
            return false;
        }
    }

    // Last dimension should be static, reduced by group_sizes configuration and divisible by 16
    if (group_sizes.back() == 1 || input_dims.back().is_dynamic || input_dims.back().v % subgroup_size != 0)
        return false;

    // Limit the size of the innermost dimension
    if (input_dims.back().v > 256)
        return false;

    // In case of HEADS_NUM * HEAD_SIZE group size, check that it fits into the supported workgroup size limit
    if (get_elements_number_per_group(dq_params) / input_dims.back().v >= params.engineInfo.maxWorkGroupSize / subgroup_size)
        return false;

    return true;
}
}  // namespace kernel_selector

