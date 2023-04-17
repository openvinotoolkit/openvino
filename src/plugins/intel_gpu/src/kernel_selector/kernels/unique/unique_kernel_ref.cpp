// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "unique_kernel_ref.hpp"

#include "kernel_selector_utils.h"

namespace kernel_selector {

namespace {

JitConstants MakeAxisJitConstants(size_t rank, int64_t axis, const std::string& input_num) {
    const std::map<char, std::string> dimensions_sizes_map = {
        {'b', "INPUT" + input_num + "_BATCH_NUM"},
        {'f', "INPUT" + input_num + "_FEATURE_NUM"},
        {'w', "INPUT" + input_num + "_SIZE_W"},
        {'z', "INPUT" + input_num + "_SIZE_Z"},
        {'y', "INPUT" + input_num + "_SIZE_Y"},
        {'x', "INPUT" + input_num + "_SIZE_X"},
    };

    auto dimensions = [rank]() -> std::vector<char> {
        switch (rank) {
        case 4:
            return {'b', 'f', 'y', 'x'};
        case 5:
            return {'b', 'f', 'z', 'y', 'x'};
        case 6:
            return {'b', 'f', 'w', 'z', 'y', 'x'};
        }
        throw std::invalid_argument("Unsupported input rank for unique primitive");
    }();
    auto& axis_dimension = dimensions.at(axis);

    const auto axis_length_name = "AXIS_LENGTH";
    const auto axis_length_val = dimensions_sizes_map.at(axis_dimension);

    // Mark axis dimension as 'i' for indexing
    axis_dimension = 'i';

    const auto get_index_name = "GET_INDEX(prefix, i)";
    const auto get_index_val = [&dimensions]() {
        std::string str = "CAT(prefix, _GET_INDEX)";
        str += '(';
        for (auto ch : dimensions) {
            str += ch;
            str += ',';
        }
        str.back() = ')';
        return str;
    }();

    const auto iterate_name = "ITERATE(body)";
    const auto iterate_val = [&dimensions, &dimensions_sizes_map]() {
        std::stringstream ss;
        for (auto ch : dimensions) {
            // No need to iterate through axis index
            if (ch == 'i') {
                continue;
            }
            ss << "for (uint " << ch << " = 0; " << ch << " < " << dimensions_sizes_map.at(ch) << "; ++" << ch << ") {";
        }
        ss << "body";
        // Note size - 1 here as we don't iterate through axis index
        for (auto i = 0U; i < dimensions.size() - 1; ++i) {
            ss << '}';
        }
        return ss.str();
    }();

    return {MakeJitConstant(axis_length_name, axis_length_val),
            MakeJitConstant(get_index_name, get_index_val),
            MakeJitConstant(iterate_name, iterate_val)};
}

JitConstants MakeFlattenedJitConstants(size_t rank, bool simple_layout) {
    const auto get_index_name = "GET_INDEX(prefix, i)";

    if (simple_layout) {
        const auto get_index_val = "i";
        return {MakeJitConstant(get_index_name, get_index_val)};
    }

    const auto dimensions = [rank]() -> std::vector<std::string> {
        switch (rank) {
        case 4:
            return {"i / (prefix##_SIZE_X * prefix##_SIZE_Y * prefix##_FEATURE_NUM)",
                    "i / (prefix##_SIZE_X * prefix##_SIZE_Y) % prefix##_FEATURE_NUM",
                    "i / prefix##_SIZE_X % prefix##_SIZE_Y",
                    "i % prefix##_SIZE_X"};
        case 5:
            return {"i / (prefix##_SIZE_X * prefix##_SIZE_Y * prefix##_SIZE_Z * prefix##_FEATURE_NUM)",
                    "i / (prefix##_SIZE_X * prefix##_SIZE_Y * prefix##_SIZE_Z) % prefix##_FEATURE_NUM",
                    "i / (prefix##_SIZE_X * prefix##_SIZE_Y) % prefix##_SIZE_Z",
                    "i / prefix##_SIZE_X % prefix##_SIZE_Y",
                    "i % prefix##_SIZE_X"};
        case 6:
            return {
                "i / (prefix##_SIZE_X * prefix##_SIZE_Y * prefix##_SIZE_Z * prefix##_SIZE_W * prefix##_FEATURE_NUM)",
                "i / (prefix##_SIZE_X * prefix##_SIZE_Y * prefix##_SIZE_Z * prefix##_SIZE_W) % prefix##_FEATURE_NUM",
                "i / (prefix##_SIZE_X * prefix##_SIZE_Y * prefix##_SIZE_Z) % prefix##_SIZE_W",
                "i / (prefix##_SIZE_X * prefix##_SIZE_Y) % prefix##_SIZE_Z",
                "i / prefix##_SIZE_X % prefix##_SIZE_Y",
                "i % prefix##_SIZE_X"};
        }
        throw std::invalid_argument("Unsupported rank for unique primitive");
    }();

    const auto get_index_val = [&dimensions]() {
        std::string str = "CAT(prefix, _GET_INDEX)";
        str += '(';
        for (const auto& dimension : dimensions) {
            str += dimension;
            str += ',';
        }
        str.back() = ')';
        return str;
    }();

    return {MakeJitConstant(get_index_name, get_index_val)};
}

}  // namespace

KernelsData UniqueKernelRef::GetKernelsData(const Params& params, const optional_params& options) const {
    if (!Validate(params, options)) {
        return {};
    }

    auto kernel_data = KernelData::Default<unique_params>(params);
    const auto& kernel_params = dynamic_cast<const unique_params&>(*kernel_data.params);
    const auto dispatch_data = SetDefault(kernel_params);
    const auto entry_point = GetEntryPoint(kernelName, kernel_params.layerID, params, options);
    const auto jit_constants = GetJitConstants(kernel_params);
    const auto jit = CreateJit(kernelName, jit_constants, entry_point);
    auto& kernel = kernel_data.kernels.front();

    kernel_data.update_dispatch_data_func = [this](const Params& params, KernelData& kd) {
        const auto& prim_params = dynamic_cast<const unique_params&>(params);
        auto dispatchData = SetDefault(prim_params);
        OPENVINO_ASSERT(kd.kernels.size() == 1, "[GPU] Invalid kernels size for update dispatch data func");
        kd.kernels[0].params.workGroups.global = dispatchData.gws;
        kd.kernels[0].params.workGroups.local = dispatchData.lws;
        kd.kernels[0].skip_execution = KernelData::SkipKernelExecution(prim_params);
    };

    FillCLKernelData(kernel,
                     dispatch_data,
                     params.engineInfo,
                     kernelName,
                     jit,
                     entry_point,
                     {},
                     false,
                     false,
                     kernel_params.inputs.size(),
                     GetFusedPrimitiveInputsCount(kernel_params),
                     kernel_params.outputs.size(),
                     kernel_params.inputs.front().is_dynamic());

    return {kernel_data};
}

ParamsKey UniqueKernelRef::GetSupportedKey() const {
    ParamsKey key;
    key.EnableAllInputDataType();
    key.EnableAllOutputDataType();
    key.EnableDifferentTypes();
    key.EnableAllInputLayout();
    key.EnableAllOutputLayout();
    key.EnableTensorOffset();
    key.EnableTensorPitches();
    key.EnableBatching();
    key.EnableDynamicShapesSupport();
    return key;
}

bool UniqueKernelRef::Validate(const Params& params, const optional_params& options) const {
    if (params.GetType() != KernelType::UNIQUE || options.GetType() != KernelType::UNIQUE) {
        return false;
    }

    const auto& kernel_params = dynamic_cast<const unique_params&>(params);
    if (kernel_params.inputs.size() != 1) {
        return false;
    }
    if (kernel_params.outputs.size() != 5) {
        return false;
    }

    return true;
}

JitConstants UniqueKernelRef::GetJitConstants(const unique_params& kernel_params) const {
    const auto input = kernel_params.inputs.front();
    auto jit_constants = MakeBaseParamsJitConstants(kernel_params);

    jit_constants.AddConstants({
        MakeJitConstant("FLATTENED", kernel_params.flattened),
        MakeJitConstant("SORTED", kernel_params.sorted),
    });

    if (kernel_params.flattened) {
        jit_constants.Merge(MakeFlattenedJitConstants(input.Dimentions(), input.SimpleLayout()));
    } else {
        jit_constants.Merge(MakeAxisJitConstants(input.Dimentions(), kernel_params.axis, "0"));
    }

    return jit_constants;
}

CommonDispatchData UniqueKernelRef::SetDefault(const unique_params& /* kernel_params */) {
    CommonDispatchData dispatch_data;

    // For now we run only in one thread
    // TODO: Parallelize
    dispatch_data.gws = {1, 1, 1};
    dispatch_data.lws = {1, 1, 1};

    return dispatch_data;
}

KernelsData UniqueReshapeKernelRef::GetKernelsData(const Params& params, const optional_params& options) const {
    if (!Validate(params, options)) {
        return {};
    }

    auto kernel_data = KernelData::Default<unique_reshape_params>(params);
    const auto& kernel_params = dynamic_cast<const unique_reshape_params&>(*kernel_data.params);
    const auto dispatch_data = SetDefault(kernel_params);
    const auto entry_point = GetEntryPoint(kernelName, kernel_params.layerID, params, options);
    const auto jit_constants = GetJitConstants(kernel_params);
    const auto jit = CreateJit(kernelName, jit_constants, entry_point);
    auto& kernel = kernel_data.kernels.front();

    kernel_data.update_dispatch_data_func = [this](const Params& params, KernelData& kd) {
        const auto& prim_params = dynamic_cast<const unique_reshape_params&>(params);
        auto dispatchData = SetDefault(prim_params);
        OPENVINO_ASSERT(kd.kernels.size() == 1, "[GPU] Invalid kernels size for update dispatch data func");
        kd.kernels[0].params.workGroups.global = dispatchData.gws;
        kd.kernels[0].params.workGroups.local = dispatchData.lws;
        kd.kernels[0].skip_execution = KernelData::SkipKernelExecution(prim_params);
    };

    FillCLKernelData(kernel,
                     dispatch_data,
                     params.engineInfo,
                     kernelName,
                     jit,
                     entry_point,
                     {},
                     false,
                     false,
                     kernel_params.inputs.size(),
                     GetFusedPrimitiveInputsCount(kernel_params),
                     kernel_params.outputs.size(),
                     kernel_params.outputs.front().is_dynamic());

    return {kernel_data};
}

ParamsKey UniqueReshapeKernelRef::GetSupportedKey() const {
    ParamsKey key;
    key.EnableAllInputDataType();
    key.EnableAllOutputDataType();
    key.EnableDifferentTypes();
    key.EnableAllInputLayout();
    key.EnableAllOutputLayout();
    key.EnableTensorOffset();
    key.EnableTensorPitches();
    key.EnableBatching();
    key.EnableDynamicShapesSupport();
    return key;
}

bool UniqueReshapeKernelRef::Validate(const Params& params, const optional_params& options) const {
    if (params.GetType() != KernelType::UNIQUE_RESHAPE || options.GetType() != KernelType::UNIQUE_RESHAPE) {
        return false;
    }

    const auto& kernel_params = dynamic_cast<const unique_reshape_params&>(params);
    if (kernel_params.inputs.size() != 5) {
        return false;
    }
    if (kernel_params.outputs.size() != 4) {
        return false;
    }

    return true;
}

JitConstants UniqueReshapeKernelRef::GetJitConstants(const unique_reshape_params& kernel_params) const {
    const auto input = kernel_params.inputs.at(1);
    auto jit_constants = MakeBaseParamsJitConstants(kernel_params);

    jit_constants.AddConstants({
        MakeJitConstant("FLATTENED", kernel_params.flattened),
        MakeJitConstant("AXIS", kernel_params.axis),
    });

    if (kernel_params.flattened) {
        jit_constants.Merge(MakeFlattenedJitConstants(input.Dimentions(), input.SimpleLayout()));
    } else {
        jit_constants.Merge(MakeAxisJitConstants(input.Dimentions(), kernel_params.axis, "1"));
    }

    return jit_constants;
}

CommonDispatchData UniqueReshapeKernelRef::SetDefault(const unique_reshape_params& /* kernel_params */) {
    CommonDispatchData dispatch_data;

    // For now we run only in one thread
    // TODO: Parallelize
    dispatch_data.gws = {1, 1, 1};
    dispatch_data.lws = {1, 1, 1};

    return dispatch_data;
}

}  // namespace kernel_selector
