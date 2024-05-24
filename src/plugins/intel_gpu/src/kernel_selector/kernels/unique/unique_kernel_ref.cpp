// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "unique_kernel_ref.hpp"

#include "kernel_selector_utils.h"

namespace kernel_selector {

namespace {

JitConstants MakeAxisJitConstants(size_t rank, int64_t axis, const std::string& prefix_for_iterate) {
    const std::map<char, std::string> dimensions_sizes_map = {
        {'b', "_BATCH_NUM"},
        {'f', "_FEATURE_NUM"},
        {'w', "_SIZE_W"},
        {'z', "_SIZE_Z"},
        {'y', "_SIZE_Y"},
        {'x', "_SIZE_X"},
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
    const auto axis_length_val = "INPUT0" + dimensions_sizes_map.at(axis_dimension);

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
    const auto iterate_val = [&dimensions, &dimensions_sizes_map, &prefix_for_iterate]() {
        std::stringstream ss;
        for (auto ch : dimensions) {
            // No need to iterate through axis index
            if (ch == 'i') {
                continue;
            }
            const auto size = prefix_for_iterate + dimensions_sizes_map.at(ch);
            ss << "for (uint " << ch << " = 0; " << ch << " < " << size << "; ++" << ch << ") {";
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
        return {MakeJitConstant("FLATTENED", true), MakeJitConstant(get_index_name, get_index_val)};
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

    return {MakeJitConstant("FLATTENED", true), MakeJitConstant(get_index_name, get_index_val)};
}

}  // namespace

void UniqueCountKernelRef::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [](const Params& params, KernelData& kd) {
        const auto& prim_params = dynamic_cast<const unique_count_params&>(params);
        auto dispatchData = SetDefault(prim_params);
        OPENVINO_ASSERT(kd.kernels.size() == 1, "[GPU] Invalid kernels size for update dispatch data func");
        kd.kernels[0].params.workGroups.global = dispatchData.gws;
        kd.kernels[0].params.workGroups.local = dispatchData.lws;
        kd.kernels[0].skip_execution = KernelData::SkipKernelExecution(prim_params);
        // Need to adjust buffer size according to input size
        kd.internalBufferSizes.front() = prim_params.inputs.front().PhysicalSizeInBytes();
        kd.internalBufferDataType = prim_params.inputs.front().GetDType();
    };
}

KernelsData UniqueCountKernelRef::GetKernelsData(const Params& params) const {
    if (!Validate(params)) {
        return {};
    }

    auto kernel_data = KernelData::Default<unique_count_params>(params);
    const auto& kernel_params = dynamic_cast<const unique_count_params&>(*kernel_data.params);
    const auto dispatch_data = SetDefault(kernel_params);
    const auto entry_point = GetEntryPoint(kernelName, kernel_params.layerID, params);
    const auto jit_constants = GetJitConstants(kernel_params);
    const auto jit = CreateJit(kernelName, jit_constants, entry_point);
    auto& kernel = kernel_data.kernels.front();

    GetUpdateDispatchDataFunc(kernel_data);

    FillCLKernelData(kernel,
                     dispatch_data,
                     params.engineInfo,
                     kernelName,
                     jit,
                     entry_point,
                     {},
                     false,
                     false,
                     static_cast<int>(kernel_params.inputs.size()),
                     GetFusedPrimitiveInputsCount(kernel_params),
                     static_cast<int>(kernel_params.outputs.size()),
                     kernel_params.is_shape_agnostic);

    // Additional buffer to save intermediate algorithm results
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
    kernel_data.internalBufferSizes.push_back(kernel_params.inputs.front().PhysicalSizeInBytes());
    kernel_data.internalBufferDataType = kernel_params.inputs.front().GetDType();

    return {kernel_data};
}

ParamsKey UniqueCountKernelRef::GetSupportedKey() const {
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

bool UniqueCountKernelRef::Validate(const Params& params) const {
    if (params.GetType() != KernelType::UNIQUE_COUNT) {
        return false;
    }

    const auto& kernel_params = dynamic_cast<const unique_count_params&>(params);
    if (kernel_params.inputs.size() != 1) {
        return false;
    }
    if (kernel_params.outputs.size() != 1) {
        return false;
    }

    return true;
}

JitConstants UniqueCountKernelRef::GetJitConstants(const unique_count_params& kernel_params) const {
    const auto input = kernel_params.inputs.front();
    auto jit_constants = MakeBaseParamsJitConstants(kernel_params);

    if (kernel_params.flattened) {
        jit_constants.Merge(MakeFlattenedJitConstants(input.Dimentions(), input.SimpleLayout()));
    } else {
        jit_constants.Merge(MakeAxisJitConstants(input.Dimentions(), kernel_params.axis, "INPUT0"));
    }

    if (input.is_dynamic()) {
        DimensionAccessHelperJit dims(input);
        const std::string total_data_size =
            toVectorMulString({dims.x(), dims.y(), dims.z(), dims.w(), dims.f(), dims.b()});
        jit_constants.AddConstant(MakeJitConstant("TOTAL_DATA_SIZE", total_data_size));
    } else {
        jit_constants.AddConstant(MakeJitConstant("TOTAL_DATA_SIZE", input.LogicalSize()));
    }

    return jit_constants;
}

CommonDispatchData UniqueCountKernelRef::SetDefault(const unique_count_params& /* kernel_params */) {
    CommonDispatchData dispatch_data;

    // For now we run only in one thread
    // TODO: Parallelize
    dispatch_data.gws = {1, 1, 1};
    dispatch_data.lws = {1, 1, 1};

    return dispatch_data;
}

void UniqueGatherKernelRef::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [](const Params& params, KernelData& kd) {
        const auto& prim_params = dynamic_cast<const unique_gather_params&>(params);
        auto dispatchData = SetDefault(prim_params);
        OPENVINO_ASSERT(kd.kernels.size() == 1, "[GPU] Invalid kernels size for update dispatch data func");
        kd.kernels[0].params.workGroups.global = dispatchData.gws;
        kd.kernels[0].params.workGroups.local = dispatchData.lws;
        kd.kernels[0].skip_execution = KernelData::SkipKernelExecution(prim_params);
    };
}

KernelsData UniqueGatherKernelRef::GetKernelsData(const Params& params) const {
    if (!Validate(params)) {
        return {};
    }

    auto kernel_data = KernelData::Default<unique_gather_params>(params);
    const auto& kernel_params = dynamic_cast<const unique_gather_params&>(*kernel_data.params);
    const auto dispatch_data = SetDefault(kernel_params);
    const auto entry_point = GetEntryPoint(kernelName, kernel_params.layerID, params);
    const auto jit_constants = GetJitConstants(kernel_params);
    const auto jit = CreateJit(kernelName, jit_constants, entry_point);
    auto& kernel = kernel_data.kernels.front();

    GetUpdateDispatchDataFunc(kernel_data);

    FillCLKernelData(kernel,
                     dispatch_data,
                     params.engineInfo,
                     kernelName,
                     jit,
                     entry_point,
                     {},
                     false,
                     false,
                     static_cast<int>(kernel_params.inputs.size()),
                     GetFusedPrimitiveInputsCount(kernel_params),
                     static_cast<int>(kernel_params.outputs.size()),
                     kernel_params.is_shape_agnostic);

    return {kernel_data};
}

ParamsKey UniqueGatherKernelRef::GetSupportedKey() const {
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

bool UniqueGatherKernelRef::Validate(const Params& params) const {
    if (params.GetType() != KernelType::UNIQUE_GATHER) {
        return false;
    }

    const auto& kernel_params = dynamic_cast<const unique_gather_params&>(params);
    if (kernel_params.inputs.size() != 2) {
        return false;
    }
    if (kernel_params.outputs.size() != 4) {
        return false;
    }

    return true;
}

JitConstants UniqueGatherKernelRef::GetJitConstants(const unique_gather_params& kernel_params) const {
    const auto input = kernel_params.inputs.front();
    auto jit_constants = MakeBaseParamsJitConstants(kernel_params);

    if (kernel_params.sorted) {
        jit_constants.AddConstant(MakeJitConstant("SORTED", true));
    }

    if (kernel_params.flattened) {
        jit_constants.Merge(MakeFlattenedJitConstants(input.Dimentions(), input.SimpleLayout()));
    } else {
        jit_constants.Merge(MakeAxisJitConstants(input.Dimentions(), kernel_params.axis, "OUTPUT"));
    }

    if (input.is_dynamic()) {
        DimensionAccessHelperJit dims(input);
        const std::string total_data_size =
            toVectorMulString({dims.x(), dims.y(), dims.z(), dims.w(), dims.f(), dims.b()});
        jit_constants.AddConstant(MakeJitConstant("TOTAL_DATA_SIZE", total_data_size));
    } else {
        jit_constants.AddConstant(MakeJitConstant("TOTAL_DATA_SIZE", input.LogicalSize()));
    }

    return jit_constants;
}

CommonDispatchData UniqueGatherKernelRef::SetDefault(const unique_gather_params& /* kernel_params */) {
    CommonDispatchData dispatch_data;

    // For now we run only in one thread
    // TODO: Parallelize
    dispatch_data.gws = {1, 1, 1};
    dispatch_data.lws = {1, 1, 1};

    return dispatch_data;
}

}  // namespace kernel_selector
