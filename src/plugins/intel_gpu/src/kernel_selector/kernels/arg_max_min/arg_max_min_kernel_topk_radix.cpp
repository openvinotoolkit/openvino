// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "arg_max_min_kernel_topk_radix.h"
#include <kernel_selector_utils.h>
#include "common_tools.h"
#include <cstdlib>

namespace kernel_selector {

namespace {

// The kernel gets these as JIT constants, and Validate sizes its SLM from them.
constexpr size_t kWgSize = 256;
constexpr size_t kSubGroupSize = 16;
constexpr size_t kNumBuckets = 256;
constexpr size_t kBucketGroupSize = 16;
constexpr size_t kTieGroupSize = 16;
static_assert(kNumBuckets == 256, "The kernel resolves 8-bit digits: NUM_PASSES and its 0xFF masks assume 256 buckets");
static_assert(kNumBuckets % kBucketGroupSize == 0 && kWgSize % kTieGroupSize == 0,
              "Bucket and tie groups must evenly divide the buckets and the work-group");

constexpr size_t kTargetChunkSize = 8192;
constexpr size_t kMinChunkedSortSize = 2 * kTargetChunkSize;
constexpr size_t kMaxCandidatesShare = 8;
constexpr size_t kMergeBalance = 2;
constexpr size_t kMinChunkSize = 1024;
constexpr size_t kMinFewRowsSortSize = 4096;
constexpr size_t kFewRowsFillDivisor = 2;

struct ChunkConfig {
    size_t chunks = 1;
    size_t chunk_size = 0;
};

size_t GetOperationNumber(const arg_max_min_params& params) {
    switch (params.argMaxMinAxis) {
        case ArgMaxMinAxis::BATCH: return params.outputs[0].Feature().v * params.outputs[0].Z().v * params.outputs[0].Y().v * params.outputs[0].X().v;
        case ArgMaxMinAxis::FEATURE: return params.outputs[0].Batch().v * params.outputs[0].Z().v * params.outputs[0].Y().v * params.outputs[0].X().v;
        case ArgMaxMinAxis::Z: return params.outputs[0].Batch().v * params.outputs[0].Feature().v * params.outputs[0].Y().v * params.outputs[0].X().v;
        case ArgMaxMinAxis::Y: return params.outputs[0].Batch().v * params.outputs[0].Feature().v * params.outputs[0].Z().v * params.outputs[0].X().v;
        case ArgMaxMinAxis::X: return params.outputs[0].Batch().v * params.outputs[0].Feature().v * params.outputs[0].Z().v * params.outputs[0].Y().v;
        default: throw std::invalid_argument("Unsupported axis");
    }
}

std::string GetOperationNumberString(const arg_max_min_params& params) {
    const auto& output = params.outputs[0];
    DimensionAccessHelperJit dims(output);
    switch (params.argMaxMinAxis) {
        case ArgMaxMinAxis::BATCH: return toVectorMulString({dims.x(), dims.y(), dims.z(), dims.f()});
        case ArgMaxMinAxis::FEATURE: return toVectorMulString({dims.x(), dims.y(), dims.z(), dims.b()});
        case ArgMaxMinAxis::Z: return toVectorMulString({dims.y(), dims.z(), dims.f(), dims.b()});
        case ArgMaxMinAxis::Y: return toVectorMulString({dims.x(), dims.z(), dims.f(), dims.b()});
        case ArgMaxMinAxis::X: return toVectorMulString({dims.y(), dims.z(), dims.f(), dims.b()});
        default: throw std::invalid_argument("Unsupported axis");
    }
}

size_t GetSortSize(const arg_max_min_params& params) {
    switch (params.argMaxMinAxis) {
        case ArgMaxMinAxis::BATCH: return params.inputs[0].Batch().v;
        case ArgMaxMinAxis::FEATURE: return params.inputs[0].Feature().v;
        case ArgMaxMinAxis::Z: return params.inputs[0].Z().v;
        case ArgMaxMinAxis::Y: return params.inputs[0].Y().v;
        case ArgMaxMinAxis::X: return params.inputs[0].X().v;
        default: throw std::invalid_argument("Unsupported axis");
    }
}

bool IsSortSizeDynamic(const arg_max_min_params& params) {
    switch (params.argMaxMinAxis) {
        case ArgMaxMinAxis::BATCH: return params.inputs[0].Batch().is_dynamic;
        case ArgMaxMinAxis::FEATURE: return params.inputs[0].Feature().is_dynamic;
        case ArgMaxMinAxis::Z: return params.inputs[0].Z().is_dynamic;
        case ArgMaxMinAxis::Y: return params.inputs[0].Y().is_dynamic;
        case ArgMaxMinAxis::X: return params.inputs[0].X().is_dynamic;
        default: throw std::invalid_argument("Unsupported axis");
    }
}

// Work-groups of kWgSize work-items that the device keeps resident at once.
size_t GetResidentWorkGroups(const arg_max_min_params& params) {
    return std::max<size_t>(1, params.engineInfo.maxThreadsPerDevice / (kWgSize / kSubGroupSize));
}

// Splits long rows so that a few operations still occupy many work-groups. By default the chunk
// count only depends on the static sort size, so dynamic operation counts reuse the same kernels.
// When the operation count is known at compile time and fills less than half of the resident
// work-groups, rows of kMinFewRowsSortSize or more values are split further, into chunks of at
// least kMinChunkSize values. Dynamic operation counts keep the default, because short chunks
// are slower than unchunked rows once many rows fill the device.
ChunkConfig GetChunkConfig(const arg_max_min_params& params, bool static_operation_num) {
    ChunkConfig config;
    if (IsSortSizeDynamic(params)) {
        return config;
    }

    const size_t sort_size = GetSortSize(params);
    const size_t top_k = params.topK;
    size_t chunks = sort_size >= kMinChunkedSortSize ? CeilDiv(sort_size, kTargetChunkSize) : 1;
    if (static_operation_num && sort_size >= kMinFewRowsSortSize) {
        const size_t ops_num = GetOperationNumber(params);
        const size_t target_work_groups = GetResidentWorkGroups(params) / kFewRowsFillDivisor;
        if (ops_num * chunks < target_work_groups) {
            chunks = std::max(chunks, std::min(CeilDiv(target_work_groups, ops_num), sort_size / kMinChunkSize));
        }
    }

    for (; chunks > 1; --chunks) {
        const size_t chunk_size = CeilDiv(sort_size, chunks);
        if (chunk_size * (chunks - 1) >= sort_size || sort_size - chunk_size * (chunks - 1) < top_k) {
            continue;
        }
        // The merge stage selects again from chunks * top_k candidates, so large K needs long chunks:
        // every chunk holds kMaxCandidatesShare * top_k values, and the merge input stays within
        // 1 / kMergeBalance of a chunk.
        if (chunks * top_k * kMaxCandidatesShare > sort_size || chunks * chunks * top_k * kMergeBalance > sort_size) {
            continue;
        }
        config.chunks = chunks;
        config.chunk_size = chunk_size;
        break;
    }
    return config;
}

void SetInternalBuffers(const arg_max_min_params& params, const ChunkConfig& config, KernelData& kd) {
    const size_t ops_num = GetOperationNumber(params);
    kd.internalBuffers.clear();
    if (config.chunks == 1) {
        // Sortable keys: VALUES_NUM per operation
        kd.internalBuffers.push_back(sizeof(uint32_t) * GetSortSize(params) * ops_num);
    } else {
        // Sortable keys per chunk, then top-K keys and indices of every chunk
        const size_t candidates_size = sizeof(uint32_t) * config.chunks * params.topK * ops_num;
        kd.internalBuffers.push_back(sizeof(uint32_t) * config.chunks * config.chunk_size * ops_num);
        kd.internalBuffers.push_back(candidates_size);
        kd.internalBuffers.push_back(candidates_size);
    }
    kd.internalBufferDataType = Datatype::UINT32;
}

}  // namespace

ParamsKey ArgMaxMinKernelTopKRadix::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableAllOutputDataType();
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::byxf);
    k.EnableOutputLayout(DataLayout::byxf);
    k.EnableInputLayout(DataLayout::bfzyx);
    k.EnableOutputLayout(DataLayout::bfzyx);
    k.EnableArgMaxMinAxis(ArgMaxMinAxis::BATCH);
    k.EnableArgMaxMinAxis(ArgMaxMinAxis::X);
    k.EnableArgMaxMinAxis(ArgMaxMinAxis::Y);
    k.EnableArgMaxMinAxis(ArgMaxMinAxis::Z);
    k.EnableArgMaxMinAxis(ArgMaxMinAxis::FEATURE);
    k.EnableDifferentTypes();
    k.EnableBatching();
    k.EnableTensorPitches();
    k.EnableTensorOffset();
    k.EnableDynamicShapesSupport();
    return k;
}

bool ArgMaxMinKernelTopKRadix::Validate(const Params& p) const {
    if (!ArgMaxMinKernelBase::Validate(p)) {
        DO_NOT_USE_THIS_KERNEL(p.layerID);
    }

    const auto& params = static_cast<const arg_max_min_params&>(p);

    // Radix approach relies on bit manipulation of the IEEE-754 representation
    if (params.inputs[0].GetDType() != Datatype::F16 && params.inputs[0].GetDType() != Datatype::F32) {
        DO_NOT_USE_THIS_KERNEL(p.layerID);
    }

    const size_t sort_size = GetSortSize(params);

    if (sort_size < 2) {
        DO_NOT_USE_THIS_KERNEL(p.layerID);
    }

    // PADDED_K (next power of 2 >= topK) must fit in SLM:
    // sort_keys[PADDED_K] + sort_idxs[PADDED_K] + histogram[NUM_BUCKETS] + bucket/tie counters + scalars
    size_t padded_k = 1;
    while (padded_k < params.topK) {
        padded_k <<= 1;
    }
    const size_t counters = kNumBuckets / kBucketGroupSize + kWgSize + kWgSize / kTieGroupSize;
    const size_t slm_needed = (padded_k * 2 + kNumBuckets + counters) * sizeof(uint32_t) + 24;
    if (slm_needed > params.engineInfo.maxLocalMemSize) {
        DO_NOT_USE_THIS_KERNEL(p.layerID);
    }

    return true;
}

ArgMaxMinKernelBase::DispatchData ArgMaxMinKernelTopKRadix::SetDefault(const arg_max_min_params& params) const {
    DispatchData dispatchData;

    size_t ops_size = 1;
    if (!params.has_dynamic_tensors()) {
        ops_size = GetOperationNumber(params);
    }

    dispatchData.gws = { ops_size * kWgSize, 1, 1 };
    dispatchData.lws = { kWgSize, 1, 1 };

    return dispatchData;
}

void ArgMaxMinKernelTopKRadix::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [this](const Params& params, KernelData& kd) {
        const auto& prim_params = static_cast<const arg_max_min_params&>(params);
        auto dispatchData = SetDefault(prim_params);
        OPENVINO_ASSERT(kd.kernels.size() == 1 || kd.kernels.size() == 2,
                        "[GPU] Invalid kernels size for update dispatch data func");
        // Only dynamic implementations get here, and their chunk config ignores the operation count.
        // A dynamic sort axis disables chunking at compile time, so the kernel count decides the mode.
        const ChunkConfig config = kd.kernels.size() == 1 ? ChunkConfig{} : GetChunkConfig(prim_params, false);
        OPENVINO_ASSERT(kd.kernels.size() == 1 || config.chunks > 1, "[GPU] Unexpected radix TopK chunk config");

        const bool skip_execution = KernelData::SkipKernelExecution(prim_params);
        for (size_t i = 0; i < kd.kernels.size(); ++i) {
            auto gws = dispatchData.gws;
            if (i == 0 && config.chunks > 1) {
                gws[0] *= config.chunks;
            }
            kd.kernels[i].params.workGroups.global = gws;
            kd.kernels[i].params.workGroups.local = dispatchData.lws;
            kd.kernels[i].skip_execution = skip_execution;
        }

        SetInternalBuffers(prim_params, config, kd);
    };
}

JitConstants ArgMaxMinKernelTopKRadix::GetJitConstants(const arg_max_min_params& params) const {
    auto jit = ArgMaxMinKernelBase::GetJitConstants(params);

    jit.AddConstant(MakeJitConstant("WG_SIZE", kWgSize));
    jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", kSubGroupSize));
    jit.AddConstant(MakeJitConstant("NUM_BUCKETS", kNumBuckets));
    jit.AddConstant(MakeJitConstant("BUCKET_GROUP_SIZE", kBucketGroupSize));
    jit.AddConstant(MakeJitConstant("TIE_GROUP_SIZE", kTieGroupSize));
    jit.AddConstant(MakeJitConstant("SORTABLE_BITS", params.inputs[0].GetDType() == Datatype::F16 ? 16 : 32));

    // PADDED_K: next power of 2 >= TOP_K (for bitonic sort)
    size_t padded_k = 1;
    while (padded_k < params.topK) padded_k <<= 1;
    jit.AddConstant(MakeJitConstant("PADDED_K", padded_k));

    if (params.has_dynamic_tensors()) {
        jit.AddConstant(MakeJitConstant("OPERATION_NUM", GetOperationNumberString(params)));
    } else {
        jit.AddConstant(MakeJitConstant("OPERATION_NUM", GetOperationNumber(params)));
    }

    if (params.argMaxMinSortType == ArgMaxMinSortType::VALUE) {
        jit.AddConstant(MakeJitConstant("SORT_BY_VALUE", 1));
    }

    if (params.values_first) {
        jit.AddConstant(MakeJitConstant("TOP_K_ORDER", 1));
    }

    return jit;
}

KernelsData ArgMaxMinKernelTopKRadix::GetKernelsData(const Params& params) const {
    if (!Validate(params)) {
        return {};
    }

    const auto& orgParams = static_cast<const arg_max_min_params&>(params);
    const ChunkConfig config = GetChunkConfig(orgParams, !orgParams.has_dynamic_tensors());
    const bool chunked = config.chunks > 1;
    const size_t kernels_num = chunked ? 2 : 1;
    const uint32_t internal_buffers_num = chunked ? 3 : 1;

    KernelData kd = KernelData::Default<arg_max_min_params>(params, kernels_num);
    GetUpdateDispatchDataFunc(kd);

    const auto base_jit = GetJitConstants(orgParams);
    for (size_t i = 0; i < kernels_num; ++i) {
        auto dispatchData = SetDefault(orgParams);
        auto cldnn_jit = base_jit;
        if (chunked) {
            cldnn_jit.AddConstants({
                MakeJitConstant(i == 0 ? "RADIX_CHUNK_STAGE" : "RADIX_MERGE_STAGE", 1),
                MakeJitConstant("RADIX_CHUNKS", config.chunks),
                MakeJitConstant("RADIX_CHUNK_SIZE", config.chunk_size),
                MakeJitConstant("RADIX_CANDIDATES_NUM", config.chunks * orgParams.topK),
            });
            if (i == 0) {
                dispatchData.gws[0] *= config.chunks;
            }
        }

        auto entry_point = GetEntryPoint(kernelName, orgParams.layerID, params, i);
        auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

        auto& kernel = kd.kernels[i];
        FillCLKernelData(kernel, dispatchData, params.engineInfo, kernelName, jit, entry_point,
                         EXE_MODE_DEFAULT, false, false, 1,
                         GetFusedPrimitiveInputsCount(params), orgParams.outputs_num,
                         orgParams.is_shape_agnostic);
        for (uint32_t buffer = 0; buffer < internal_buffers_num; ++buffer) {
            kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, buffer});
        }
    }

    SetInternalBuffers(orgParams, config, kd);

    return {kd};
}

KernelsPriority ArgMaxMinKernelTopKRadix::GetKernelsPriority(const Params& p) const {
    const auto& params = static_cast<const arg_max_min_params&>(p);
    const size_t sort_size = GetSortSize(params);

    // Radix sort excels at large sort sizes with large k (e.g., N=8400+, k=300).
    // For small sort sizes, and for k=1 (pure argmax/argmin) below 1024 values,
    // the axis kernel's simple reduction is more efficient — especially when there
    // are many independent operations that amplify per-WG overhead.
    if (sort_size < 256 || (params.topK == 1 && sort_size < 1024)) {
        return FORCE_PRIORITY_5;
    }

    return FORCE_PRIORITY_1;
}

}  // namespace kernel_selector
