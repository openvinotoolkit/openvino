// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_selector_params.h"
#include "intel_gpu/runtime/kernel_args.hpp"

#include <cfloat>
#include <cstdint>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#define EXE_MODE_DEFAULT ""
#define EXE_MODE_AGE_BASED "-cl-no-subgroup-ifp"
#define EXE_MODE_NO_PRERA_SCH "-cl-intel-no-prera-scheduling"

namespace micro {
struct MicroKernelPackage;
}  // namspace

namespace kernel_selector {

#ifndef UNUSED
#define UNUSED(a) (void)a
#endif

// TODO: current solution until we will have kernel selection time based
#define KernelsPriority float
#define FORCE_PRIORITY_1 (0.0000001f)
#define FORCE_PRIORITY_2 (0.0000002f)
#define FORCE_PRIORITY_3 (0.0000003f)
#define FORCE_PRIORITY_4 (0.0000004f)
#define FORCE_PRIORITY_5 (0.0000005f)
#define FORCE_PRIORITY_6 (0.0000006f)
#define FORCE_PRIORITY_7 (0.0000007f)
#define FORCE_PRIORITY_8 (0.0000008f)
#define FORCE_PRIORITY_9 (0.0000009f)
#define DONT_USE_IF_HAVE_SOMETHING_ELSE (1000000.f)
#define TUTORIAL_PRIORITY (DONT_USE_IF_HAVE_SOMETHING_ELSE + 1.f)
#define NOT_SUPPORTED (FLT_MAX)

std::string GetStringEnv(const char* varName);

using KernelString = cldnn::kernel_string;
using WorkGroupSizes = cldnn::work_group_sizes;
using ScalarDescriptor = cldnn::scalar_desc;
using Scalars = cldnn::scalars_desc;
using ArgumentDescriptor = cldnn::argument_desc;
using Arguments = cldnn::arguments_desc;
using KernelParams = cldnn::kernel_arguments_desc;


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// KernelCode
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct KernelCode {
    std::shared_ptr<KernelString> kernelString;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// clKernelData
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct clKernelData {
    KernelCode code;
    KernelParams params;
    std::vector<std::shared_ptr<micro::MicroKernelPackage>> micro_kernels;
    bool skip_execution = false;

    void save(cldnn::BinaryOutputBuffer& ob) const;
    void load(cldnn::BinaryInputBuffer& ib);
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// WeightsReorderParams
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct WeightsReorderParams {
    WeightsTensor src;
    WeightsTensor dest;
    bool rotate = false;
    bool is_initialized = false;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// KernelData
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct KernelData {
    std::shared_ptr<Params> params;
    std::vector<clKernelData> kernels;
    std::vector<size_t> internalBufferSizes;
    Datatype internalBufferDataType = Datatype::UNSUPPORTED;
    uint64_t runTime = std::numeric_limits<uint64_t>::max();  // kernel run time in nanoseconds

    bool reorderInput = false;
    WeightsReorderParams weightsReorderParams;
    std::string kernelName;

    std::function<void(const Params&, KernelData&)> update_dispatch_data_func = nullptr;

    int autoTuneIndex = -1;

    bool can_reuse_memory = true;
    bool needs_sub_kernels_sync = true;

    static bool SkipKernelExecution(const base_params& params, size_t kernel_id = 0) {
        for (const auto& input : params.inputs) {
            if (input.LogicalSize() == 0) {
                return true;
            }
        }
        for (const auto& output : params.outputs) {
            if (output.LogicalSize() == 0) {
                return true;
            }
        }
        return false;
    }

    static bool SkipKernelExecution(const Params& params, size_t kernel_id = 0) {
        return false;
    }

    template <typename T>
    inline static KernelData Default(const Params& _params, size_t kernel_nums = 1) {
        KernelData kd;
        const T& orgParams = static_cast<const T&>(_params);
        kd.params = std::make_shared<T>(orgParams);
        kd.kernels.resize(kernel_nums);
        kd.runTime = std::numeric_limits<uint64_t>::max();
        kd.reorderInput = false;  // for KW
        kd.autoTuneIndex = -1;
        kd.can_reuse_memory = true;
        kd.needs_sub_kernels_sync = true;

        for (auto& kernel : kd.kernels) {
            kernel.skip_execution = SkipKernelExecution(orgParams);
        }
        return kd;
    }
};

using KernelsData = std::vector<KernelData>;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// to string functions
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::string toString(ActivationFunction activation);
std::string toString(DataLayout l);
std::string toString(Datatype dType);
std::string toString(WeightsType wType);
std::string toString(KernelType kt);
std::string toString(EltwiseMode b_mode);
std::string toString(MeanSubtractMode mode);
std::string toString(ArgMaxMinOut mode);
std::string toString(ArgMaxMinAxis mode);
std::string toString(PoolType mode);
std::string toString(LRNMode mode);
std::string toString(KernelDividerMode mode);
std::string toString(SoftmaxDim d);
std::string toString(NormalizeMode mode);
std::string toString(MVNMode mode);
std::string toString(MVNEpsMode mode);
std::string toString(WeightsLayout layout);
std::string toString(ConcatAxis a);
std::string toString(GatherAxis a);
std::string toString(ScatterUpdateAxis a);
std::string toString(ResampleType type);
std::string toString(CoordinateTransformationMode mode);
std::string toString(NearestMode mode);
std::string toString(const BorderType type);
std::string toString(const Tensor::Dim& dim);
std::string toString(const DataTensor& tensor);
std::string toString(const WeightsTensor& tensor);
std::string toString_v2(const DataTensor& tensor);
std::string toString(const IndexSelectAxis& axis);
std::string toString(ReduceMode mode);
inline std::uint64_t create_hash(const unsigned char* begin, const unsigned char* end) {
    // Compatible with VS std::hash.
    constexpr auto start_acc = static_cast<std::uint64_t>(UINT64_C(14695981039346656037));
    constexpr auto mul_factor = static_cast<std::uint64_t>(UINT64_C(1099511628211));

    std::uint64_t acc = start_acc;
    for (auto elem_it = begin; elem_it != end; ++elem_it) {
        acc ^= static_cast<std::uint64_t>(*elem_it);
        acc *= mul_factor;
    }

    return acc;
}

template <typename ElemTy>
std::uint64_t create_hash(const ElemTy* begin, const std::size_t size) {
    return create_hash(reinterpret_cast<const unsigned char*>(begin),
                       reinterpret_cast<const unsigned char*>(begin + size));
}

template <typename CharTy, typename CharTraits, typename AllocatorTy>
std::uint64_t create_hash(const std::basic_string<CharTy, CharTraits, AllocatorTy>& value) {
    return create_hash<CharTy>(value.data(), value.size());
}
}  // namespace kernel_selector
