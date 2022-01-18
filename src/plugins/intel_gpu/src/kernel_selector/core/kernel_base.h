// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_selector_common.h"
#include "kernel_selector_params.h"

#include "jitter.h"
#include "primitive_db.h"
#include <string>
#include <vector>

namespace kernel_selector {
using primitive_db = kernel_selector::gpu::cache::primitive_db;

struct CommonDispatchData {
    std::vector<size_t> gws;
    std::vector<size_t> lws;

    CommonDispatchData() : gws({0, 0, 0}), lws({0, 0, 0}) {}
};

std::string toString(const kernel_selector::CommonDispatchData& dispatchData);

static inline std::ostream &operator<<(std::ostream &os, CommonDispatchData disptchData) {
    return os << toString(disptchData);
}

class KernelBase {
public:
    using FusedOpType = KernelType;
    using LoadType = FusedOpsConfiguration::LoadType;
    using BoundaryCheck = FusedOpsConfiguration::BoundaryCheck;
    using IndexType = FusedOpsConfiguration::IndexType;

    explicit KernelBase(const std::string name) : kernelName(name) {}
    virtual ~KernelBase() {}

    virtual KernelsData GetKernelsData(const Params& params, const optional_params& options) const = 0;
    virtual KernelsData GetKernelsDataForAutoTune(const Params& params, const optional_params& options) const {
        return GetKernelsData(params, options);
    }
    virtual KernelsData GetTunedKernelsDataByIndex(const Params& params,
                                                   const optional_params& options,
                                                   int /*autoTuneIndex*/) const {
        return GetKernelsData(params, options);
    }
    virtual KernelsPriority GetKernelsPriority(const Params& /*params*/,
                                               const optional_params& /*options*/) const {
        return DONT_USE_IF_HAVE_SOMETHING_ELSE;
    }

    virtual ParamsKey GetSupportedKey() const = 0;
    virtual const std::string GetName() const { return kernelName; }

    static const primitive_db& get_db() { return db; }

protected:
    static const primitive_db db;
    const std::string kernelName;

    static void CheckDispatchData(const std::string& kernelName, const kernel_selector::CommonDispatchData& dispatchData,
                                  const size_t maxWorkGroupSize);
    virtual Datatype GetUnitType(const base_params& params) const;

    bool IsFusedPrimitiveSupported(const fused_operation_desc& fused_op) const;
    bool IsSIMDSizeSupported(const EngineInfo& info, size_t simd_size) const;
    JitConstants MakeBaseParamsJitConstants(const base_params& params) const;
    virtual std::vector<FusedOpType> GetSupportedFusedOps() const;
    virtual JitConstants MakeFusedOpsJitConstants(const base_params &params, const std::vector<FusedOpsConfiguration> &conf) const;
    virtual JitConstants MakeFusedOpsDeclsJitConstants(const base_params &params, const std::vector<FusedOpsConfiguration> &conf) const;
};
}  // namespace kernel_selector
