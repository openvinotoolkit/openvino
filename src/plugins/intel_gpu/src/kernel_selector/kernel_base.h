// Copyright (C) 2018-2025 Intel Corporation
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

    virtual KernelsData GetKernelsData(const Params& params) const = 0;
    virtual KernelsData GetKernelsDataForAutoTune(const Params& params) const {
        return GetKernelsData(params);
    }
    virtual KernelsData GetTunedKernelsDataByIndex(const Params& params, int /*autoTuneIndex*/) const {
        return GetKernelsData(params);
    }
    virtual KernelsPriority GetKernelsPriority(const Params& /*params*/) const {
        return DONT_USE_IF_HAVE_SOMETHING_ELSE;
    }

    virtual ParamsKey GetSupportedKey() const = 0;
    virtual DeviceFeaturesKey get_required_device_features_key(const Params& params) const {
        return DeviceFeaturesKey();
    }
    virtual const std::string GetName() const { return kernelName; }
    virtual void GetUpdateDispatchDataFunc(KernelData& kd) const { }

    static const primitive_db& get_db() { return db; }

protected:
    static const primitive_db db;
    const std::string kernelName;

    static void CheckDispatchData(const std::string& kernelName, const kernel_selector::CommonDispatchData& dispatchData,
                                  const size_t maxWorkGroupSize);
    virtual Datatype GetUnitType(const base_params& params) const;

    bool IsFusedPrimitiveSupported(const fused_operation_desc& fused_op) const;
    bool IsSIMDSizeSupported(const EngineInfo& info, size_t simd_size) const;
    JitConstants MakeBaseParamsJitConstants(const base_params& params, bool add_tensor_definitions = true) const;
    virtual std::vector<FusedOpType> GetSupportedFusedOps() const;
    virtual JitConstants MakeFusedOpsJitConstants(const base_params &params, const std::vector<FusedOpsConfiguration> &conf) const;
    virtual JitConstants MakeFusedOpsDeclsJitConstants(const base_params &params, const std::vector<FusedOpsConfiguration> &conf) const;

    // Basic check for extensions requirements
    // If params has at least 1 tensor of fp32/fp16/(u)int8 type, then it sets corresponding subgroup extension as required
    // Should be used carefully, as in some cases tensor types may not correspond to actually used extension
    // e.g. int8 kernel may use intel_sub_group_block_read to load packed int8 data
    DeviceFeaturesKey get_common_subgroups_device_features_key(const Params& params) const;
};
}  // namespace kernel_selector
