// Copyright (c) 2016-2019 Intel Corporation
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


#pragma once

#include "kernel_selector_common.h"
#include "kernel_selector_params.h"

#include "jitter.h"
#include "primitive_db.h"
#include <string>
#include <vector>

namespace kernel_selector {
using primitive_db = kernel_selector::gpu::cache::primitive_db;

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

    virtual ParamsKey GetSupportedKey() const = 0;
    virtual const std::string GetName() const { return kernelName; }

    static const primitive_db& get_db() { return db; }
    static void ResetCounter() { counter = 0; }

protected:
    static const primitive_db db;
    const std::string kernelName;

    static size_t UniqeID() { return counter++; }  // TODO: use interlocked
    virtual Datatype GetUnitType(const base_params& params) const;

    bool IsFusedPrimitiveSupported(const fused_operation_desc& fused_op) const;
    JitConstants MakeBaseParamsJitConstants(const base_params& params) const;
    virtual std::vector<FusedOpType> GetSupportedFusedOps() const;
    virtual JitConstants MakeFusedOpsJitConstants(const base_params &params, const std::vector<FusedOpsConfiguration> &conf) const;
    virtual JitConstants MakeFusedOpsDeclsJitConstants(const base_params &params, const std::vector<FusedOpsConfiguration> &conf) const;

private:
    static size_t counter;
};
}  // namespace kernel_selector
