// Copyright (c) 2018-2021 Intel Corporation
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

#include "kernel_base_opencl.h"
#include "kernel_selector_params.h"
#include <string>

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// mvn_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct mvn_params : public base_params {
    mvn_params() : base_params(KernelType::MVN) {}

    MVNMode mvnMode = MVNMode::WITHIN_CHANNELS;
    bool mvnNormalizeVariance = false;
    float epsilon = 0.0f;
    MVNEpsMode mvnEpsMode = MVNEpsMode::INSIDE_SQRT;

    virtual ParamsKey GetParamsKey() const {
        ParamsKey k = base_params::GetParamsKey();

        k.EnableMVNMode(mvnMode);

        if (mvnNormalizeVariance)
            k.EnableMVNNormalizeVariance();

        return k;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// mvn_optional_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct mvn_optional_params : optional_params {
    mvn_optional_params() : optional_params(KernelType::MVN) {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// MVNKernelBase
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class MVNKernelBase : public KernelBaseOpenCL {
public:
    using KernelBaseOpenCL::KernelBaseOpenCL;
    virtual ~MVNKernelBase() {}

    struct DispatchData : public CommonDispatchData {
        size_t itemsNum;
        size_t leftovers;
        size_t dataSetsCount;
        size_t dataSetSize;

        DispatchData() : itemsNum(0), leftovers(0), dataSetsCount(0), dataSetSize(0) {}
    };

protected:
    bool Validate(const Params&, const optional_params&) const override;
    virtual JitConstants GetJitConstants(const mvn_params& params, DispatchData dispatchData) const;
    virtual DispatchData SetDefault(const mvn_params& params) const;
    virtual std::string GetKernelName(const mvn_params&) const { return kernelName; }
    KernelsData GetCommonKernelsData(const Params& params, const optional_params&) const;
    Datatype GetActivationType(const mvn_params& params) const;
};
}  // namespace kernel_selector
