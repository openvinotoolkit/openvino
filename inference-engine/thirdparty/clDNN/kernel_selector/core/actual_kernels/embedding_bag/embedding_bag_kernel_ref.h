/*
// Copyright (c) 2020 Intel Corporation
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
*/

#pragma once

#include "kernel_base_opencl.h"
#include "kernel_selector_params.h"

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// embedding_bag_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct embedding_bag_params : public base_params {
    embedding_bag_params() : base_params(KernelType::EMBEDDING_BAG), type(EmbeddingBagType::PACKED_SUM), default_index(-1) {}

    EmbeddingBagType type;
    int32_t default_index;

    virtual ParamsKey GetParamsKey() const { return base_params::GetParamsKey(); }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// embedding_bag_optional_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct embedding_bag_optional_params : optional_params {
    embedding_bag_optional_params() : optional_params(KernelType::EMBEDDING_BAG) {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// EmbeddingBagKernelRef
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class EmbeddingBagKernelRef : public KernelBaseOpenCL {
public:
    EmbeddingBagKernelRef() : KernelBaseOpenCL("embedding_bag_ref") {}
    virtual ~EmbeddingBagKernelRef() = default;

protected:
    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;

    virtual JitConstants GetJitConstants(const embedding_bag_params& params) const;
    virtual CommonDispatchData SetDefault(const embedding_bag_params& params) const;
    virtual bool Validate(const Params&, const optional_params&) const override;
};
}  // namespace kernel_selector
