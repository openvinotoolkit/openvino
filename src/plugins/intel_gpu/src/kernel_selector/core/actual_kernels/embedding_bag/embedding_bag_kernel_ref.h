// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;

    virtual JitConstants GetJitConstants(const embedding_bag_params& params) const;
    virtual CommonDispatchData SetDefault(const embedding_bag_params& params) const;
    bool Validate(const Params&, const optional_params&) const override;
};
}  // namespace kernel_selector
