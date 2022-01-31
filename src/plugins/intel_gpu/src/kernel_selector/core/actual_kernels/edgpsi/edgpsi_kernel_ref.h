// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"

namespace kernel_selector {
struct edgpsi_params : public base_params {
    edgpsi_params() : base_params(KernelType::EXPERIMENTAL_DETECTRON_GENERATE_PROPOSALS_SINGLE_IMAGE) {}

    float min_size{0.0f};
    float nms_threshold{0.0f};
    size_t pre_nms_count{0};
    size_t post_nms_count{0};
};

struct edgpsi_optional_params : public optional_params {
    edgpsi_optional_params() : optional_params(KernelType::EXPERIMENTAL_DETECTRON_GENERATE_PROPOSALS_SINGLE_IMAGE) {}
};

class EDGPSIRef : public KernelBaseOpenCL {
public:
    EDGPSIRef() : KernelBaseOpenCL("edgpsi_gpu_ref") {}
    ~EDGPSIRef() = default;

    using DispatchData = CommonDispatchData;

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;
protected:
    bool Validate(const Params& p, const optional_params& o) const override;
    void SetKernelArguments(const edgpsi_params& params, size_t idx, cldnn::arguments_desc& kernel) const;
};
}  // namespace kernel_selector
