// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"

namespace kernel_selector {
struct generate_proposals_params : public base_params {
    generate_proposals_params()
            : base_params(KernelType::GENERATE_PROPOSALS) {}

    float min_size{0.0f};
    float nms_threshold{0.0f};
    size_t pre_nms_count{0};
    size_t post_nms_count{0};
    bool normalized{true};
    float nms_eta{1.0f};
    Datatype roi_num_type = Datatype::INT64;
};

struct generate_proposals_optional_params : public optional_params {
    generate_proposals_optional_params()
            : optional_params(KernelType::GENERATE_PROPOSALS) {}
};

class GenerateProposalsRef : public KernelBaseOpenCL {
public:
    GenerateProposalsRef()
            : KernelBaseOpenCL("generate_proposals_ref") {}

    ~GenerateProposalsRef() = default;

    using DispatchData = CommonDispatchData;

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;
protected:
    bool Validate(const Params& p, const optional_params& o) const override;
    void SetKernelArguments(const generate_proposals_params& params,
                            size_t idx, cldnn::arguments_desc& kernel) const;
};
}  // namespace kernel_selector
