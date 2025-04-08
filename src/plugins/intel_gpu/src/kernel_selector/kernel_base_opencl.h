// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base.h"
#include "jitter.h"
#include <string>
#include <memory>
#include <vector>

namespace kernel_selector {

class KernelBaseOpenCL : public KernelBase {
public:
    using KernelBase::KernelBase;
    virtual ~KernelBaseOpenCL() {}

protected:
    virtual bool Validate(const Params&) const { return true; }
    std::pair<std::string, std::string> CreateJit(const std::string& template_name,
                          const JitConstants& constants,
                          const std::string& kernel_name) const;
    std::string GetEntryPoint(const std::string& templateName,
                              const std::string& layerID,
                              const Params& params,
                              const size_t partID = 0) const;
    Arguments GetArgsDesc(uint32_t num_of_input,
                          bool use_weights,
                          bool use_bias,
                          uint32_t number_of_inputs_for_fused_prim = 0,
                          uint32_t num_of_outpus = 1,
                          bool is_dynamic = false) const;
    std::shared_ptr<KernelString> GetKernelString(const std::string& kernel_name,
                                                  const std::pair<std::string, std::string>& jit,
                                                  const std::string& entry_point,
                                                  const EngineInfo& engine_info,
                                                  const std::string& exe_mode = EXE_MODE_DEFAULT) const;

    uint32_t GetFusedPrimitiveInputsCount(const Params &params) const;

    void FillCLKernelData(clKernelData& kernel,
                          const CommonDispatchData& dispatchData,
                          const EngineInfo& engine_info,
                          const std::string& kernel_map_name,
                          const std::pair<std::string, std::string>& jit,
                          const std::string& entry_point,
                          const std::string& exe_mode = EXE_MODE_DEFAULT,
                          bool weights = false,
                          bool bias = false,
                          int number_of_inputs = 1,
                          uint32_t number_of_inputs_for_fused_prims = 0,
                          int number_of_outputs = 1,
                          bool is_dynamic = false) const;

    bool layout_is_one_of(const MultiDataTensor& tensors, const std::vector<DataLayout>& allowed_layouts) const;
};
}  // namespace kernel_selector
