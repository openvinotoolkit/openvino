// Copyright (C) 2018-2026 Intel Corporation
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
    ~KernelBaseOpenCL() override = default;

protected:
    virtual bool Validate(const Params& /*unused*/) const {
        return true;
    }
    static std::pair<std::string, std::string> CreateJit(const std::string& template_name, const JitConstants& constants, const std::string& kernel_id);
    static std::string GetEntryPoint(const std::string& templateName, const std::string& layerID, const Params& params, size_t partID = 0);
    static Arguments GetArgsDesc(uint32_t num_of_input,
                                 bool use_weights,
                                 bool use_bias,
                                 uint32_t number_of_inputs_for_fused_prim = 0,
                                 uint32_t num_of_output = 1,
                                 bool is_dynamic = false);
    static std::shared_ptr<KernelString> GetKernelString(const std::string& kernel_name,
                                                         const std::pair<std::string, std::string>& jit,
                                                         const std::string& entry_point,
                                                         const EngineInfo& engine_info,
                                                         const std::string& exe_mode = EXE_MODE_DEFAULT);

    static uint32_t GetFusedPrimitiveInputsCount(const Params& params);

    static void FillCLKernelData(clKernelData& kernel,
                                 const CommonDispatchData& dispatchData,
                                 const EngineInfo& engine_info,
                                 const std::string& kernelMapName,
                                 const std::pair<std::string, std::string>& jit,
                                 const std::string& entryPoint,
                                 const std::string& exeMode = EXE_MODE_DEFAULT,
                                 bool weights = false,
                                 bool bias = false,
                                 int number_of_inputs = 1,
                                 uint32_t number_of_inputs_for_fused_prims = 0,
                                 int number_of_outputs = 1,
                                 bool is_dynamic = false);

    static bool layout_is_one_of(const MultiDataTensor& tensors, const std::vector<DataLayout>& allowed_layouts);
};
}  // namespace kernel_selector
