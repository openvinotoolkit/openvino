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

#include "kernel_base.h"
#include "jitter.h"
#include <string>
#include <memory>

namespace kernel_selector {
struct CommonDispatchData {
    // TODO: change it to std::vector<size_t>
    size_t gws0, gws1, gws2;
    size_t lws0, lws1, lws2;
    bool
        fp16UnitUsed;  ///< Value indicating that FP16 half precision floating point type will be used (instead of single precision).
    float effiency;

    CommonDispatchData() : gws0(0), gws1(0), gws2(0), lws0(0), lws1(0), lws2(0), fp16UnitUsed(false), effiency(0.0f){}
};

class common_kernel_base : public KernelBase {
public:
    using KernelBase::KernelBase;
    virtual ~common_kernel_base() {}

protected:
    virtual bool Validate(const Params&, const optional_params&) const { return true; }
    std::string CreateJit(const std::string& template_name,
                          const JitConstants& constants,
                          const std::string& kernel_name) const;
    std::string GetEntryPoint(const std::string& templateName,
                              const std::string& layerID,
                              const optional_params& options) const;
    Arguments GetArgsDesc(uint32_t num_of_input,
                          bool use_weights,
                          bool use_bias,
                          bool use_quantization = false,
                          bool use_calibration = 0,
                          uint32_t number_of_inputs_for_fused_prim = 0) const;
    std::shared_ptr<KernelString> GetKernelString(const std::string& kernel_name,
                                                  const std::string& jit,
                                                  const std::string& entry_point,
                                                  const EngineInfo& engine_info,
                                                  const std::string& exe_mode = DEFAULT) const;
    void FillCLKernelData(clKernelData& kernel,
                          const CommonDispatchData& runInfo,
                          const EngineInfo& engine_info,
                          const std::string& kernel_map_name,
                          const std::string& jit,
                          const std::string& entry_point,
                          const std::string& exe_mode = DEFAULT,
                          bool weights = false,
                          bool bias = false,
                          uint32_t number_of_inputs = 1,
                          bool quantization = false,
                          bool calibration = false,
                          uint32_t number_of_inputs_for_fused_prims = 0) const;
};
}  // namespace kernel_selector
