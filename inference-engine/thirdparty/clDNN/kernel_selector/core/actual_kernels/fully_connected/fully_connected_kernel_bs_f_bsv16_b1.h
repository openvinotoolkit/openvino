/*
// Copyright (c) 2016 Intel Corporation
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

#include "fully_connected_kernel_base.h"

namespace kernel_selector {

    class FullyConnected_bs_f_bsv16_b1 : public FullyConnectedKernelBase
    {
    public:
        FullyConnected_bs_f_bsv16_b1() : FullyConnectedKernelBase("fully_connected_gpu_bs_f_bsv16_b1") {}

        KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
        ParamsKey GetSupportedKey() const override;
    
    protected:
        struct DispatchData : public FullyConnectedKernelBase::DispatchData
        {
            DispatchData(const FullyConnectedKernelBase::DispatchData& base_dispatch_data)
                : FullyConnectedKernelBase::DispatchData(base_dispatch_data),
                unit_byte_size(0), chunk_type(nullptr), chunk_byte_size(0), units_per_chunk(0), bytes_per_sg_read(0),
                units_per_sg_read(0), responses_per_sg_exec(0), in_chunk_prefetch_size(0), filter_chunk_prefetch_size(0)
            {}

            uint32_t    unit_byte_size;
            const char* chunk_type;
            uint32_t    chunk_byte_size;
            uint32_t    units_per_chunk;
            uint32_t    bytes_per_sg_read;
            uint32_t    units_per_sg_read;
            uint32_t    responses_per_sg_exec;
            uint32_t    in_chunk_prefetch_size;
            uint32_t    filter_chunk_prefetch_size;
        };

        JitConstants GetJitConstants(const fully_connected_params& params, const FullyConnectedKernelBase::DispatchData& kd) const override;
        std::unique_ptr<FullyConnectedKernelBase::DispatchData> SetDefault(const fully_connected_params& arg) const override;
    };
}