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

#include "kernel_selector_common.h"
#include "kernel_selector_params.h"
#include "primitive_db.h"

namespace kernel_selector 
{
    using primitive_db = kernel_selector::gpu::cache::primitive_db;

    class KernelBase
    {
    public:
        KernelBase(const std::string name) : kernelName(name) {}
        virtual ~KernelBase() {}

        virtual KernelsData GetKernelsData(const Params& params, const optional_params& options) const = 0;
        virtual KernelsData GetKernelsDataForAutoTune(const Params& params, const optional_params& options) const
        {
            return GetKernelsData(params, options);
        }
        virtual KernelsData GetTunedKernelsDataByIndex(const Params& params, const optional_params& options, int /*autoTuneIndex*/) const
        {
            return GetKernelsData(params, options);
        }

        virtual bool Supports(const Params& params, const optional_params& options) const
        {
            const ParamsKey requireKey = params.GetParamsKey().Merge(options.GetSupportedKey());
            return GetSupportedKey().Support(requireKey);
        }

        bool SupportsTuning() const
        {
            return GetSupportedKey().TuningSupport();
        }

        virtual const std::string GetName() const { return kernelName; }

        static const primitive_db& get_db() { return db; }
    
    protected:
        static const primitive_db db;
        const std::string kernelName;

        static size_t UniqeID() { return counter++; } // TODO: use interlocked
        virtual ParamsKey GetSupportedKey() const = 0;
        
    private:
        static size_t counter;
    };
}
