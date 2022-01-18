// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_selector_common.h"
#include "kernel_runner_interface.h"
#include "auto_tuner.h"
#include <vector>
#include <memory>
#include <string>
#include <map>

namespace kernel_selector {
class KernelBase;

using KernelList = std::vector<std::shared_ptr<KernelBase>>;
using ForceList = std::map<std::string, bool>;

class kernel_selector_base {
public:
    kernel_selector_base();
    virtual ~kernel_selector_base() {}

    virtual KernelsData GetBestKernels(const Params& params, const optional_params& options) const = 0;

protected:
    template <typename T>
    inline void Attach() {
        implementations.push_back(std::make_shared<T>());
    }

    virtual KernelsData GetNaiveBestKernel(const Params& params,
                                           const optional_params& options,
                                           KernelType kType) const;

    virtual KernelsData GetAutoTuneBestKernel(const Params& params,
                                              const optional_params& options,
                                              KernelType kType) const;

    KernelList GetAllImplementations(const Params& params, const optional_params& options, KernelType kType) const;

    KernelList implementations;
    ForceList forceKernels;

    static AutoTuner autoTuner;
};
}  // namespace kernel_selector
