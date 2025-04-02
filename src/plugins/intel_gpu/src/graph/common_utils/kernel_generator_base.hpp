// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <type_traits>

#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/primitives/reorder.hpp"
#include "intel_gpu/runtime/kernel_args.hpp"

namespace micro {
struct MicroKernelPackage;
}  // namespace micro

namespace ov::intel_gpu {

using KernelParams = cldnn::kernel_arguments_desc;

using KernelString = cldnn::kernel_string;
using WorkGroupSizes = cldnn::work_group_sizes;
using ScalarDescriptor = cldnn::scalar_desc;
using Scalars = cldnn::scalars_desc;
using ArgumentDescriptor = cldnn::argument_desc;
using Arguments = cldnn::arguments_desc;
using cldnn::WeightsReorderParams;
using KernelLanguage = cldnn::kernel_language;

struct KernelData;
struct ImplRuntimeParams {};

struct DispatchDataFunc {
    using FuncType = std::function<void(const RuntimeParams&, KernelData&, ImplRuntimeParams*)>;
    FuncType m_dispatch_data_func = nullptr;

    template <typename Callable, typename std::enable_if_t<std::is_constructible_v<FuncType, Callable>, bool> = true>
    explicit DispatchDataFunc(Callable&& func) : m_dispatch_data_func(std::forward<Callable>(func)) {}
    explicit DispatchDataFunc(std::nullptr_t) {}

    void operator()(const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params = nullptr) const {
        m_dispatch_data_func(params, kd, rt_params);
    }
};

struct KernelData {
    std::shared_ptr<KernelString> code;
    KernelParams params;
    std::vector<std::shared_ptr<micro::MicroKernelPackage>> micro_kernels;
    DispatchDataFunc update_dispatch_data_func{nullptr};
    WeightsReorderParams weights_reorder_params;
    bool need_args_update{true};
    bool need_dispatch_data_update{true};

    void save(cldnn::BinaryOutputBuffer& ob) const;
    void load(cldnn::BinaryInputBuffer& ib);
};

class KernelGeneratorBase {
public:
    KernelGeneratorBase() = default;
    KernelGeneratorBase(const KernelGeneratorBase&) = delete;
    KernelGeneratorBase(KernelGeneratorBase&&) = delete;
    KernelGeneratorBase& operator=(const KernelGeneratorBase&) = delete;
    KernelGeneratorBase& operator=(KernelGeneratorBase&&) = delete;
    virtual ~KernelGeneratorBase() = default;

    [[nodiscard]] virtual KernelData get_kernel_data(const RuntimeParams& params) const = 0;
    [[nodiscard]] virtual DispatchDataFunc get_dispatch_data_func() const = 0;
};

}  // namespace ov::intel_gpu
