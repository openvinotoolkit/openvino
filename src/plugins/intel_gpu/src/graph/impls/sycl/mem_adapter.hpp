// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Copyright (C) 2026 FUJITSU LIMITED
//

#pragma once

#include "intel_gpu/runtime/memory.hpp"
#include "sycl/sycl_wrapper.hpp"
#include "sycl/sycl_memory.hpp"

#include <memory>
#include <utility>

namespace cldnn {
namespace sycl {

template<typename DType>
struct BufferAdapter {
    using BufferType = decltype(std::declval<cldnn::sycl::gpu_buffer>().get_buffer().reinterpret<DType>());
    BufferType buf;

    static BufferAdapter from_memory(const cldnn::memory::ptr& mem) {
        return { std::dynamic_pointer_cast<cldnn::sycl::gpu_buffer>(mem)->get_buffer().reinterpret<DType>() };
    }

    auto bind_read(::sycl::handler& cgh) {
        return buf.template get_access<::sycl::access::mode::read>(cgh);
    }
    auto bind_write(::sycl::handler& cgh) {
        return buf.template get_access<::sycl::access::mode::write>(cgh);
    }
};

template<typename DType>
struct UsmAdapter {
    DType* ptr;

    static UsmAdapter from_memory(const cldnn::memory::ptr& mem) {
        return { static_cast<DType*>(std::dynamic_pointer_cast<cldnn::sycl::gpu_usm>(mem)->get_buffer().get()) };
    }

    DType* bind_read(::sycl::handler& /*cgh*/) const { return ptr; }
    DType* bind_write(::sycl::handler& /*cgh*/) const { return ptr; }
};

}  // namespace sycl
}  // namespace cldnn
