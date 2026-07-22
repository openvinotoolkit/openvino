// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Copyright (C) 2026 FUJITSU LIMITED
//

#pragma once

#include "sycl_common.hpp"
#include "intel_gpu/runtime/kernel_args.hpp"
#include "intel_gpu/runtime/kernel.hpp"

#include <mutex>
#include <vector>

namespace cldnn {
namespace sycl {

class sycl_base_kernel : public cldnn::kernel {
public:
    // Backend-independent description of one kernel argument slot.
    struct arg_t {
        enum class kind_t { BUFFER, SCALAR, LOCAL_MEM };
        kind_t kind = kind_t::BUFFER;
        memory::ptr mem;         // BUFFER
        scalar_desc scalar{};    // SCALAR
        size_t local_size = 0;   // LOCAL_MEM (bytes)
    };

    // Store per-dispatch kernel arguments (called before launch).
    void set_arguments(const kernel_arguments_desc& args_desc,
                       const kernel_arguments_data& data);

    // Record this kernel into the given SYCL command group (handler).
    virtual void launch(::sycl::handler& cgh,
                        const kernel_arguments_desc& args_desc) = 0;

protected:
    // Access stored arguments by reference.  Safe when the caller runs on the
    // same thread that produced them.
    const std::vector<arg_t>& stored_args() const { return _stored_args; }

    // Snapshot of the stored arguments as a value copy.  Required when the
    // arguments are consumed asynchronously (e.g. captured by a host_task
    // lambda that runs after set_arguments/launch have returned).
    std::vector<arg_t> stored_args_snapshot() const;

    mutable std::mutex _args_mutex;
    std::vector<arg_t> _stored_args;
};

}  // namespace sycl
}  // namespace cldnn
