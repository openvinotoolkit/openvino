// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "sycl_common.hpp"
#include "intel_gpu/runtime/engine.hpp"

namespace cldnn {
namespace sycl {

class command_queues_builder {
public:
    command_queues_builder();
    sycl_queue_type build(const ::sycl::context& context, const ::sycl::device& device);
    void set_profiling(bool flag) { _profiling = flag; }
    void set_out_of_order(bool flag) { _out_of_order = flag; }

private:
    bool _profiling;
    bool _out_of_order;
    ::sycl::property_list get_properties(const ::sycl::device& device);
};

}  // namespace sycl
}  // namespace cldnn
