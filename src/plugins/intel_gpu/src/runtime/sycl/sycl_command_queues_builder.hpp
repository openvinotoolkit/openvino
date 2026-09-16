// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Copyright (C) 2026 FUJITSU LIMITED
//

#pragma once

#include "sycl_common.hpp"
#include "intel_gpu/runtime/engine.hpp"

namespace cldnn {
namespace sycl {

class command_queues_builder {
public:
    command_queues_builder(bool profiling = false, bool out_of_order = false);
    sycl_queue_type build(const ::sycl::context& context, const ::sycl::device& device);

private:
    bool _profiling;
    bool _out_of_order;
    ::sycl::property_list get_properties(const ::sycl::device& device);
};

}  // namespace sycl
}  // namespace cldnn
