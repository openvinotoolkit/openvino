// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ocl_stream.hpp"
#include "sycl_engine.hpp"

#include "sycl/sycl.hpp"


namespace cldnn {
namespace ocl {

class sycl_stream : public ocl_stream {
public:
    sycl_stream(const sycl_engine& engine, const ExecutionConfig& config);
    sycl_stream(const sycl_engine &engine, const ExecutionConfig& config, void *handle);

    ::sycl::queue& get_sycl_queue();
    std::unique_ptr<::sycl::queue> sycl_queue = nullptr;
};

}  // namespace ocl
}  // namespace cldnn
