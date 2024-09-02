// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include "ocl_engine.hpp"

#include "sycl/sycl.hpp"


namespace cldnn {
namespace ocl {

class sycl_engine : public ocl_engine {
public:
    sycl_engine(const device::ptr dev, runtime_types runtime_type);

    stream_ptr create_stream(const ExecutionConfig& config) const override;
    stream_ptr create_stream(const ExecutionConfig& config, void *handle) const override;

    ::sycl::context& get_sycl_context() const;
    std::unique_ptr<::sycl::context> sycl_context = nullptr;
};

}  // namespace ocl
}  // namespace cldnn
