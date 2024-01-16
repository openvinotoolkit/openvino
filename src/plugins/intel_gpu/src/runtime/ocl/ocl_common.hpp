// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "ocl_wrapper.hpp"

#include "openvino/core/except.hpp"

#include <vector>

namespace cldnn {
namespace ocl {

using ocl_queue_type = cl::CommandQueue;
using ocl_kernel_type = cl::KernelIntel;

class ocl_error : public ov::Exception {
public:
    explicit ocl_error(cl::Error const& err);
};

#define OCL_ERR_MSG_FMT(err) ("[GPU] " + std::string(err.what()) + std::string(", error code: ") + std::to_string(err.err()))

}  // namespace ocl
}  // namespace cldnn
