// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gemm_inst.h"
#include "primitive_onednn_base.h"
#include "impls/implementation_map.hpp"

#include "kernel_selector_common.h"

#include <oneapi/dnnl/dnnl.hpp>

#include <algorithm>
#include <memory>
namespace cldnn {
namespace onednn {

namespace detail {

attach_gemm_onednn::attach_gemm_onednn() {
}

}  // namespace detail
}  // namespace onednn
}  // namespace cldnn
