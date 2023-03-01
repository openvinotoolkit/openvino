// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "eltwise_inst.h"
#include "primitive_onednn_base.h"
#include "implementation_map.hpp"

#include "kernel_selector_common.h"

#include <oneapi/dnnl/dnnl.hpp>

#include <algorithm>
#include <memory>
namespace cldnn {
namespace onednn {

namespace detail {

attach_eltwise_onednn::attach_eltwise_onednn() {
}

}  // namespace detail
}  // namespace onednn
}  // namespace cldnn
