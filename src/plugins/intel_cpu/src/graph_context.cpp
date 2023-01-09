// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <dnnl_types.h>
#include "graph_context.h"

namespace ov {
namespace intel_cpu {

dnnl::engine GraphContext::eng(dnnl::engine::kind::cpu, 0);

}   // namespace intel_cpu
}   // namespace ov
