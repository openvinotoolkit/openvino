// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <dnnl_types.h>
#include "graph_context.h"

namespace ov {
namespace intel_cpu {

const dnnl::engine& GraphContext::getEngine() {
    static const dnnl::engine eng(dnnl::engine::kind::cpu, 0);
    return eng;
}

}   // namespace intel_cpu
}   // namespace ov
