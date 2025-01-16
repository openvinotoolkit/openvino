// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <oneapi/dnnl/dnnl.hpp>

#include "nodes/executors/dnnl/dnnl_post_op_data.hpp"

namespace ov {
namespace intel_cpu {

struct DnnlShapeAgnosticData {
    DnnlShapeAgnosticData(DnnlPrimitiveAttrs primAttrs)
        : primAttrs(std::move(primAttrs)) {}

    DnnlPrimitiveAttrs primAttrs;
};

using DnnlShapeAgnosticDataPtr = std::shared_ptr<DnnlShapeAgnosticData>;

}  // namespace intel_cpu
}  // namespace ov
