// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <oneapi/dnnl/dnnl.hpp>

#include "nodes/executors/dnnl/dnnl_post_op_data.hpp"
#include "onednn/iml_type_mapper.h"

namespace ov::intel_cpu {

struct DnnlShapeAgnosticData {
    DnnlShapeAgnosticData(DnnlPrimitiveAttrs primAttrs, impl_desc_type implType = impl_desc_type::undef)
        : m_primAttrs(std::move(primAttrs)),
          m_implType(implType) {}

    DnnlPrimitiveAttrs m_primAttrs;
    // implementation type is a part of shape agnostic data to allow using
    // the same implementation for different shapes to avoid dealing with
    // multiple packed weights based on different implementations even it
    // may be not optimal from a performance perspective
    impl_desc_type m_implType;
};

using DnnlShapeAgnosticDataPtr = std::shared_ptr<DnnlShapeAgnosticData>;

}  // namespace ov::intel_cpu
