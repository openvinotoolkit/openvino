// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include "nodes/executors/dnnl/dnnl_post_op_data.hpp"
#include "onednn/iml_type_mapper.h"

namespace ov::intel_cpu {

struct DnnlShapeAgnosticData {
    explicit DnnlShapeAgnosticData(DnnlPrimitiveAttrs primAttrs,
                                   impl_desc_type implType = impl_desc_type::undef,
                                   dnnl::primitive_desc preselectedPrimitive = {})
        : m_primAttrs(std::move(primAttrs)),
          m_implType(implType),
          m_preselectedPrimitive(std::move(preselectedPrimitive)) {}

    DnnlPrimitiveAttrs m_primAttrs;
    // implementation type is a part of shape agnostic data to allow using
    // the same implementation for different shapes to avoid dealing with
    // multiple packed weights based on different implementations even it
    // may be not optimal from a performance perspective
    impl_desc_type m_implType;
    // A descriptor selected for an executor with static shapes.
    dnnl::primitive_desc m_preselectedPrimitive;
};

using DnnlShapeAgnosticDataPtr = std::shared_ptr<DnnlShapeAgnosticData>;

}  // namespace ov::intel_cpu
