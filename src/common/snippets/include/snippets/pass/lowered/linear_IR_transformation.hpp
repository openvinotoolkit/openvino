// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/lowered_expr.hpp"
#include "openvino/core/rtti.hpp"
#include "openvino/core/type.hpp"

namespace ngraph {
namespace snippets {
namespace pass {
namespace lowered {

/**
 * @interface linearIRTransformation
 * @brief Base class for transformations on linear IR
 * @ingroup snippets
 */
class LinearIRTransformation {
public:
    LinearIRTransformation() = default;
    virtual ~LinearIRTransformation() = default;
    // Note that get_type_info_static and get_type_info are needed to mimic OPENVINO_RTTI interface,
    // so the standard OPENVINO_RTTI(...) macros could be used in derived classes.
    _OPENVINO_HIDDEN_METHOD static const ::ov::DiscreteTypeInfo& get_type_info_static() {
        static ::ov::DiscreteTypeInfo type_info_static {"LinearIRTransformation"};
        type_info_static.hash();
        return type_info_static;
    }

    virtual const DiscreteTypeInfo& get_type_info() const {
        return get_type_info_static();
    }

    const char* get_type_name() const {
        return get_type_info().name;
    }

    virtual bool run(LoweredExprIR& linear_ir) = 0;
};

} // namespace lowered
} // namespace pass
} // namespace snippets
} // namespace ngraph
