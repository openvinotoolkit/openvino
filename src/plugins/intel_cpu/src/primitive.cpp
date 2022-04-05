// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <dnnl_types.h>
#include "primitive.h"

namespace ov {
namespace intel_cpu {

Primitive::Primitive() {}

Primitive::operator bool() const {
    return prim ? true : false;
}

dnnl::primitive Primitive::operator*() {
    return *prim;
}

void Primitive::reset(dnnl::primitive* primitive) {
    prim.reset(primitive);
}

Primitive &Primitive::operator=(const std::shared_ptr<dnnl::primitive>& primitive) {
    prim = primitive;
    return *this;
}

}   // namespace intel_cpu
}   // namespace ov
