// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <ie_common.h>
#include <vector>
#include <memory>

#include "onednn/dnnl.h"

namespace ov {
namespace intel_cpu {

class Primitive {
public:
    Primitive();
    operator bool() const;
    Primitive& operator=(const std::shared_ptr<dnnl::primitive>& primitive);
    dnnl::primitive operator*();
    void reset(dnnl::primitive* primitive);

private:
    std::shared_ptr<dnnl::primitive> prim;
};

}   // namespace intel_cpu
}   // namespace ov
