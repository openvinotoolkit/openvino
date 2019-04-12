// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "inference_engine.hpp"
#include "prim_tensor.h"
#include "mkldnn.hpp"
#include <memory>

namespace MKLDNNPlugin {

class CpuPrimTensor;

using CpuPrimTensorPtr = std::shared_ptr<CpuPrimTensor>;

class CpuPrimTensor : public PrimTensor {
public:
    using Memory = std::shared_ptr<mkldnn::memory>;
    using PrimitiveDesc = std::shared_ptr<mkldnn::memory::primitive_desc>;

    explicit CpuPrimTensor(mkldnn::memory::desc desc) :
            desc(desc) {}


    mkldnn::memory getPrimitive() { return *(memory.get()); }

private:
    Memory memory;
    mkldnn::memory::desc desc;

    friend class CpuEngine;
};
}  // namespace MKLDNNPlugin