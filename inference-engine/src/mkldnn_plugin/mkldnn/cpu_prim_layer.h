// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "inference_engine.hpp"
#include "prim_layer.h"
#include "mkldnn.hpp"
#include <memory>

using namespace InferenceEngine;
using namespace mkldnn;

namespace MKLDNNPlugin {

class CpuPrimLayer : public PrimLayer {
    friend class CpuEngine;

    mkldnn::engine eng;
    std::shared_ptr<mkldnn::primitive> prim;

public:
    explicit CpuPrimLayer(engine eng) : eng(eng) {}
};

template<typename LYR>
class Layer : public CpuPrimLayer {
    typename LYR::desc desc;
    typename LYR::primitive_desc prim_desc;

public:
    Layer(typename LYR::desc desc, engine eng) :
            CpuPrimLayer(eng),
            desc(desc),
            prim_desc(desc, eng) {}

    friend class CpuEngine;
};

class ReorderLayer : public CpuPrimLayer {
    reorder::primitive_desc prim_desc;

public:
    ReorderLayer(reorder::primitive_desc desc, engine eng) :
            CpuPrimLayer(eng),
            prim_desc(desc) {}

    friend class CpuEngine;
};
}  // namespace MKLDNNPlugin