// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>

#include <functional>
#include <memory>
#include <mkldnn.hpp>
#include <vector>

namespace MKLDNNPlugin {

class MKLDNNPrimitive {
public:
    MKLDNNPrimitive();
    operator bool();
    MKLDNNPrimitive& operator=(const std::shared_ptr<mkldnn::primitive>& primitive);
    mkldnn::primitive operator*();

    void reset(mkldnn::primitive* primitive);

private:
    std::shared_ptr<mkldnn::primitive> prim;
};

}  // namespace MKLDNNPlugin