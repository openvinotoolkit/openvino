// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mkldnn.hpp>
#include <functional>
#include <ie_common.h>
#include <vector>
#include <memory>

namespace MKLDNNPlugin {

class MKLDNNPrimitive {
public:
    MKLDNNPrimitive();
    operator bool() const;
    MKLDNNPrimitive& operator=(const std::shared_ptr<mkldnn::primitive>& primitive);
    mkldnn::primitive operator*();
    void reset(mkldnn::primitive* primitive);

private:
    std::shared_ptr<mkldnn::primitive> prim;
};

}  // namespace MKLDNNPlugin