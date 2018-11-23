// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mkldnn.hpp>
#include <functional>
#include <ie_common.h>
#include <vector>
#include <memory>
#include <details/ie_exception.hpp>

namespace MKLDNNPlugin {

class MKLDNNPrimitive {
public:
    MKLDNNPrimitive();
    operator bool();
    MKLDNNPrimitive& operator=(const std::shared_ptr<mkldnn::primitive>& prim);
    mkldnn::primitive operator*();

    void reset(mkldnn::primitive* prim);
    void setBatchLimit(int batch, size_t inputNum, size_t outputNum);

private:
    std::shared_ptr<mkldnn::primitive> prim;
    std::vector<int> originInputBatches;
    std::vector<int> originOutputBatches;
};

}  // namespace MKLDNNPlugin