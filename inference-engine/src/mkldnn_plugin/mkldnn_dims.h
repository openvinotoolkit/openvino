// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "perf_count.h"
#include <vector>
#include <utility>
#include <mkldnn_types.h>
#include <ie_common.h>
#include <mkldnn.hpp>

namespace MKLDNNPlugin {

class MKLDNNDims {
public:
    MKLDNNDims() = default;

    explicit MKLDNNDims(const InferenceEngine::SizeVector& size) {
        dims = std::vector<ptrdiff_t>(size.begin(), size.end());
    }

    explicit MKLDNNDims(const std::vector<ptrdiff_t>& dim) {
        dims = dim;
    }

    MKLDNNDims(const mkldnn_dims_t dnn_dims, int dnn_ndims) {
        dims = std::vector<ptrdiff_t>(dnn_dims, dnn_dims + dnn_ndims);
    }

    explicit MKLDNNDims(std::initializer_list<ptrdiff_t> ilist) : dims(ilist) {}
    explicit MKLDNNDims(std::initializer_list<size_t > ilist) : dims(ilist.begin(), ilist.end()) {}

    InferenceEngine::SizeVector ToSizeVector() const {
        InferenceEngine::SizeVector size;
        for (auto i : dims) {
            size.push_back(i);
        }

        return size;
    }

    int ndims() const {
        return dims.size();
    }

    ptrdiff_t size() const {
        return size(0);
    }

    ptrdiff_t size(int start) const {
        ptrdiff_t size = 1;

        for (int i = start; i < dims.size(); i++) {
            size *= dims[i];
        }

        return size;
    }

    void push_back(int val) {
        dims.push_back(val);
    }

    operator mkldnn::memory::dims() const {
        return dims;
    }

    bool operator == (const MKLDNNDims& rhs) const {
        if (dims.size() != rhs.dims.size()) {
            return false;
        }

        return std::equal(rhs.dims.begin(), rhs.dims.end(), dims.begin());
    }

    bool operator != (const MKLDNNDims& rhs) const {
        return !(*this == rhs);
    }

    ptrdiff_t& operator[](int idx) {
        return dims[idx];
    }

    ptrdiff_t operator[](int idx) const {
        return dims[idx];
    }

private:
    std::vector<ptrdiff_t> dims;
};

}  // namespace MKLDNNPlugin
