// Copyright (C) 2018 Intel Corporation
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
        dims = std::vector<int>(size.begin(), size.end());
    }

    explicit MKLDNNDims(const std::vector<int>& dim) {
        dims = dim;
    }

    MKLDNNDims(const mkldnn_dims_t dnn_dims, int dnn_ndims) {
        dims = std::vector<int>(dnn_dims, dnn_dims + dnn_ndims);
    }

    explicit MKLDNNDims(std::initializer_list<int> ilist) : dims(ilist) {}
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

    int size() const {
        return size(0);
    }

    int size(int start) const {
        int size = 1;

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

    bool operator == (const MKLDNNDims& rhs) {
        if (dims.size() != rhs.dims.size()) {
            return false;
        }

        return std::equal(rhs.dims.begin(), rhs.dims.end(), dims.begin());
    }

    bool operator != (const MKLDNNDims& rhs) {
        return !(*this == rhs);
    }

    int& operator[](int idx) {
        return dims[idx];
    }

    int operator[](int idx) const {
        return dims[idx];
    }

private:
    std::vector<int> dims;
};

}  // namespace MKLDNNPlugin
