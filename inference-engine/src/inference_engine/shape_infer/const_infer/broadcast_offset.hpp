// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_blob.h>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <ie_layers.h>
#include <ie_precision.hpp>
#include <precision_utils.h>

namespace InferenceEngine {
namespace ShapeInfer {
class BroadcastOffset {
        SizeVector dims;
        SizeVector offset_v;

        SizeVector getDims(const SizeVector& originDims, const SizeVector& outputDims) {
            SizeVector d(outputDims.size(), 1);
            for (int i = 0; i < originDims.size(); i++) {
                d[d.size() - 1 - i] = originDims[originDims.size() - 1 - i];
            }
            return d;
        }

        SizeVector getOffset(const SizeVector& originDims, const SizeVector& outDims) {
            SizeVector o(originDims.size());
            if (originDims.size() != outDims.size())
                THROW_IE_EXCEPTION << "Cannot calculate offsets! Incorrect patameters for eltwise broadcast!";
            int k = 1;
            for (int i = originDims.size() - 1; i >= 0; i--) {
                o[i] = (originDims[i] == outDims[i]) ? k : 0;
                k *= originDims[i];
            }
            return o;
        }

    public:
        BroadcastOffset(const SizeVector& originDims, const SizeVector& outputDims) {
            dims = getDims(originDims, outputDims);
            offset_v = getOffset(dims, outputDims);
        }

        size_t offset(const SizeVector& v) const {
            size_t off = 0;
            if (v.size() != offset_v.size())
                THROW_IE_EXCEPTION << "Cannot calculate offsets! Incorrect patameters for eltwise broadcast!";
            for (size_t i = 0; i < v.size(); i++) {
                off += v[i] * offset_v[i];
            }
            return off;
        }

        SizeVector offset_dims(size_t l) const {
            size_t n_dims = dims.size();
            SizeVector pos(n_dims);
            for (int rd = 1; rd <= n_dims; ++rd) {
                const size_t d = n_dims - rd;
                const size_t cur_dim = dims[d];
                pos[d] = l % cur_dim;
                l /= cur_dim;
            }
            return pos;
        }
};
}  // namespace ShapeInfer
}  // namespace InferenceEngine