// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "perf_count.h"
#include <vector>
#include <utility>
#include <ie_common.h>
#include <ngraph/partial_shape.hpp>
#include "mkldnn_dims.h"

namespace MKLDNNPlugin {

class Shape {
public:
    Shape() = default;

    explicit Shape(const ngraph::PartialShape& shape) {
        minDims = shape.get_min_shape();
        maxDims = shape.get_max_shape();
        type = shape.is_static() ? ShapeType::Static : ShapeType::Dynamic;

        initDims();
    }

    explicit Shape(const InferenceEngine::SizeVector& shape) {
        minDims = shape;
        maxDims = shape;
        type = ShapeType::Static;

        initDims();
    }

    // TODO [DS]: Added for migration period. Should be deleted once we will get rid of MKLDNNDims.
    explicit Shape(const MKLDNNDims& shape) {
        minDims = shape.ToSizeVector();
        maxDims = shape.ToSizeVector();
        type = ShapeType::Static;

        initDims();
    }

    const std::vector<size_t>& getMinDims() const {
        return minDims;
    }
    const std::vector<size_t>& getMaxDims() const {
        return maxDims;
    }
    const std::vector<size_t>& getStaticDims() const {
        if (type != ShapeType::Static) {
            IE_THROW() << "Cannot get dims for non static shape";
        }

        return minDims;
    }

    mkldnn::memory::dims getStaticMklDims() const {
        auto& staticDims = getStaticDims();
        return mkldnn::memory::dims(staticDims.begin(), staticDims.end());
    }

    const std::vector<size_t>& getDims() const {
        return dims;
    }
    bool isStatic() const {
        return type == ShapeType::Static;
    }

    size_t getRank() const {
        return minDims.size();
    }

    size_t getElementsCount() const {
        if (type != ShapeType::Static) {
            IE_THROW() << "Cannot get elements count for non static shape";
        }

        size_t size = 1;

        for (int i = 0; i < minDims.size(); i++) {
            size *= minDims[i];
        }

        return size;
    }

    ngraph::PartialShape toPartialShape() const {
        std::vector<ngraph::Dimension> nGraphDims;
        nGraphDims.reserve(minDims.size());
        for (int i = 0; i < minDims.size(); i++) {
            nGraphDims.emplace_back(minDims[i], maxDims[i]);
        }
        return ngraph::PartialShape(nGraphDims);
    }

    // TODO [DS]: Added for migration period. Should be deleted once we will get rid of MKLDNNDims.
    operator MKLDNNDims() const {
        return MKLDNNDims(getStaticDims());
    }

    bool operator == (const Shape& rhs) const {
        return minDims == rhs.minDims && maxDims == rhs.maxDims;
    }

    bool operator != (const Shape& rhs) const {
        return !(*this == rhs);
    }

    enum : size_t {
        UNDEFINED_DIM = 0xffffffffffffffff
    };

private:
    void initDims() {
        dims.resize(minDims.size());
        for (int i = 0; i < minDims.size(); i++) {
            dims[i] = minDims[i] == maxDims[i] ? minDims[i] : UNDEFINED_DIM;
        }
    }

    enum class ShapeType {
        Static,
        Dynamic
    } type {ShapeType::Static};

    std::vector<size_t> minDims;
    std::vector<size_t> maxDims;
    std::vector<size_t> dims;
};

inline bool dimsEqualStrong(size_t lhs, size_t rhs) {
    return (lhs == rhs && lhs != Shape::UNDEFINED_DIM && rhs != Shape::UNDEFINED_DIM);
}

inline bool dimsEqualWeak(size_t lhs, size_t rhs) {
    return (lhs == Shape::UNDEFINED_DIM || rhs == Shape::UNDEFINED_DIM || lhs == rhs);
}

inline bool isEqualOrUndefined(const std::vector<size_t> lhs, const std::vector<size_t>& rhs, size_t skipAxis = Shape::UNDEFINED_DIM) {
    if (lhs.size() != rhs.size())
        return false;

    for (size_t i = 0; i < lhs.size(); i++) {
        if (i != skipAxis && !dimsEqualWeak(lhs[i], rhs[i]))
            return false;
    }

    return true;
}

}  // namespace MKLDNNPlugin
