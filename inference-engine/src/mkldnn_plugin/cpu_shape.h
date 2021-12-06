// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "perf_count.h"
#include <vector>
#include <utility>
#include <ie_common.h>
#include <ngraph/partial_shape.hpp>
#include "cpu_types.h"

namespace MKLDNNPlugin {

class Shape {
public:
    Shape() = default;

    explicit Shape(const ngraph::PartialShape& shape) {
        minDims = shape.get_min_shape();
        std::transform(minDims.begin(), minDims.end(), minDims.begin(), [](Dim x){ return ngraph::Interval::s_max == x ? UNDEFINED_DIM : x;});
        maxDims = shape.get_max_shape();
        std::transform(maxDims.begin(), maxDims.end(), maxDims.begin(), [](Dim x){ return ngraph::Interval::s_max == x ? UNDEFINED_DIM : x;});
        type = shape.is_static() ? ShapeType::Static : ShapeType::Dynamic;

        initDims();

        hasZeroDimensions = std::any_of(dims.begin(), dims.end(), [](size_t dim) { return dim == 0; } );
    }

    explicit Shape(const InferenceEngine::SizeVector& shape) {
        minDims = shape;
        maxDims = shape;
        type = ShapeType::Static;

        initDims();

        hasZeroDimensions = std::any_of(dims.begin(), dims.end(), [](size_t dim) { return dim == 0; } );
    }

    /**
     * @brief
     * for static shape
     * maxDims = [2, 3, 4, 5]
     * minDims = [2, 3, 4, 5]
     * dims = [2, 3, 4, 5]
     * @return return lower bound of shape = [2, 3, 4, 5]
     * for dynamic shape
     * maxDims = [6, 6, 6, 6]
     * minDims = [1, 1, 1, 1]
     * dims = [UNDEFINED_DIM, UNDEFINED_DIM, UNDEFINED_DIM, UNDEFINED_DIM]
     * @return return lower bound of shape = [1, 1, 1, 1]
     */
    const VectorDims& getMinDims() const {
        return minDims;
    }

    /**
     * @brief
     * for static shape
     * maxDims = [2, 3, 4, 5]
     * minDims = [2, 3, 4, 5]
     * dims = [2, 3, 4, 5]
     * @return return upper bound of shape = [2, 3, 4, 5]
     * for dynamic shape
     * maxDims = [6, 6, 6, 6]
     * minDims = [1, 1, 1, 1]
     * dims = [UNDEFINED_DIM, UNDEFINED_DIM, UNDEFINED_DIM, UNDEFINED_DIM]
     * @return return upper bound of shape = [6, 6, 6, 6]
     */
    const VectorDims& getMaxDims() const {
        return maxDims;
    }

    /**
     * @brief return defined shape or throw exception for dynamic case
     * @return return shape
     */
    const VectorDims& getStaticDims() const {
        if (type != ShapeType::Static) {
            IE_THROW() << "Cannot get dims for non static shape";
        }

        return minDims;
    }

    /**
     * @brief
     * for static shape
     * maxDims = [2, 3, 4, 5]
     * minDims = [2, 3, 4, 5]
     * dims = [2, 3, 4, 5]
     * @return return defined shape = [2, 3, 4, 5]
     * for dynamic shape
     * maxDims = [2, 3, 6, 6]
     * minDims = [2, 3, 1, 1]
     * dims = [2, 3, UNDEFINED_DIM, UNDEFINED_DIM]
     * @return return shape with defined and undefined dims = [2, 3, UNDEFINED_DIM, UNDEFINED_DIM]
     */
    const VectorDims& getDims() const {
        return dims;
    }

    bool isStatic() const {
        return type == ShapeType::Static;
    }

    bool isDynamic() const {
        return type == ShapeType::Dynamic;
    }

    bool hasZeroDims() const {
        return hasZeroDimensions;
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
        using ngraph::Dimension;
        std::vector<Dimension> nGraphDims;
        nGraphDims.reserve(minDims.size());
        for (int i = 0; i < minDims.size(); i++) {
            Dimension::value_type minDim = Shape::UNDEFINED_DIM == minDims[i] ? -1 : minDims[i];
            Dimension::value_type maxDim = Shape::UNDEFINED_DIM == maxDims[i] ? -1 : maxDims[i];
            nGraphDims.emplace_back(minDim, maxDim);
        }
        return ngraph::PartialShape(nGraphDims);
    }

    bool isCompatible(const VectorDims& vecDims) const;

    std::string toString() const;

    bool operator == (const Shape& rhs) const {
        return minDims == rhs.minDims && maxDims == rhs.maxDims;
    }

    bool operator != (const Shape& rhs) const {
        return !(*this == rhs);
    }

    bool hasDefinedUpperBounds() const {
        return std::all_of(maxDims.begin(), maxDims.end(), [](Dim dim){ return dim != UNDEFINED_DIM; });
    }

    enum : Dim {
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

    bool hasZeroDimensions = false;

    VectorDims minDims;
    VectorDims maxDims;
    VectorDims dims;
};
}  // namespace MKLDNNPlugin
