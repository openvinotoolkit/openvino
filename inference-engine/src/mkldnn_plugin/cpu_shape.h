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
    const std::vector<size_t>& getMinDims() const {
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
    const std::vector<size_t>& getMaxDims() const {
        return maxDims;
    }

    /**
     * @brief return defined shape or throw exception for dynamic case 
     * @return return shape
     */
    const std::vector<size_t>& getStaticDims() const {
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

}  // namespace MKLDNNPlugin
