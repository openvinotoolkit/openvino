// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <string>
#include <vector>

#include "cpu_types.h"
#include "openvino/core/except.hpp"
#include "openvino/core/interval.hpp"
#include "openvino/core/partial_shape.hpp"

namespace ov::intel_cpu {

class Shape {
public:
    Shape() = default;

    explicit Shape(const ov::PartialShape& shape) {
        if (!shape.rank().is_dynamic()) {
            const auto shape_rank = shape.rank().get_length();
            minDims.reserve(shape_rank);
            maxDims.reserve(shape_rank);

            for (const auto& d : shape) {
                minDims.push_back(d.get_min_length() == ov::Interval::s_max ? UNDEFINED_DIM : d.get_min_length());
                maxDims.push_back(d.get_max_length() == ov::Interval::s_max ? UNDEFINED_DIM : d.get_max_length());
            }
        }

        type = shape.is_static() ? ShapeType::Static : ShapeType::Dynamic;
        initDims();

        hasZeroDimensions = std::any_of(dims.begin(), dims.end(), [](size_t dim) {
            return dim == 0;
        });
    }

    explicit Shape(const VectorDims& shape) : dims(minDims = maxDims = shape) {
        hasZeroDimensions = std::any_of(dims.begin(), dims.end(), [](size_t dim) {
            return dim == 0;
        });
    }

    Shape(const VectorDims& minDims, const VectorDims& maxDims) {
        OPENVINO_ASSERT(minDims.size() == maxDims.size(),
                        "Can't create shape due to min/max vectors dims size mismatch");
        this->minDims = minDims;
        this->maxDims = maxDims;

        initDims();

        if (std::any_of(dims.begin(), dims.end(), [](size_t dim) {
                return dim == Shape::UNDEFINED_DIM;
            })) {
            type = ShapeType::Dynamic;
        } else {
            type = ShapeType::Static;
        }

        hasZeroDimensions = std::any_of(dims.begin(), dims.end(), [](size_t dim) {
            return dim == 0;
        });
    }

    Shape(const std::initializer_list<Dim>& shape) {
        minDims.reserve(shape.size());
        maxDims.reserve(shape.size());

        for (auto dim : shape) {
            minDims.push_back(dim);
            maxDims.push_back(dim);
        }

        initDims();

        hasZeroDimensions = std::any_of(dims.begin(), dims.end(), [](size_t dim) {
            return dim == 0;
        });
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
    [[nodiscard]] const VectorDims& getMinDims() const {
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
    [[nodiscard]] const VectorDims& getMaxDims() const {
        return maxDims;
    }

    /**
     * @brief return defined shape or throw exception for dynamic case
     * @return return shape
     */
    [[nodiscard]] const VectorDims& getStaticDims() const {
        OPENVINO_ASSERT(type == ShapeType::Static, "Cannot get dims for non static shape");
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
    [[nodiscard]] const VectorDims& getDims() const {
        return dims;
    }

    [[nodiscard]] bool isStatic() const {
        return type == ShapeType::Static;
    }

    [[nodiscard]] bool isDynamic() const {
        return type == ShapeType::Dynamic;
    }

    [[nodiscard]] bool hasZeroDims() const {
        return hasZeroDimensions;
    }

    [[nodiscard]] size_t getRank() const {
        return minDims.size();
    }

    [[nodiscard]] size_t getElementsCount() const {
        OPENVINO_ASSERT(type == ShapeType::Static, "Cannot get elements count for non static shape");
        size_t size = 1;

        for (uint64_t minDim : minDims) {
            size *= minDim;
        }

        return size;
    }

    [[nodiscard]] ov::PartialShape toPartialShape() const {
        using ov::Dimension;
        std::vector<Dimension> nGraphDims;
        nGraphDims.reserve(minDims.size());
        for (size_t i = 0; i < minDims.size(); i++) {
            Dimension::value_type minDim = Shape::UNDEFINED_DIM == minDims[i] ? -1 : minDims[i];
            Dimension::value_type maxDim = Shape::UNDEFINED_DIM == maxDims[i] ? -1 : maxDims[i];
            nGraphDims.emplace_back(minDim, maxDim);
        }
        return {nGraphDims};
    }

    [[nodiscard]] bool isCompatible(const VectorDims& vecDims) const;

    [[nodiscard]] std::string toString() const;

    bool operator==(const Shape& rhs) const {
        return minDims == rhs.minDims && maxDims == rhs.maxDims;
    }

    bool operator!=(const Shape& rhs) const {
        return !(*this == rhs);
    }

    [[nodiscard]] bool hasDefinedUpperBounds() const {
        return std::all_of(maxDims.begin(), maxDims.end(), [](Dim dim) {
            return dim != UNDEFINED_DIM;
        });
    }

    enum : Dim { UNDEFINED_DIM = std::numeric_limits<Dim>::max() };

private:
    void initDims() {
        dims.resize(minDims.size());
        for (size_t i = 0; i < minDims.size(); i++) {
            dims[i] = minDims[i] == maxDims[i] ? minDims[i] : UNDEFINED_DIM;
        }
    }

    enum class ShapeType : uint8_t { Static, Dynamic } type{ShapeType::Static};

    bool hasZeroDimensions = false;

    VectorDims minDims;
    VectorDims maxDims;
    VectorDims dims;
};

/**
 * @brief Merges two shapes overlapping their dims intervals.
 * @note When one of the dims intervals are not overlapped an exception is thrown.
 * @param lhs
 * first shape
 * @param rhs
 * second shape
 * @return resulting shape
 */

Shape mergeShapes(const Shape& lhs, const Shape& rhs);

}  // namespace ov::intel_cpu
