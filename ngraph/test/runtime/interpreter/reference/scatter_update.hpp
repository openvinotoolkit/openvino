// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include "ngraph/coordinate_transform.hpp"
#include "ngraph/shape.hpp"

using namespace ngraph;

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename dataType, typename indicesType, typename axisType>
            void scatterUpdate(const dataType* inputData,
                               const indicesType* indices,
                               const dataType* updates,
                               const axisType* _axis,
                               dataType* outBuf,
                               const Shape& dataShape,
                               const Shape& indicesShape,
                               const Shape& updatesShape)
            {
                int rank = static_cast<int>(dataShape.size());
                if (_axis[0] < -rank || _axis[0] > rank - 1)
                {
                    std::string error =
                        std::string("ScatterUpdate layer has out of bounds axis value: ") +
                        std::to_string(_axis[0]);
                    throw ngraph_error(error);
                }
                size_t axis = _axis[0] < 0 ? _axis[0] + rank : _axis[0];
                CoordinateTransform indicesTransform{indicesShape};

                Shape dataShapeIter = dataShape;
                dataShapeIter.erase(dataShapeIter.begin() + axis);
                CoordinateTransform dataTransfIter{dataShapeIter};

                CoordinateTransform updateTransform{updatesShape};
                CoordinateTransform dataTransform{dataShape};

                std::memcpy(outBuf, inputData, sizeof(dataType) * shape_size(dataShape));

                for (const Coordinate& indicesCoordIt : indicesTransform)
                {
                    const size_t indicesIdx = indicesTransform.index(indicesCoordIt);

                    if (indices[indicesIdx] < 0)
                    {
                        std::string error =
                            std::string("ScatterUpdate layer has negative index value: ") +
                            std::to_string(indices[indicesIdx]);
                        throw ngraph_error(error);
                    }
                    const size_t idx = static_cast<size_t>(indices[indicesIdx]);
                    if (dataShape[axis] <= idx)
                    {
                        std::string error =
                            std::string("ScatterUpdate layer has out of bounds coordinate: ") +
                            std::to_string(idx) + " on 'data' input on " + std::to_string(axis) +
                            "th axis";
                        throw ngraph_error(error);
                    }

                    for (const Coordinate& dataCoordIt : dataTransfIter)
                    {
                        Coordinate dataCoord = dataCoordIt;
                        dataCoord.insert(dataCoord.begin() + axis, idx);
                        const size_t startIndices = dataTransform.index(dataCoord);

                        auto updCoord = dataCoordIt;
                        updCoord.insert(
                            updCoord.begin() + axis, indicesCoordIt.begin(), indicesCoordIt.end());
                        const size_t startUpd = updateTransform.index(updCoord);
                        outBuf[startIndices] = updates[startUpd];
                    }
                }
            }
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
