// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/coordinate_transform.hpp"
#include "ngraph/shape.hpp"

using namespace ngraph;

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename dataType, typename indicesType>
            void scatterNdUpdate(const dataType* inputData,
                                 const indicesType* indices,
                                 const dataType* updates,
                                 dataType* outBuf,
                                 const Shape& dataShape,
                                 const Shape& indicesShape,
                                 const Shape& updatesShape)
            {
                size_t numSlices = 1;
                size_t sliceSize = 1;
                for (size_t i = 0; i < indicesShape.size() - 1; i++)
                {
                    numSlices *= indicesShape[i];
                }
                for (size_t i = indicesShape.size() - 1; i < updatesShape.size(); i++)
                {
                    sliceSize *= updatesShape[i];
                }

                const size_t k = indicesShape.back();
                std::memcpy(outBuf, inputData, sizeof(dataType) * shape_size(dataShape));
                CoordinateTransform dataTransform{dataShape};

                for (size_t i = 0; i < numSlices; i++)
                {
                    Coordinate coord;
                    for (size_t j = 0; j < k; j++)
                    {
                        coord.push_back(indices[i * k + j]);
                    }
                    for (size_t j = k; j < dataShape.size(); j++)
                    {
                        coord.push_back(0);
                    }

                    const size_t startDataIdx = dataTransform.index(coord);
                    for (size_t j = 0; j < sliceSize; j++)
                    {
                        outBuf[startDataIdx + j] = updates[i * sliceSize + j];
                    }
                }
            }
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
