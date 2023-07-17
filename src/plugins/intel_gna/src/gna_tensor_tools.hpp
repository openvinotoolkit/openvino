// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_memcpy.h>

#include "gna_data_types.hpp"

namespace ov {
namespace intel_gna {

/**
 * @brief convert a tensor or its parts from NCHW to NHWC order on the base of transposition information.
 * The tensor to be converted from NCHW to NHWC may be 2D. But we may need to change data order inside one of its
 * dimensions since its data is reshaped to/from 4D data used by convolution. TranspositionInfo contains number of rows
 * (corresponds to C dimension) and columns (corresponds to HW dimensions) of every part of the tensor which should or
 * shouldn't be transposed (for example if output of some convolution layer is concatenated with output of another
 * convolution layer and there is a fully connected layer after that, we need to transpose its weights parts
 * corresponding to different convolutions outputs separately). If the tensor is input, output data, scaleshift weights
 * or a constant input of eltwise, it's rows number is equal to 1 and its columns number is equal to the product of
 * input dimensions (N * C * H * W). The row (or its separate parts) should be transposed. If the tensor is fully
 * connected layer weights, it's rows number is equal to the product of input dimensions and it's columns number is
 * equal to the product of output dimensions. Every row of this tensor (or its separate parts) should be transposed if
 * there are convolution layers before this layer. Every column (or its separate parts) should be transposed if there
 * are convolution layers after this layer.
 * @param precision data precision
 * @param rows number of rows in the whole tensor
 * @param columns number of columns in the whole tensor
 * @param buffer pointer to a tensor buffer
 * @param transpose_rows flag indicated if tensor rows or tensor columns must be transposed
 * @param transpositionInfo vector of structures with information about transposition: every element contains
 * information about tensor part which should be transposed separately or shouldn't be transposed
 */
inline void ConvertTensorFromNCHWToNHWC(size_t precision,
                                        size_t rows,
                                        size_t columns,
                                        uint8_t* buffer,
                                        bool transpose_rows,
                                        const std::vector<TranspositionInfo>& transpositionInfo) {
    size_t weightsTotalSize = rows * columns * precision;
    std::vector<uint8_t> transposedWeights(weightsTotalSize);
    size_t weightsPartOffset = 0;
    bool transposed = false;
    for (const auto& transpositionInfoPart : transpositionInfo) {
        auto partSize = transpositionInfoPart.num_transpose_rows * transpositionInfoPart.num_transpose_columns;
        size_t weightsPartSize = partSize * precision * (transpose_rows ? rows : columns);
        if (transpositionInfoPart.transpose && transpositionInfoPart.num_transpose_rows != 1 &&
            transpositionInfoPart.num_transpose_columns != 1) {
            if (transpose_rows) {
                for (size_t weightsRowIx = 0; weightsRowIx < rows; ++weightsRowIx) {
                    auto weightsRowsOffset = weightsRowIx * partSize * precision;
                    auto cbuffer = buffer + weightsPartOffset + weightsRowsOffset;
                    auto weights_ptr = transposedWeights.data() + weightsPartOffset + weightsRowsOffset;
                    for (size_t colsIx = 0; colsIx < transpositionInfoPart.num_transpose_columns; ++colsIx) {
                        for (size_t rowIx = 0; rowIx < transpositionInfoPart.num_transpose_rows; ++rowIx) {
                            auto offsetWrite = (colsIx * transpositionInfoPart.num_transpose_rows + rowIx) * precision;
                            auto offsetRead =
                                (transpositionInfoPart.num_transpose_columns * rowIx + colsIx) * precision;
                            ie_memcpy(weights_ptr + offsetWrite,
                                      weightsPartSize - weightsRowsOffset - offsetWrite,
                                      cbuffer + offsetRead,
                                      precision);
                        }
                    }
                }
            } else {
                auto cbuffer = buffer + weightsPartOffset;
                auto weights_ptr = transposedWeights.data() + weightsPartOffset;
                for (size_t colsIx = 0; colsIx < transpositionInfoPart.num_transpose_columns; ++colsIx) {
                    for (size_t rowIx = 0; rowIx < transpositionInfoPart.num_transpose_rows; ++rowIx) {
                        auto offsetWrite =
                            (colsIx * transpositionInfoPart.num_transpose_rows + rowIx) * columns * precision;
                        auto offsetRead =
                            (transpositionInfoPart.num_transpose_columns * rowIx + colsIx) * columns * precision;
                        ie_memcpy(weights_ptr + offsetWrite,
                                  weightsPartSize - offsetWrite,
                                  cbuffer + offsetRead,
                                  columns * precision);
                    }
                }
            }
            transposed = true;
        } else {
            // Just copy data which should not be transposed
            ie_memcpy(transposedWeights.data() + weightsPartOffset,
                      weightsPartSize,
                      buffer + weightsPartOffset,
                      weightsPartSize);
        }
        weightsPartOffset += weightsPartSize;
    }
    if (transposed) {
        ie_memcpy(buffer, weightsTotalSize, transposedWeights.data(), weightsTotalSize);
    }
}

}  // namespace intel_gna
}  // namespace ov
