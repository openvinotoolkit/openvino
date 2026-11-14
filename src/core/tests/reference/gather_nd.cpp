// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/gather_nd.hpp"

#include <array>
#include <cstdint>

#include "gtest/gtest.h"
#include "openvino/core/except.hpp"
#include "openvino/reference/scatter_nd_update.hpp"

TEST(ReferenceGatherND, FullRankCoordinatesPositiveAndNegative) {
    const std::array<int, 6> params{0, 1, 2, 3, 4, 5};
    const std::array<int64_t, 4> indices{0, 1, -1, -1};
    std::array<int, 2> output{};

    const ov::Shape params_shape{2, 3};
    const ov::Shape indices_shape{2, 2};
    const ov::Shape output_shape{2};

    ov::reference::gather_nd(params.data(), indices.data(), output.data(), params_shape, indices_shape, output_shape);

    EXPECT_EQ(output, (std::array<int, 2>{1, 5}));
}

TEST(ReferenceGatherND, SliceGatherValidControl) {
    const std::array<int, 6> params{0, 1, 2, 3, 4, 5};
    const std::array<int64_t, 2> indices_valid{1, 0};
    std::array<int, 6> output{};

    const ov::Shape params_shape{2, 3};
    const ov::Shape indices_shape{2, 1};
    const ov::Shape output_shape{2, 3};

    ov::reference::gather_nd(params.data(),
                             indices_valid.data(),
                             output.data(),
                             params_shape,
                             indices_shape,
                             output_shape);

    EXPECT_EQ(output, (std::array<int, 6>{3, 4, 5, 0, 1, 2}));
}

TEST(ReferenceGatherND, OutOfBoundsCoordinateRejected) {
    const std::array<int, 6> params{0, 1, 2, 3, 4, 5};
    const std::array<int64_t, 2> indices_oob{0, 3};
    std::array<int, 1> output{};

    const ov::Shape params_shape{2, 3};
    const ov::Shape indices_shape{1, 2};
    const ov::Shape output_shape{1};

    EXPECT_THROW(ov::reference::gather_nd(params.data(),
                                          indices_oob.data(),
                                          output.data(),
                                          params_shape,
                                          indices_shape,
                                          output_shape),
                 ov::Exception);
}

TEST(ReferenceGatherND, ScatterNDUpdateScalarDataShape) {
    const std::array<int, 1> input_data{10};
    const std::array<int64_t, 0> indices{};
    const std::array<int, 1> updates{42};
    std::array<int, 1> output{0};

    const ov::Shape data_shape{};
    const ov::Shape indices_shape{1, 0};
    const ov::Shape updates_shape{1};

    ov::reference::scatterNdUpdate(input_data.data(),
                                   indices.data(),
                                   updates.data(),
                                   output.data(),
                                   data_shape,
                                   indices_shape,
                                   updates_shape);

    EXPECT_EQ(output[0], 42);
}
