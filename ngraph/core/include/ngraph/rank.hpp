// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/dimension.hpp"

namespace ngraph
{
    /// \brief Alias for Dimension, used when the value represents the number of axes in a shape,
    ///        rather than the size of one dimension in a shape.
    ///
    /// XXX: THIS TYPE IS EXPERIMENTAL AND THE ENTIRE DESIGN IS SUBJECT TO CHANGE.
    using Rank = Dimension;
} // namespace ngraph
