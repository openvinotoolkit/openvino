// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/ngraph.hpp"
#include "batch_extraction_configuration.hpp"

namespace vpu {

SliceConfiguration sliceUnaryEltwise(const ngraph::Node& node);

}  // namespace vpu
