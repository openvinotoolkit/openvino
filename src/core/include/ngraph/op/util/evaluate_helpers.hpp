// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/axis_set.hpp"
#include "ngraph/descriptor/tensor.hpp"
#include "ngraph/util.hpp"
#include "ngraph/validation_util.hpp"

namespace ngraph {
/// \brief Extracts the tensor data and returns a set of normalized axes created out of it.
///
/// \param tensor A pointer to a HostTensor object containing the raw axes data
/// \param rank Rank of an operator's input data tensor (used to normalize the axes)
/// \param node_description An identifier of the operator's node (used to report errors)
///
/// \return Normalized (positive only) axes as an AxisSet object.
AxisSet get_normalized_axes_from_tensor(const HostTensorPtr tensor,
                                        const ngraph::Rank& rank,
                                        const std::string& node_description);
}  // namespace ngraph
