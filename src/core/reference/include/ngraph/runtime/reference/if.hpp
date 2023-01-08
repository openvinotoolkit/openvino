// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>

#include "ngraph/op/util/multi_subgraph_base.hpp"

namespace ngraph {
namespace runtime {
namespace reference {
void if_reference(const std::vector<std::shared_ptr<Function>>& body,
                  const std::vector<op::util::MultiSubgraphOutputDescriptionVector>& out_descs,
                  const std::vector<op::util::MultiSubgraphInputDescriptionVector>& input_descs,
                  const HostTensorVector& out,
                  const HostTensorVector& args);
}
}  // namespace runtime
}  // namespace ngraph