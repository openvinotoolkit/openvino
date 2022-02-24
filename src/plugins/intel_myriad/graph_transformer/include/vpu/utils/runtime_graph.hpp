// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vpu/utils/perf_report.hpp>
#include <cpp/ie_cnn_network.h>

#include <vector>

namespace vpu {

std::shared_ptr<ngraph::Function> buildRuntimeGraph(
        GraphMetaInfo& graphMetaInfo,
        const std::vector<float>& perfInfo);


}  // namespace vpu
