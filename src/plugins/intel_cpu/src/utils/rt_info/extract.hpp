// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/core/node.hpp>
#include <utils/ngraph_utils.hpp>
#include "onednn/dnnl.h"
#include "onednn/iml_type_mapper.h"
#include <oneapi/dnnl/dnnl.hpp>

#include <string>
namespace ov {
namespace intel_cpu {
namespace rt_info {

std::string getOriginalLayerNames(const ov::Node::RTMap& rtInfo, const std::string& opName);

std::vector<impl_desc_type> getPrimitivesPriority(const ov::Node::RTMap& rtInfo);

std::vector<dnnl::memory::format_tag> getInputMemoryFormatsFilter(const ov::Node::RTMap& rtInfo);

std::vector<dnnl::memory::format_tag> getOutputMemoryFormatsFilter(const ov::Node::RTMap& rtInfo);

bool shouldEnforceBF16evenForGraphTail(const ov::Node::RTMap& rtInfo);

}   // namespace rt_info
}   // namespace intel_cpu
}   // namespace ov
