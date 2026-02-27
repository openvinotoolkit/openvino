// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dynamic_parser.hpp"

#include "dynamic_graph.hpp"
#include "intel_npu/common/itt.hpp"

namespace intel_npu {

DynamicParser::DynamicParser(const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStruct)
    : _zeroInitStruct(zeroInitStruct),
      _logger("DynamicParser", Logger::global().level()) {
    _logger.info("initialize DynamicParser start");
}

std::shared_ptr<IGraph> DynamicParser::parse(const ov::Tensor& mainBlob,
                                             const FilteredConfig& config,
                                             const std::optional<std::vector<ov::Tensor>>&,
                                             std::optional<std::shared_ptr<const ov::Model>>&& model) const {
    OV_ITT_TASK_CHAIN(PARSE_BLOB, itt::domains::NPUPlugin, "DynamicParser", "parse");

    _logger.debug("Create graph for LLVM IR!");
    return std::make_shared<DynamicGraph>(_zeroInitStruct, std::move(mainBlob), true, config);
}

}  // namespace intel_npu
