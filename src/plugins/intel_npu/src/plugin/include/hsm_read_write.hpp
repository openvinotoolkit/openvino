// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_npu/common/blob_sections_registry.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "openvino/runtime/hsm_reader.hpp"

namespace intel_npu::hsm {

BlobSectionsRegistry read_from(
    ov::runtime::hsm::Reader& base_reader,
    const std::unordered_map<SectionType, ov::runtime::hsm::SectionDecoder<std::shared_ptr<ISection>>>& readers,
    const std::unordered_map<SectionType, std::shared_ptr<ISectionTypeEvaluator>>& type_evaluators,
    const std::unordered_map<SectionType, std::shared_ptr<ISectionInstanceEvaluator>>& instance_evaluators);

void write_to(std::ostream& stream, const BlobSectionsRegistry& blob_sections_registry);

}  // namespace intel_npu::hsm
