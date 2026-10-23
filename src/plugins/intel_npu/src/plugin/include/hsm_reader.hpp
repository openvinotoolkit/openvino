// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_npu/common/blob_sections_registry.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "openvino/runtime/hsm_reader.hpp"

namespace intel_npu::hsm {

// TODO too many resources wasted to make this a class?
class Reader final {
public:
    Reader(const std::unordered_map<SectionType, ov::runtime::hsm::SectionDecoder<std::shared_ptr<ISection>>>& readers,
           const std::unordered_map<SectionType, std::shared_ptr<ISectionTypeEvaluator>>& type_evaluators,
           const std::unordered_map<SectionType, std::shared_ptr<ISectionInstanceEvaluator>>& instance_evaluators);

    BlobSectionsRegistry read(ov::runtime::hsm::Reader& base_reader) const;

private:
    std::unordered_map<SectionType, ov::runtime::hsm::SectionDecoder<std::shared_ptr<ISection>>> m_readers;
    std::unordered_map<SectionType, std::shared_ptr<ISectionTypeEvaluator>> m_type_evaluators;
    std::unordered_map<SectionType, std::shared_ptr<ISectionInstanceEvaluator>> m_instance_evaluators;

    Logger m_logger;
};

}  // namespace intel_npu::hsm
