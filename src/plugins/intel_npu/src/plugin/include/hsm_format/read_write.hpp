// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "hsm_format/blob_sections_registry.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "openvino/runtime/hsm_reader.hpp"

namespace intel_npu::hsm {

/**
 * @brief Parses the NPU formatted HSM container and places the resulted sections inside a new registry.
 * @details The first parsing step is to look for a runtime requirements section and evaluate it. If the runtime
 * requirements could not be evaluated for whatever reason, then all sections are considered optional. Otherwise, the
 * handling of each section will depend on the evaluation results. Some sections will be considered mandatory. The
 * others will be either skipped or parsed optionally.
 *
 * @param base_reader The OV common reader, used to extract the payload of each section.
 * @param readers Handlers capable of taking a buffer formatted according to a section type and turning it into a
 * section instance of the same type.
 * @param type_evaluators Teach the CRE how to evaluate the section types contained within. Note that missing entries
 * will be evaluated to "supported" by default.
 * @param instance_evaluators Teach the CRE how to evaluate every section instance of a given type. Note that missing
 * entries will be evaluated to "supported" by default.
 * @return A blob sections registry containing all sections parsed during the call. If a runtime requirements section
 * was found and evaluated, then the call will also return the runtime requirements object.
 */
std::pair<BlobSectionsRegistry, std::optional<RuntimeRequirements>> read_from(
    ov::runtime::hsm::Reader& base_reader,
    const std::unordered_map<SectionType, ov::runtime::hsm::SectionDecoder<std::shared_ptr<ISection>>>& readers,
    const std::unordered_map<SectionType, std::shared_ptr<ISectionTypeEvaluator>>& type_evaluators,
    const std::unordered_map<SectionType, std::shared_ptr<ISectionInstanceEvaluator>>& instance_evaluators);

/**
 * @brief Writes all sections found in the registry and their runtime requirements to the provided stream using the HSM
 * format.
 */
void write_to(std::ostream& stream, const BlobSectionsRegistry& blob_sections_registry);

}  // namespace intel_npu::hsm
