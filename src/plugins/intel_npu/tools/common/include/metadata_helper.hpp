//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "metadata.hpp"

namespace npu {
namespace utils {

/**
 * @brief Parses metadata from an input stream
 *
 * @param stream input stream where blob with metadata resides
 */
std::pair<uint32_t, std::unique_ptr<intel_npu::MetadataBase>> parseNPUMetadata(std::istream& stream);

/**
 * @brief Prints content of a NPU metadata alongside it's version
 *
 * @param version The version of a metadata
 * @param metadataPtr pointer to the metadata itself
 */
void printNPUMetadata(uint32_t version, const intel_npu::MetadataBase* metadataPtr);

}  // namespace utils
}  // namespace npu
