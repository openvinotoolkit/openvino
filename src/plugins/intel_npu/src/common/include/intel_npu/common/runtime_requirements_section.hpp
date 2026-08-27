// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cre.hpp"
#include "intel_npu/common/isection.hpp"

namespace intel_npu {

/**
 * @brief Section able to write the runtime requirements of the whole model
 */
class RuntimeRequirementsSection final : public ISection {
public:
    /**
     * @brief Construct a section able to write the runtime requirements of the whole model.
     *
     * @param sections_requirements A mapping from <string section type>_<section ID> to the corresponding runtime
     * requirements. The map is ordered, in order to assure subsequent writes produce identical blobs.
     * @param cre The compatiblity requirements expression, tying together the requirements of individual sections.
     */
    RuntimeRequirementsSection(const std::map<std::string, std::string>& sections_requirements,
                               const CRE& cre,
                               const ov::log::Level log_level = ov::log::Level::WARNING);

    void write(BlobWriterInterface& writer) override;

    std::map<std::string, std::string> get_sections_requirements() const;

    CRE get_cre() const;

    static std::shared_ptr<ISection> read(BlobReaderInterface& blob_reader);

private:
    std::map<std::string, std::string> m_sections_requirements;
    CRE m_cre;

    Logger m_logger;
};

}  // namespace intel_npu
