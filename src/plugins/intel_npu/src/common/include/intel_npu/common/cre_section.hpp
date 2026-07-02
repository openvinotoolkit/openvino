// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cre.hpp"
#include "intel_npu/common/isection.hpp"

namespace intel_npu {

class CRESection final : public ISection {
public:
    CRESection(const CRE& cre, const ov::log::Level log_level = ov::log::Level::WARNING);

    void write(BlobWriterInterface& writer) override;

    CRE get_cre() const;

    static std::shared_ptr<ISection> read(BlobReaderInterface& blob_reader);

private:
    CRE m_cre;

    Logger m_logger;
};

}  // namespace intel_npu
