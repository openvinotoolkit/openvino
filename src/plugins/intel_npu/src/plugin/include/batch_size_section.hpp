// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>

#include "intel_npu/common/isection.hpp"
#include "openvino/core/layout.hpp"

namespace intel_npu {

class BatchSizeSection final : public ISection {
public:
    BatchSizeSection(const int64_t batch_size);

    void write(std::ostream& stream, BlobWriter* writer) override;

    std::optional<uint64_t> get_length() const override;

    int64_t get_batch_size() const;

    static std::shared_ptr<ISection> read(BlobReader* blob_reader, const size_t section_length);

private:
    int64_t m_batch_size;
};

}  // namespace intel_npu
