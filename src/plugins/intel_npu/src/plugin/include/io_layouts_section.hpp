// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>

#include "intel_npu/common/isection.hpp"
#include "openvino/core/layout.hpp"

namespace intel_npu {

class IOLayoutsSection final : public ISection {
public:
    IOLayoutsSection(const std::vector<ov::Layout>& input_layouts, const std::vector<ov::Layout>& output_layouts);

    void write(std::ostream& stream, BlobWriter* writer) override;

    std::vector<ov::Layout> get_input_layouts() const;

    std::vector<ov::Layout> get_output_layouts() const;

    static std::shared_ptr<ISection> read(BlobReader* blob_reader, const size_t section_length);

private:
    std::vector<ov::Layout> m_input_layouts;
    std::vector<ov::Layout> m_output_layouts;
};

}  // namespace intel_npu
