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

    std::optional<uint64_t> get_length() const override;

private:
    std::vector<ov::Layout> m_input_layouts;
    std::vector<ov::Layout> m_output_layouts;
};

}  // namespace intel_npu
