// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "hsm_format/isection.hpp"
#include "openvino/core/layout.hpp"

namespace intel_npu {

/**
 * @brief Section responsible for encoding and decoding the input and output layouts of the original model.
 */
class IOLayoutsSection final : public ISection {
public:
    IOLayoutsSection(const std::vector<ov::Layout>& input_layouts, const std::vector<ov::Layout>& output_layouts);

    std::vector<ov::Layout> get_input_layouts() const;

    std::vector<ov::Layout> get_output_layouts() const;

    /**
     * @brief Writes all layouts into the provided sink using the following format:
     * <number of input layouts> <number of output layouts>
     * <input layout 1 size> <input layout 1 value> ... <input layout n size> <input layout n value>
     * <output layout 1 size> <output layout 1 value> ... <output layout n size> <output layout n value>
     */
    void encode(const ov::runtime::hsm::SectionSink& sink) const override;

    /**
     * @brief Parses all layouts to form a new @ref IOLayoutsSection. Expected format:
     * <number of input layouts> <number of output layouts>
     * <input layout 1 size> <input layout 1 value> ... <input layout n size> <input layout n value>
     * <output layout 1 size> <output layout 1 value> ... <output layout n size> <output layout n value>
     */
    static ov::runtime::hsm::SectionDecoder<std::shared_ptr<ISection>> decode;

private:
    std::vector<ov::Layout> m_input_layouts;
    std::vector<ov::Layout> m_output_layouts;

    Logger m_logger;
};

}  // namespace intel_npu
