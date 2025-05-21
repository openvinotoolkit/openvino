// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "node.h"
#include "rdft.h"

namespace ov::intel_cpu::node {

class ISTFT : public Node {
public:
    ISTFT(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    bool created() const override;
    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;
    bool needPrepareParams() const override;
    void createPrimitive() override;

    void execute(const dnnl::stream& strm) override;
    void executeDynamicImpl(const dnnl::stream& strm) override;
    bool canBeInPlace() const override {
        return false;
    }

protected:
    bool needShapeInfer() const override;

private:
    /// ISTFT params
    bool m_center = false;
    bool m_normalized = false;

    // RDFT executor
    std::shared_ptr<RDFTExecutor> rdft_executor = nullptr;

    bool m_is_frame_size_const = false;
    bool m_is_frame_step_const = false;
    bool m_is_signal_length_const = false;
    bool m_has_signal_length_input = false;

    // Input indices
    static constexpr size_t DATA_IDX = 0lu;
    static constexpr size_t WINDOW_IDX = 1lu;
    static constexpr size_t FRAME_SIZE_IDX = 2lu;
    static constexpr size_t FRAME_STEP_IDX = 3lu;
    static constexpr size_t SIGNAL_LENGTH_IDX = 4lu;
};

}  // namespace ov::intel_cpu::node
