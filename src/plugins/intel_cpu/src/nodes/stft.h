// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "node.h"

namespace ov {
namespace intel_cpu {
namespace node {

class STFT : public Node {
public:
    STFT(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    bool created() const override;
    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;
    bool needPrepareParams() const override;

    void execute(dnnl::stream strm) override;
    void executeDynamicImpl(dnnl::stream strm) override;
    bool canBeInPlace() const override {
        return false;
    }

protected:
    bool needShapeInfer() const override;

private:
    /// STFT params
    bool m_transpose_frames = false;

    bool m_is_frame_size_const = false;
    bool m_is_frame_step_const = false;

    // Input indices
    static constexpr size_t DATA_IDX = 0lu;
    static constexpr size_t WINDOW_IDX = 1lu;
    static constexpr size_t FRAME_SIZE_IDX = 2lu;
    static constexpr size_t FRAME_STEP_IDX = 3lu;
};

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
