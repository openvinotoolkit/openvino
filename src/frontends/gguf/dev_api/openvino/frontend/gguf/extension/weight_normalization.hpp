// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <utility>

#include "openvino/core/extension.hpp"
#include "openvino/pass/manager.hpp"

namespace ov::frontend::gguf {

/// Overrides weight normalization on one frontend instance. The callback must register
/// the complete replacement pipeline; ordinary conversion uses the built-in materializer.
class WeightNormalizationExtension : public ov::Extension {
public:
    OPENVINO_RTTI("gguf::WeightNormalizationExtension");

    explicit WeightNormalizationExtension(std::function<void(ov::pass::Manager&)> register_passes)
        : m_register_passes(std::move(register_passes)) {
        OPENVINO_ASSERT(m_register_passes, "Weight normalization extension requires a callback");
    }

    void register_passes(ov::pass::Manager& manager) const {
        m_register_passes(manager);
    }

private:
    std::function<void(ov::pass::Manager&)> m_register_passes;
};

}  // namespace ov::frontend::gguf
