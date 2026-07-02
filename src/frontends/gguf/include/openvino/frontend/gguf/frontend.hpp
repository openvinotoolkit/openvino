// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/extension/telemetry.hpp"
#include "openvino/frontend/frontend.hpp"
#include "openvino/frontend/gguf/visibility.hpp"

namespace ov {
namespace frontend {
namespace gguf {

/// \brief OpenVINO frontend that loads a GGUF container directly into an ov::Model.
///
/// Unlike op-by-op translation frontends, this frontend builds the whole transformer graph for a
/// supported architecture. Quantized weights are kept as raw GGUF block bytes inside gguf_* typed
/// Constants and consumed by FullyConnectedCompressed nodes (no in-graph dequantization).
///
/// This release supports the "qwen3" architecture only; other architectures throw on convert().
class GGUF_FRONTEND_API FrontEnd : public ov::frontend::FrontEnd {
public:
    using Ptr = std::shared_ptr<FrontEnd>;
    FrontEnd() = default;

    /// \brief Build a complete ov::Model from the input GGUF model.
    std::shared_ptr<Model> convert(const ov::frontend::InputModel::Ptr& model) const override;

    std::string get_name() const override {
        return "gguf";
    }

    void add_extension(const std::shared_ptr<ov::Extension>& extension) override;

protected:
    bool supported_impl(const std::vector<ov::Any>& variants) const override;
    ov::frontend::InputModel::Ptr load_impl(const std::vector<ov::Any>& variants) const override;

    TelemetryExtension::Ptr m_telemetry;
};

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
