// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "gguf_reader.hpp"
#include "openvino/frontend/input_model.hpp"

namespace ov {
namespace frontend {
namespace gguf {

/// \brief InputModel for the GGUF frontend.
///
/// This is a "whole-model" frontend: the InputModel simply owns the parsed GGUFReader. The
/// architecture-specific graph builder consumes it during FrontEnd::convert(). The place/editing
/// API of the base class is intentionally not implemented.
class InputModel : public ov::frontend::InputModel {
public:
    explicit InputModel(const std::shared_ptr<GGUFReader>& reader) : m_reader(reader) {}

    const std::shared_ptr<GGUFReader>& reader() const {
        return m_reader;
    }

private:
    std::shared_ptr<GGUFReader> m_reader;
};

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
