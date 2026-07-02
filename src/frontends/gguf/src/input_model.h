// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/frontend/input_model.hpp>

#include "openvino/frontend/gguf/decoder.h"
#include "openvino/frontend/gguf/visibility.hpp"

namespace ov {
namespace frontend {
namespace gguf {

class FrontEnd;

class GGUF_FRONTEND_API InputModel : public ov::frontend::InputModel {
    friend class ::ov::frontend::gguf::FrontEnd;

public:
    explicit InputModel(const std::shared_ptr<GgufDecoder>& gdecoder, bool naive = false);

    const std::shared_ptr<GgufDecoder>& get_model_decoder() const;

    bool is_naive() const;

private:
    std::shared_ptr<GgufDecoder> m_decoder;
    bool m_naive;
};

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
