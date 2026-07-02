// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "input_model.h"

#include "openvino/frontend/gguf/decoder.h"

namespace ov {
namespace frontend {
namespace gguf {

InputModel::InputModel(const std::shared_ptr<GgufDecoder>& gdecoder, bool naive)
    : m_decoder(gdecoder),
      m_naive(naive) {}

const std::shared_ptr<GgufDecoder>& InputModel::get_model_decoder() const {
    return m_decoder;
}

bool InputModel::is_naive() const {
    return m_naive;
}

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
