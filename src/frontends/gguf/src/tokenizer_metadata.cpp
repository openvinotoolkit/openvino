// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/gguf/tokenizer_metadata.hpp"

namespace ov {
namespace frontend {
namespace gguf {

const std::string& gguf_tokenizer_metadata_key() {
    static const std::string key = "gguf_tokenizer_metadata";
    return key;
}

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
