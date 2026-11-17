// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

namespace ov {
class Model;
}

namespace ov::frontend::gguf {

void fold_hadamard_input_signs(const std::shared_ptr<ov::Model>& model);

}  // namespace ov::frontend::gguf
