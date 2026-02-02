// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/openvino.hpp"

namespace ov ::npuw ::util {

void prepare_text_embedding_model(std::shared_ptr<ov::Model> model, uint32_t seq_len_dim);

}  // namespace ov::npuw::util
