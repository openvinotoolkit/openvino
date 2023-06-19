// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <vector>
#include "llm_types.hpp"
#include "llm_emb_gpt.hpp"

namespace llmdnn {

std::shared_ptr<emb_gpt::impl> new_impl_avx512();

}
