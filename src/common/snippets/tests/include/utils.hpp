// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/pass/tokenization.hpp"

namespace ov {
namespace test {
namespace snippets {

static ov::snippets::pass::SnippetsTokenization::Config get_default_tokenization_config() {
    return { 1, std::numeric_limits<size_t>::max(), true, true, true, { 3, 4 }};
}

}  // namespace snippets
}  // namespace test
}  // namespace ov
