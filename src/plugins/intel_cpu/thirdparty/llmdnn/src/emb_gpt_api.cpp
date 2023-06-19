// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>

#include "emb_gpt_avx512.hpp"

namespace llmdnn {

// interface
emb_gpt::emb_gpt(): _impl(new_impl_avx512()) {
}

bool emb_gpt::create(const create_param& param) {
    return _impl->create(param);
}

void emb_gpt::exec(const exec_param& param) {
    _impl->exec(param);
}

}