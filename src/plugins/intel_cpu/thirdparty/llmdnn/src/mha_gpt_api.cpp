// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>

#include "mha_gpt_amx.hpp"

namespace llmdnn {

// interface
mha_gpt::mha_gpt(): _impl(new_impl_amx()) {
}

bool mha_gpt::create(const create_param& param) {
    return _impl->create(param);
}

void mha_gpt::exec(const exec_param& param) {
    _impl->exec(param);
}

}