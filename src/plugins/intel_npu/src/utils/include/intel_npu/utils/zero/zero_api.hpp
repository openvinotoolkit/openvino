// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#define ZERO_API_KEEP_SYMBOLS_LIST_MACRO
#include <openvino/zero_api.hpp>
#undef ZERO_API_KEEP_SYMBOLS_LIST_MACRO

namespace intel_npu {
    using ZeroApi = ::ov::ZeroApi;

#define symbol_statement(symbol) inline decltype(&::symbol) symbol = ::ov::symbol;
symbols_list();
weak_symbols_list();
#undef symbol_statement
}  // namespace intel_npu
