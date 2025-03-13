// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
namespace ov::util {

template <class... Ts>
struct VariantVisitor : Ts... {
    using Ts::operator()...;
};

template <class... Ts>
VariantVisitor(Ts...) -> VariantVisitor<Ts...>;
}  // namespace ov::util
