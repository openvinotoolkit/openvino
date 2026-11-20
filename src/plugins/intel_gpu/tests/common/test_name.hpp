// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <utility>

namespace ov::test::utils {

template <typename NameGenerator>
auto make_hyphen_free_test_name(NameGenerator generator) {
    return [generator = std::move(generator)](const auto& info) mutable {
        auto name = generator(info);
        // GoogleTest interprets a hyphen in a filter as the start of exclusions.
        std::replace(name.begin(), name.end(), '-', '_');
        return name;
    };
}

}  // namespace ov::test::utils
