// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace cldnn {
template <typename BufferType, typename T, typename Enable = void>
class Serializer {
public:
    static void save(BufferType& buffer, const T& object) {
        object.save(buffer);
    }

    static void load(BufferType& buffer, T& object) {
        object.load(buffer);
    }
};
}  // namespace cldnn
