// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

namespace GNAPluginNS {
namespace memory {

template<class T>
class IPolymorphAllocator {
public:
    virtual T *allocate(std::size_t n)  = 0;
    virtual void deallocate(T *p, std::size_t n)  = 0;
};
}  // namespace memory
}  // namespace GNAPluginNS
