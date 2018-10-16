// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#ifndef ALLOC_HPP
#define ALLOC_HPP

#include <cstddef> //size_t

namespace ade
{
void* aligned_alloc(std::size_t size, std::size_t alignment);
void aligned_free(void* ptr);

}

#endif // ALLOC_HPP
