// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MEMORY_TYPES_HPP
#define MEMORY_TYPES_HPP

#include "util/md_size.hpp"
#include "util/md_span.hpp"
#include "util/md_view.hpp"

namespace ade
{
namespace memory
{
static const constexpr std::size_t MaxDimensions = 6;
using DynMdSize = util::DynMdSize<MaxDimensions>;
using DynMdSpan = util::DynMdSpan<MaxDimensions>;

template<typename T>
using DynMdView = util::DynMdView<MaxDimensions, T>;

}
}

#endif // MEMORY_TYPES_HPP
