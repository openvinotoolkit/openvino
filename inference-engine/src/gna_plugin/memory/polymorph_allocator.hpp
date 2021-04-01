// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include "ipolymorph_allocator.hpp"

namespace GNAPluginNS {
namespace memory {
/**
 * @brief c++17 concept simulation
 */

template<class T>
class PolymorphAllocator {
    std::shared_ptr<IPolymorphAllocator<T>> _impl;
 public:
    explicit PolymorphAllocator(const std::shared_ptr<IPolymorphAllocator<T>> &impl) : _impl(impl) {}

    T *allocate(std::size_t n) {
        return _impl->allocate(n);
    }

    void deallocate(T *p, std::size_t n) {
        _impl->deallocate(p, n);
    }
};

/**
 * transform any allocator into polymorph type
 * @tparam origin
 */

template<class origin>
class PolymorphAdapter : public IPolymorphAllocator<typename origin::value_type> {
    origin _impl;
    using T = typename origin::value_type;

 public:
    template<class ...Args>
    explicit PolymorphAdapter(Args &&... args)
        :_impl(std::forward<Args>(args)...) {
    }
    T *allocate(std::size_t n) override {
        return _impl.allocate(n);
    }
    void deallocate(T *p, std::size_t n) override {
        _impl.deallocate(p, n);
    }
};

template<class T, class ...Args>
inline PolymorphAllocator<typename T::value_type> make_polymorph(Args &&... args) {
    auto sp = std::make_shared<PolymorphAdapter<T>>(std::forward<Args>(args)...);
    auto ipoly = std::static_pointer_cast<IPolymorphAllocator<typename T::value_type>>(sp);

    return PolymorphAllocator<typename T::value_type>(ipoly);
}
}  // namespace memory
}  // namespace GNAPluginNS
