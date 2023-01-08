// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cassert>

#include <vector>

#include <vpu/utils/small_vector.hpp>

namespace vpu {

namespace details {

template <class Container, typename T>
auto contAddImpl_(Container& cont, const T& val, int) -> decltype(cont.push_back(val)) {
    return cont.push_back(val);
}
template <class Container, typename T>
auto contAddImpl_(Container& cont, const T& val, ...) -> decltype(cont.insert(val)) {
    return cont.insert(val);
}
template <class Container, typename T>
auto contAddImpl(Container& cont, const T& val) -> decltype(contAddImpl_(cont, val, 0)) {
    return contAddImpl_(cont, val, 0);
}

template <bool const_time_size>
struct ToVectorHelper final {
    static void reserve(...) {}
};
template <>
struct ToVectorHelper<true> final {
    template <class Range, typename T>
    static void reserve(std::vector<T>& vec, const Range& range) {
        vec.reserve(range.size());
    }

    template <class Range, typename T, int Capacity>
    static void reserve(SmallVector<T, Capacity>& vec, const Range& range) {
        vec.reserve(range.size());
    }

    static void reserve(...) {}
};

template <class Container, class Range>
void fillContainer(const Range& range, Container& out) {
    if (out.empty()) {
        details::ToVectorHelper<Range::const_time_size>::reserve(out, range);
        for (const auto& item : range) {
            details::contAddImpl(out, item);
        }
    }
}

}  // namespace details

template <class Container, class Range>
Container toContainer(const Range& range) {
    Container out;
    details::fillContainer(range, out);
    return out;
}

template <class Container, class Range>
Container& toContainer(const Range& range, Container& out) {
    details::fillContainer(range, out);
    return out;
}

template <class Range>
std::vector<typename Range::value_type> toVector(const Range& range) {
    return toContainer<std::vector<typename Range::value_type>>(range);
}

template <int Capacity = 8, class Range>
SmallVector<typename Range::value_type, Capacity> toSmallVector(const Range& range) {
    return toContainer<SmallVector<typename Range::value_type, Capacity>>(range);
}

namespace details {

template <class Container>
struct ToContainerTag final {};

template <class Container>
struct ToRefContainerTag final {
    std::reference_wrapper<typename std::remove_reference<Container>::type> cont;
};

struct ToVectorTag final {};

template <int Capacity>
struct ToSmallVectorTag final {
};

template <class Container, class Range>
auto operator|(const Range& range, ToContainerTag<Container>&&) ->
        decltype(toContainer<Container>(range)) {
    return toContainer<Container>(range);
}

template <class Container, class Range>
auto operator|(const Range& range, ToRefContainerTag<Container>&& t) ->
        decltype(toContainer<Container>(range, t.cont.get())) {
    return toContainer<Container>(range, t.cont.get());
}

template <class Range>
auto operator|(const Range& range, ToVectorTag&&) ->
        decltype(toVector(range)) {
    return toVector(range);
}

template <int Capacity, class Range>
auto operator|(const Range& range, ToSmallVectorTag<Capacity>&&) ->
        decltype(toSmallVector<Capacity>(range)) {
    return toSmallVector<Capacity>(range);
}

}  // namespace details

template <class Container>
details::ToContainerTag<Container> asContainer() {
    return details::ToContainerTag<Container>{};
}

template <class Container>
details::ToRefContainerTag<Container> asContainer(Container& cont) {
    return details::ToRefContainerTag<Container>{cont};
}

inline details::ToVectorTag asVector() {
    return details::ToVectorTag{};
}

template <int Capacity = 8>
details::ToSmallVectorTag<Capacity> asSmallVector() {
    return details::ToSmallVectorTag<Capacity>{};
}

}  // namespace vpu
