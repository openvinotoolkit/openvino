// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <initializer_list>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "except.hpp"

#include <string_view>

namespace ov::util {

template <typename Container>
struct Joined {
    const Container& c;
    std::string_view sep;

    friend std::ostream& operator<<(std::ostream& os, const Joined& jv) {
        auto first = std::begin(jv.c);
        const auto last = std::end(jv.c);
        if (first != last) {
            os << *first;
            for (++first; first != last; ++first)
                os << jv.sep << *first;
        }
        return os;
    }

    operator std::string() const {
        std::ostringstream ss;
        ss << *this;
        return ss.str();
    }
};

template <class R = std::string, typename Container>
auto join(const Container& c, std::string_view sep = ", ") {
    if constexpr (std::is_same_v<R, std::string>) {
        return static_cast<std::string>(Joined<Container>{c, sep});
    } else if constexpr (std::is_same_v<R, std::ostream>) {
        return Joined<Container>{c, sep};
    }
}

}  // namespace ov::util

namespace ov::intel_gpu {
namespace detail {
template <bool do_move, typename T, typename U, std::size_t N, std::size_t... I>
[[nodiscard]] constexpr std::array<std::remove_cv_t<T>, N> to_array_impl(U (&values)[N], std::index_sequence<I...> /*unused*/) noexcept {
    if constexpr (do_move) {
        return {{static_cast<T>(std::move(values[I]))...}};
    }
    return {{ static_cast<T>(values[I])...}};
}
}  // namespace detail

template <typename T, typename U, std::size_t N, std::enable_if_t<std::is_convertible_v<T, U>, bool> = true>
[[nodiscard]] constexpr std::array<std::remove_cv_t<T>, N> to_array(U (&values)[N]) noexcept {
    static_assert(N > 0, "[GPU] An array must not be empty");
    return detail::to_array_impl<false, T>(values, std::make_index_sequence<N>());
}
template <typename T, typename U, std::size_t N, std::enable_if_t<std::is_convertible_v<T, U>, bool> = true>
[[nodiscard]] constexpr std::array<std::remove_cv_t<T>, N> to_array(U (&&values)[N]) noexcept {
    static_assert(N > 0, "[GPU] An array must not be empty");
    return detail::to_array_impl<true, T>(values, std::make_index_sequence<N>());
}
}  // namespace ov::intel_gpu