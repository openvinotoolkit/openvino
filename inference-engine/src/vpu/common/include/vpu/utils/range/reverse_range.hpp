// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cassert>

#include <type_traits>
#include <iterator>
#include <utility>

#include <vpu/utils/range/helpers.hpp>
#include <vpu/utils/checked_cast.hpp>

namespace vpu {

namespace details {

template <class BaseRange>
class ReverseRange final {
public:
    static constexpr bool has_reverse_iter = true;
    static constexpr bool has_random_access = HasRandomAccess<BaseRange>::value;
    static constexpr bool const_time_size = BaseRange::const_time_size;

    using value_type = typename BaseRange::value_type;
    using size_type = std::size_t;

    using iterator = typename BaseRange::const_reverse_iterator;
    using reverse_iterator = typename BaseRange::const_iterator;

    using const_iterator = iterator;
    using const_reverse_iterator = reverse_iterator;

    ReverseRange() = default;

    explicit ReverseRange(BaseRange base) :
            _base(std::move(base)) {
    }

    const_iterator begin() const { return _base.rbegin(); }
    const_iterator end() const { return _base.rend(); }

    const_reverse_iterator rbegin() const { return _base.begin(); }
    const_reverse_iterator rend() const { return _base.end(); }

    auto front() const -> decltype(std::declval<BaseRange>().back()) {
        return _base.back();
    }
    auto back() const -> decltype(std::declval<BaseRange>().front()) {
        return _base.front();
    }

    size_type size() const {
        return checked_cast<size_type>(_base.size());
    }

    bool empty() const {
        return _base.empty();
    }

    template <typename Q = BaseRange, typename = typename std::enable_if<HasRandomAccess<Q>::value>::type>
    auto operator[](int ind) const ->
            decltype(std::declval<const Q&>().operator[](ind)) {
        assert(ind >= 0 && ind < _base.size());
        return _base[_base.size() - ind - 1];
    }

private:
    BaseRange _base;
};

}  // namespace details

template <class BaseRange>
details::ReverseRange<BaseRange> reverseRange(BaseRange base) {
    return details::ReverseRange<BaseRange>(std::move(base));
}

namespace details {

struct ReverseRangeTag final {};

template <class BaseRange>
ReverseRange<BaseRange>
        operator|(BaseRange base, ReverseRangeTag&&) {
    return reverseRange(std::move(base));
}

}  // namespace details

inline details::ReverseRangeTag reverse() {
    return details::ReverseRangeTag{};
}

}  // namespace vpu
