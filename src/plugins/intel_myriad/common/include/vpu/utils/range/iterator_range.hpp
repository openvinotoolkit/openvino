// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cassert>

#include <type_traits>
#include <iterator>
#include <utility>

#include <vpu/utils/checked_cast.hpp>

namespace vpu {

namespace details {

template <class Iterator>
class IteratorRange final {
public:
    static constexpr bool has_reverse_iter = true;
    static constexpr bool has_random_access = false;
    static constexpr bool const_time_size =
            std::is_same<typename std::iterator_traits<Iterator>::iterator_category, std::random_access_iterator_tag>::value;

    using value_type = typename std::iterator_traits<Iterator>::value_type;
    using size_type = std::size_t;

    using iterator = Iterator;
    using reverse_iterator = std::reverse_iterator<Iterator>;

    using const_iterator = iterator;
    using const_reverse_iterator = reverse_iterator;

    IteratorRange() = default;
    IteratorRange(Iterator begin, Iterator end) : _begin(std::move(begin)), _end(std::move(end)) {}

    const_iterator begin() const { return _begin; }
    const_iterator end() const { return _end; }

    const_reverse_iterator rbegin() const { return const_reverse_iterator(_begin); }
    const_reverse_iterator rend() const { return const_reverse_iterator(_end); }

    auto front() const -> decltype(*std::declval<iterator>()) {
        assert(_begin != _end);
        return *_begin;
    }
    auto back() const -> decltype(*std::declval<const_reverse_iterator>()) {
        assert(_begin != _end);
        return *rbegin();
    }

    size_type size() const {
        return checked_cast<size_type>(std::distance(_begin, _end));
    }

    bool empty() const {
        return _begin == _end;
    }

private:
    Iterator _begin;
    Iterator _end;
};

}  // namespace details

template <class Iterator>
details::IteratorRange<Iterator> iteratorRange(Iterator begin, Iterator end) {
    return details::IteratorRange<Iterator>(std::move(begin), std::move(end));
}

}  // namespace vpu
