// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cassert>

#include <iterator>
#include <utility>

#include <vpu/utils/range/helpers.hpp>

namespace vpu {

namespace details {

template <typename T>
class SingleElementRange final : private DebugRange<SingleElementRange<T>> {
public:
    static constexpr bool has_reverse_iter = true;
    static constexpr bool has_random_access = true;
    static constexpr bool const_time_size = true;

private:
    class Iterator final : private DebugIterator<SingleElementRange> {
    public:
        using value_type = T;

        using pointer = value_type*;
        using reference = value_type&;

        using iterator_category = std::input_iterator_tag;
        using difference_type = std::ptrdiff_t;

        Iterator() = default;

        const T& operator*()  const {
            assert(this->range() != nullptr);

            return this->range()->_elem;
        }

        Iterator& operator++() {
            assert(this->range() != nullptr);

            this->reset();

            return *this;
        }

        bool operator==(const Iterator& other) const {
            return this->range() == other.range();
        }

        bool operator!=(const Iterator& other) const {
            return !(*this == other);
        }

    private:
        using DebugIterator<SingleElementRange>::DebugIterator;

    private:
        friend SingleElementRange;
    };

public:
    using value_type = T;
    using size_type = std::size_t;

    using iterator = Iterator;
    using reverse_iterator = Iterator;

    using const_iterator = iterator;
    using const_reverse_iterator = reverse_iterator;

    SingleElementRange() = default;
    explicit SingleElementRange(T elem) : _elem(std::move(elem)) {}

    const_iterator begin() const { return Iterator(this); }
    const_iterator end() const { return Iterator(); }

    const_reverse_iterator rbegin() const { return Iterator(this); }
    const_reverse_iterator rend() const { return Iterator(); }

    const T& front() const {
        return _elem;
    }
    const T& back() const {
        return _elem;
    }

    size_type size() const {
        return 1;
    }

    bool empty() const {
        return false;
    }

    const T& operator[](int ind) {
        (void)ind;
        assert(ind == 0);
        return _elem;
    }

private:
    T _elem;

private:
    friend Iterator;

    friend DebugIterator<SingleElementRange>;
};

}  // namespace details

template <typename T>
details::SingleElementRange<T> singleElementRange(T elem) {
    return details::SingleElementRange<T>(std::move(elem));
}

namespace details {

struct SingleElementRangeTag final {};

template <typename T>
auto operator|(T elem, SingleElementRangeTag&&) ->
        decltype(singleElementRange(std::move(elem))) {
    return singleElementRange(std::move(elem));
}

}  // namespace details

inline details::SingleElementRangeTag asSingleElementRange() {
    return details::SingleElementRangeTag{};
}

}  // namespace vpu
