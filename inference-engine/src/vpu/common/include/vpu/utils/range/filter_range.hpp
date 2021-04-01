// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cassert>

#include <type_traits>
#include <iterator>
#include <utility>
#include <algorithm>

#include <vpu/utils/range/helpers.hpp>
#include <vpu/utils/checked_cast.hpp>

namespace vpu {

namespace details {

template <class BaseRange, class FilterOp>
class FilterRange final : private DebugRange<FilterRange<BaseRange, FilterOp>> {
public:
    static constexpr bool has_reverse_iter = HasReverseIterator<BaseRange>::value;
    static constexpr bool has_random_access = false;
    static constexpr bool const_time_size = false;

private:
    using BaseIterator = typename BaseRange::const_iterator;
    using BaseRevIterator = typename GetReverseIterator<BaseRange, has_reverse_iter>::reverse_iterator;
    using BaseIteratorValue = typename std::iterator_traits<BaseIterator>::value_type;

    template <typename BaseIterator, bool reverse>
    class Iterator final : private DebugIterator<FilterRange> {
    public:
        using value_type = BaseIteratorValue;

        using pointer = void;
        using reference = void;

        using iterator_category = std::input_iterator_tag;
        using difference_type = std::ptrdiff_t;

        Iterator() = default;

        value_type operator*() const {
            assert(this->range() != nullptr);
            assert(_cur != _end);

            return *_cur;
        }

        Iterator& operator++() {
            assert(this->range() != nullptr);

            if (_cur == _end) {
                this->reset();
            } else {
                ++_cur;
                postAdvance();
            }

            return *this;
        }

        bool operator==(const Iterator& other) const {
            if (this->range() != other.range()) {
                return false;
            }
            if (this->range() != nullptr) {
                return _cur == other._cur;
            }
            return true;
        }
        bool operator!=(const Iterator& other) const {
            return !(*this == other);
        }

    private:
        explicit Iterator(const FilterRange* range) :
                DebugIterator<FilterRange>(range) {
            _cur = IteratorAccess<reverse>::getBegin(this->range()->_base);
            _end = IteratorAccess<reverse>::getEnd(this->range()->_base);

            postAdvance();
        }

        void postAdvance() {
            assert(this->range() != nullptr);

            _cur = std::find_if(_cur, _end, this->range()->_op);

            if (_cur == _end) {
                this->reset();
            }
        }

    private:
        BaseIterator _cur;
        BaseIterator _end;

    private:
        friend FilterRange;
    };

    template <bool reverse>
    class Iterator<void, reverse> final {};

public:
    using value_type = typename BaseRange::value_type;
    using size_type = std::size_t;

    using iterator = Iterator<BaseIterator, false>;
    using reverse_iterator = typename std::conditional<has_reverse_iter, Iterator<BaseRevIterator, true>, void>::type;

    using const_iterator = iterator;
    using const_reverse_iterator = reverse_iterator;

    FilterRange() = default;

    FilterRange(BaseRange base, FilterOp op) :
            _base(std::move(base)), _op(std::move(op)) {
    }

    const_iterator begin() const {
        return const_iterator(this);
    }
    const_iterator end() const {
        return const_iterator();
    }

    template <typename Q = BaseRange, typename = typename std::enable_if<HasReverseIterator<Q>::value>::type>
    const_reverse_iterator rbegin() const {
        return const_reverse_iterator(this);
    }
    template <typename Q = BaseRange, typename = typename std::enable_if<HasReverseIterator<Q>::value>::type>
    const_reverse_iterator rend() const {
        return const_reverse_iterator();
    }

    value_type front() const {
        auto b = this->begin();
        assert(b != this->end());
        return *b;
    }
    template <typename Q = BaseRange, typename = typename std::enable_if<HasReverseIterator<Q>::value>::type>
    value_type back() const {
        auto b = this->rbegin();
        assert(b != this->rend());
        return *b;
    }

    size_type size() const {
        return checked_cast<size_type>(std::count_if(_base.begin(), _base.end(), _op));
    }

    bool empty() const {
        return size() == 0;
    }

private:
    BaseRange _base;
    FilterOp _op;

private:
    template <typename BaseIterator, bool reverse>
    friend class Iterator;

    friend DebugIterator<FilterRange>;
};

}  // namespace details

template <class BaseRange, class FilterOp>
details::FilterRange<BaseRange, FilterOp> filterRange(BaseRange base, FilterOp op) {
    return details::FilterRange<BaseRange, FilterOp>(std::move(base), std::move(op));
}
template <class FilterOp, class BaseRange>
details::FilterRange<BaseRange, FilterOp> filterRange(BaseRange base) {
    return details::FilterRange<BaseRange, FilterOp>(std::move(base), FilterOp());
}

namespace details {

template <class FilterOp>
struct FilterRangeTag final {
    FilterOp op;
};

template <class BaseRange, class FilterOp>
auto operator|(BaseRange base, FilterRangeTag<FilterOp>&& t) ->
        decltype(filterRange(std::move(base), std::move(t.op))) {
    return filterRange(std::move(base), std::move(t.op));
}

}  // namespace details

template <class FilterOp>
details::FilterRangeTag<FilterOp> filter(FilterOp op) {
    return details::FilterRangeTag<FilterOp>{std::move(op)};
}
template <class FilterOp>
details::FilterRangeTag<FilterOp> filter() {
    return details::FilterRangeTag<FilterOp>{};
}
template <class FilterOp, typename... Args>
details::FilterRangeTag<FilterOp> filter(Args&&... args) {
    return details::FilterRangeTag<FilterOp>{FilterOp{std::forward<Args>(args)...}};
}

struct NonNull final {
public:
    template <class Ptr>
    bool operator()(const Ptr& ptr) const {
        return ptr != nullptr;
    }
};

}  // namespace vpu
