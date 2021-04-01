// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cassert>

#include <type_traits>
#include <iterator>
#include <utility>

#include <vpu/utils/range/helpers.hpp>
#include <vpu/utils/handle.hpp>
#include <vpu/utils/checked_cast.hpp>

namespace vpu {

namespace details {

template <class BaseRange, class MapOp>
class MapRange final : private DebugRange<MapRange<BaseRange, MapOp>> {
public:
    static constexpr bool has_reverse_iter = HasReverseIterator<BaseRange>::value;
    static constexpr bool has_random_access = HasRandomAccess<BaseRange>::value;
    static constexpr bool const_time_size = BaseRange::const_time_size;

private:
    using BaseIterator = typename BaseRange::const_iterator;
    using BaseRevIterator = typename GetReverseIterator<BaseRange, has_reverse_iter>::reverse_iterator;
    using BaseIteratorValue = decltype(*std::declval<BaseIterator>());
    using MapOpResult = decltype(std::declval<MapOp>()(std::declval<BaseIteratorValue>()));
    using MapOpValue = typename std::decay<MapOpResult>::type;

    template <typename BaseIterator, bool reverse>
    class Iterator final : private DebugIterator<MapRange> {
    public:
        using value_type = MapOpValue;

        using pointer = void;
        using reference = void;

        using iterator_category = std::input_iterator_tag;
        using difference_type = std::ptrdiff_t;

        Iterator() = default;

        value_type operator*() const {
            assert(this->range() != nullptr);
            assert(_cur != _end);

            return (this->range()->_op)(*_cur);
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
        explicit Iterator(const MapRange* range) :
                DebugIterator<MapRange>(range) {
            _cur = IteratorAccess<reverse>::getBegin(this->range()->_base);
            _end = IteratorAccess<reverse>::getEnd(this->range()->_base);

            postAdvance();
        }

        void postAdvance() {
            assert(this->range() != nullptr);

            if (_cur == _end) {
                this->reset();
            }
        }

    private:
        BaseIterator _cur;
        BaseIterator _end;

    private:
        friend MapRange;
    };

    template <bool reverse>
    class Iterator<void, reverse> final {};

public:
    using value_type = MapOpValue;
    using size_type = std::size_t;

    using iterator = Iterator<BaseIterator, false>;
    using reverse_iterator = typename std::conditional<has_reverse_iter, Iterator<BaseRevIterator, true>, void>::type;

    using const_iterator = iterator;
    using const_reverse_iterator = reverse_iterator;

    MapRange() = default;

    MapRange(BaseRange base, MapOp op) :
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
        return _op(_base.front());
    }
    template <typename Q = BaseRange, typename = typename std::enable_if<HasReverseIterator<Q>::value>::type>
    value_type back() const {
        return _op(_base.back());
    }

    size_type size() const {
        return checked_cast<size_type>(_base.size());
    }

    bool empty() const {
        return _base.empty();
    }

    template <typename Q = BaseRange, typename = typename std::enable_if<HasRandomAccess<Q>::value>::type>
    value_type operator[](int ind) const {
        return _op(_base[ind]);
    }

private:
    BaseRange _base;
    MapOp _op;

private:
    template <typename BaseIterator, bool reverse>
    friend class Iterator;

    friend DebugIterator<MapRange>;
};

}  // namespace details

template <class BaseRange, class MapOp>
details::MapRange<BaseRange, MapOp> mapRange(BaseRange base, MapOp op) {
    return details::MapRange<BaseRange, MapOp>(std::move(base), std::move(op));
}
template <class MapOp, class BaseRange>
details::MapRange<BaseRange, MapOp> mapRange(BaseRange base) {
    return details::MapRange<BaseRange, MapOp>(std::move(base), MapOp());
}

namespace details {

template <class MapOp>
struct MapRangeTag final {
    MapOp op;
};

template <class BaseRange, class MapOp>
auto operator|(BaseRange base, MapRangeTag<MapOp>&& t) ->
        decltype(mapRange(std::move(base), std::move(t.op))) {
    return mapRange(std::move(base), std::move(t.op));
}

}  // namespace details

template <class MapOp>
details::MapRangeTag<MapOp> map(MapOp op) {
    return details::MapRangeTag<MapOp>{std::move(op)};
}
template <class MapOp>
details::MapRangeTag<MapOp> map() {
    return details::MapRangeTag<MapOp>{};
}
template <class MapOp, typename... Args>
details::MapRangeTag<MapOp> map(Args&&... args) {
    return details::MapRangeTag<MapOp>{MapOp{std::forward<Args>(args)...}};
}

}  // namespace vpu
