// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cassert>

#include <type_traits>
#include <iterator>
#include <utility>
#include <vector>
#include <memory>
#include <functional>
#include <unordered_set>

#include <vpu/utils/containers.hpp>
#include <vpu/utils/handle.hpp>
#include <vpu/utils/optional.hpp>

namespace vpu {

//
// IterRange
//

namespace impl {

template <class Iterator>
class IterRange final {
public:
    using value_type = typename Iterator::value_type;

    using iterator = Iterator;
    using const_iterator = Iterator;

    inline IterRange() = default;
    inline IterRange(const IterRange&) = default;
    inline IterRange& operator=(const IterRange&) = default;
    inline IterRange(IterRange&&) = default;
    inline IterRange& operator=(IterRange&&) = default;

    template <class It1, class It2>
    inline IterRange(It1&& b, It2&& e) : _begin(std::forward<It1>(b)), _end(std::forward<It2>(e)) {}

    inline Iterator begin() const { return _begin; }
    inline Iterator end() const { return _end; }

    Iterator cbegin() const { return _begin; }
    Iterator cend() const { return _end; }

private:
    Iterator _begin;
    Iterator _end;
};

}  // namespace impl

template <class Iterator>
inline impl::IterRange<Iterator> iterRange(const Iterator& begin, const Iterator& end) {
    return impl::IterRange<Iterator>(begin, end);
}
template <class Iterator>
inline impl::IterRange<Iterator> iterRange(Iterator&& begin, Iterator&& end) {
    return impl::IterRange<Iterator>(std::move(begin), std::move(end));
}

//
// ContRange
//

namespace impl {

template <class Cont>
class ContRange final {
public:
    using value_type = typename Cont::value_type;

    using iterator = typename Cont::iterator;
    using const_iterator = typename Cont::const_iterator;

    inline ContRange() = default;
    inline ContRange(const ContRange&) = default;
    inline ContRange& operator=(const ContRange&) = default;
    inline ContRange(ContRange&&) = default;
    inline ContRange& operator=(ContRange&&) = default;

    inline explicit ContRange(const Cont& cont) : _cont(&cont) {}

    inline const_iterator begin() const {
        assert(_cont != nullptr);
        return _cont->begin();
    }
    inline const_iterator end() const {
        assert(_cont != nullptr);
        return _cont->end();
    }

    inline const_iterator cbegin() const {
        assert(_cont != nullptr);
        return _cont->begin();
    }
    inline const_iterator cend() const {
        assert(_cont != nullptr);
        return _cont->end();
    }

private:
    const Cont* _cont = nullptr;
};

}  // namespace impl

template <class Cont>
inline impl::ContRange<Cont> contRange(const Cont& cont) {
    return impl::ContRange<Cont>(cont);
}

//
// MapRange
//

namespace impl {

template <class BaseRange, class MapOp>
class MapRange final {
public:
    class Iterator final {
    public:
        using base_iterator = typename BaseRange::const_iterator;
        using base_iterator_value = decltype(*base_iterator());

        using map_op_value = typename std::result_of<MapOp(base_iterator_value)>::type;

        using value_type = typename std::decay<map_op_value>::type;
        using pointer = value_type*;
        using reference = value_type&;

        using iterator_category = std::input_iterator_tag;
        using difference_type = std::ptrdiff_t;

        inline Iterator() = default;
        inline Iterator(const Iterator&) = default;
        inline Iterator& operator=(const Iterator&) = default;
        inline Iterator(Iterator&&) = default;
        inline Iterator& operator=(Iterator&&) = default;

        template <class BI>
        inline Iterator(BI&& cur, BI&& end, const MapOp& op) :
                _cur(std::forward<BI>(cur)),
                _end(std::forward<BI>(end)),
                _op(&op) {
        }

        inline const value_type& operator*() const {
            assert(_cur != _end);
            _curValue = (*_op)(*_cur);
            return _curValue.get();
        }

        inline Iterator& operator++() {
            ++_cur;
            return *this;
        }

        inline bool operator==(const Iterator& other) const {
            return _cur == other._cur;
        }
        inline bool operator!=(const Iterator& other) const {
            return _cur != other._cur;
        }

    private:
        base_iterator _cur;
        base_iterator _end;
        const MapOp* _op = nullptr;

        mutable Optional<value_type> _curValue;
    };

    using base_iterator = typename BaseRange::const_iterator;
    using base_iterator_value = decltype(*base_iterator());

    using map_op_value = typename std::result_of<MapOp(base_iterator_value)>::type;

    using value_type = typename std::decay<map_op_value>::type;

    using iterator = Iterator;
    using const_iterator = Iterator;

    inline MapRange() = default;
    inline MapRange(const MapRange&) = default;
    inline MapRange& operator=(const MapRange&) = default;
    inline MapRange(MapRange&&) = default;
    inline MapRange& operator=(MapRange&&) = default;

    template <class _B, class _M>
    inline MapRange(_B&& base, _M&& op) :
            _base(std::forward<_B>(base)),
            _op(std::forward<_M>(op)) {
    }

    inline Iterator begin() const { return Iterator(_base.begin(), _base.end(), _op); }
    inline Iterator end() const { return Iterator(_base.end(), _base.end(), _op); }

    inline Iterator cbegin() const { return Iterator(_base.begin(), _base.end(), _op); }
    inline Iterator cend() const { return Iterator(_base.end(), _base.end(), _op); }

private:
    BaseRange _base;
    MapOp _op;
};

}  // namespace impl

template <class BaseRange, class MapOp>
inline impl::MapRange<typename std::decay<BaseRange>::type, typename std::decay<MapOp>::type>
        mapRange(BaseRange&& base, MapOp&& op) {
    return impl::MapRange<typename std::decay<BaseRange>::type, typename std::decay<MapOp>::type>(
        std::forward<typename std::remove_reference<BaseRange>::type>(base),
        std::forward<typename std::remove_reference<MapOp>::type>(op));
}
template <class MapOp, class BaseRange>
inline impl::MapRange<typename std::decay<BaseRange>::type, MapOp> mapRange(BaseRange&& base) {
    return impl::MapRange<typename std::remove_reference<BaseRange>::type, MapOp>(
        std::forward<typename std::remove_reference<BaseRange>::type>(base),
        MapOp());
}

//
// FilterRange
//

namespace impl {

template <class BaseRange, class FilterOp>
class FilterRange final {
public:
    class Iterator final {
    public:
        using base_iterator = typename BaseRange::const_iterator;
        using base_iterator_value = decltype(*base_iterator());

        using value_type = typename std::decay<base_iterator_value>::type;
        using pointer = value_type*;
        using reference = value_type&;

        using iterator_category = std::input_iterator_tag;
        using difference_type = std::ptrdiff_t;

        inline Iterator() = default;
        inline Iterator(const Iterator&) = default;
        inline Iterator& operator=(const Iterator&) = default;
        inline Iterator(Iterator&&) = default;
        inline Iterator& operator=(Iterator&&) = default;

        template <class BI>
        inline Iterator(BI&& cur, BI&& end, const FilterOp& op) :
                _cur(std::forward<BI>(cur)),
                _end(std::forward<BI>(end)),
                _op(&op) {
            advance();
        }

        inline const value_type& operator*() const {
            assert(_cur != _end);
            _curValue = *_cur;
            return _curValue.get();
        }

        inline Iterator& operator++() {
            ++_cur;
            advance();
            return *this;
        }

        inline bool operator==(const Iterator& other) const {
            return _cur == other._cur;
        }
        inline bool operator!=(const Iterator& other) const {
            return _cur != other._cur;
        }

    private:
        inline void advance() {
            while (_cur != _end) {
                _curValue = *_cur;
                if ((*_op)(_curValue.get())) {
                    break;
                }
                ++_cur;
            }
        }

    private:
        base_iterator _cur;
        base_iterator _end;
        const FilterOp* _op = nullptr;

        mutable Optional<value_type> _curValue;
    };

    using value_type = typename BaseRange::value_type;

    using iterator = Iterator;
    using const_iterator = Iterator;

    inline FilterRange() = default;
    inline FilterRange(const FilterRange&) = default;
    inline FilterRange& operator=(const FilterRange&) = default;
    inline FilterRange(FilterRange&&) = default;
    inline FilterRange& operator=(FilterRange&&) = default;

    template <class _B, class _F>
    inline FilterRange(_B&& base, _F&& op) :
            _base(std::forward<_B>(base)),
            _op(std::forward<_F>(op)) {
    }

    inline Iterator begin() const { return Iterator(_base.begin(), _base.end(), _op); }
    inline Iterator end() const { return Iterator(_base.end(), _base.end(), _op); }

    inline Iterator cbegin() const { return Iterator(_base.begin(), _base.end(), _op); }
    inline Iterator cend() const { return Iterator(_base.end(), _base.end(), _op); }

private:
    BaseRange _base;
    FilterOp _op;
};

}  // namespace impl

template <class BaseRange, class FilterOp>
inline impl::FilterRange<typename std::decay<BaseRange>::type, typename std::decay<FilterOp>::type>
        filterRange(BaseRange&& base, FilterOp&& op) {
    return impl::
    FilterRange<typename std::decay<BaseRange>::type, typename std::decay<FilterOp>::type>(
        std::forward<typename std::remove_reference<BaseRange>::type>(base),
        std::forward<typename std::remove_reference<FilterOp>::type>(op));
}
template <class FilterOp, class BaseRange>
inline impl::FilterRange<typename std::decay<BaseRange>::type, FilterOp> filterRange(BaseRange&& base) {
    return impl::FilterRange<typename std::decay<BaseRange>::type, FilterOp>(
        std::forward<typename std::remove_reference<BaseRange>::type>(base),
        FilterOp());
}

struct NonNull final {
public:
    template <class Ptr>
    inline bool operator()(const Ptr& ptr) const {
        return ptr != nullptr;
    }
};

struct PtrToHandle final {
    template <typename T>
    inline Handle<T> operator()(const std::shared_ptr<T>& ptr) const {
        return Handle<T>(ptr);
    }
};

//
// toVector
//

template <class Range>
inline std::vector<typename std::decay<typename Range::value_type>::type> toVector(const Range& range, int capacity = 0) {
    std::vector<typename std::decay<typename Range::value_type>::type> out;
    if (capacity > 0) {
        out.reserve(capacity);
    }
    for (const auto& item : range) {
        out.emplace_back(item);
    }
    return out;
}

template <int Capacity, class Range>
inline SmallVector<typename std::decay<typename Range::value_type>::type, Capacity> toSmallVector(const Range& range) {
    SmallVector<typename std::decay<typename Range::value_type>::type, Capacity> out;
    for (const auto& item : range) {
        out.emplace_back(item);
    }
    return out;
}

}  // namespace vpu
