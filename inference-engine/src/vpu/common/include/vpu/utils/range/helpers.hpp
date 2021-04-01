// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cassert>

#include <type_traits>
#include <unordered_set>

namespace vpu {

namespace details {

template <class Container>
class HasReverseIterator final {
    template <class Q1, class Q2 = decltype(std::declval<Q1>().rbegin())>
    static std::true_type check(const Q1& cont, int);

    template <class Q1>
    static std::false_type check(const Q1& cont, ...);

public:
    using type = decltype(check(std::declval<Container>(), 0));
    static constexpr bool value = type::value;
};

template <class Container, bool has_reverse_iter>
struct GetReverseIterator final {
    using reverse_iterator = typename Container::const_reverse_iterator;
};
template <class Container>
struct GetReverseIterator<Container, false> final {
    using reverse_iterator = void;
};

template <class Container>
class HasRandomAccess final {
    template <class Q1, class Q2 = decltype(std::declval<Q1>().operator[](0))>
    static std::true_type check(const Q1& cont, int);

    template <class Q1>
    static std::false_type check(const Q1& cont, ...);

public:
    using type = decltype(check(std::declval<Container>(), 0));
    static constexpr bool value = type::value;
};

template <class Range>
class DebugIterator;

template <class Range>
class DebugRange {
#ifndef NDEBUG
protected:
    ~DebugRange() {
        for (const auto& iter : _iterators) {
            iter->_range = nullptr;
        }
    }

private:
    mutable std::unordered_set<DebugIterator<Range>*> _iterators;

    friend DebugIterator<Range>;
#endif
};

template <class Range>
class DebugIterator {
protected:
    DebugIterator() = default;

    explicit DebugIterator(const Range* range) :
            _range(range) {
#ifndef NDEBUG
        assert(_range != nullptr);

        assert(_range->_iterators.count(this) == 0);
        _range->_iterators.insert(this);
#endif
    }

#ifndef NDEBUG
    ~DebugIterator() {
        reset();
    }

    DebugIterator(const DebugIterator& other) : _range(other._range) {
        if (_range != nullptr) {
            assert(_range->_iterators.count(this) == 0);
            _range->_iterators.insert(this);
        }
    }

    DebugIterator& operator=(const DebugIterator& other) {
        if (&other != this) {
            if (_range != nullptr) {
                assert(_range->_iterators.count(this) != 0);
                _range->_iterators.erase(this);
            }

            _range = other._range;

            if (_range != nullptr) {
                assert(_range->_iterators.count(this) == 0);
                _range->_iterators.insert(this);
            }
        }
        return *this;
    }
#endif

    const Range* range() const {
        return static_cast<const Range*>(_range);
    }

    void reset() {
#ifndef NDEBUG
        if (_range != nullptr) {
            assert(_range->_iterators.count(this) != 0);
            _range->_iterators.erase(this);
        }
#endif

        _range = nullptr;
    }

private:
    const DebugRange<Range>* _range = nullptr;

private:
    friend DebugRange<Range>;
};

template <bool reverse> struct IteratorAccess;
template <> struct IteratorAccess<false> {
    template <typename Range>
    static auto getBegin(const Range& r) ->
            decltype(r.begin()) {
        return r.begin();
    }

    template <typename Range>
    static auto getEnd(const Range& r) ->
            decltype(r.end()) {
        return r.end();
    }
};
template <> struct IteratorAccess<true> {
    template <typename Range>
    static auto getBegin(const Range& r) ->
            decltype(r.rbegin()) {
        return r.rbegin();
    }

    template <typename Range>
    static auto getEnd(const Range& r) ->
            decltype(r.rend()) {
        return r.rend();
    }
};

}  // namespace details

}  // namespace vpu
