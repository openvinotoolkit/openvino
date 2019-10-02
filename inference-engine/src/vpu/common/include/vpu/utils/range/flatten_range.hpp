// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cassert>

#include <type_traits>
#include <iterator>
#include <utility>
#include <numeric>
#include <memory>

#include <vpu/utils/range/helpers.hpp>
#include <vpu/utils/checked_cast.hpp>

namespace vpu {

namespace details {

template <class BaseOuterRange>
class FlattenRange final : private DebugRange<FlattenRange<BaseOuterRange>> {
private:
    using BaseOuterIterator = typename BaseOuterRange::const_iterator;
    using BaseOuterRevIterator = typename GetReverseIterator<BaseOuterRange, HasReverseIterator<BaseOuterRange>::value>::reverse_iterator;
    using BaseInnerRange = typename BaseOuterRange::value_type;

public:
    static constexpr bool has_reverse_iter = HasReverseIterator<BaseOuterRange>::value && HasReverseIterator<BaseInnerRange>::value;
    static constexpr bool has_random_access = false;
    static constexpr bool const_time_size = false;

private:
    template <typename BaseOuterIterator, bool reverse>
    class Iterator final : private DebugIterator<FlattenRange> {
        using BaseInnerIterator = typename std::conditional<
                reverse,
                typename BaseInnerRange::const_reverse_iterator,
                typename BaseInnerRange::const_iterator
            >::type;
        using BaseInnerIteratorValue = typename std::iterator_traits<BaseInnerIterator>::value_type;

    public:
        using value_type = BaseInnerIteratorValue;

        using pointer = void;
        using reference = void;

        using iterator_category = std::input_iterator_tag;
        using difference_type = std::ptrdiff_t;

        Iterator() = default;

        value_type operator*() const {
            assert(this->range() != nullptr);
            assert(_outCur != _outEnd);
            assert(_innerCur != _innerEnd);

            return *_innerCur;
        }

        Iterator& operator++() {
            assert(this->range() != nullptr);

            if (_outCur == _outEnd) {
                this->reset();
            } else {
                if (_innerCur != _innerEnd) {
                    ++_innerCur;
                }

                postAdvance();
            }

            return *this;
        }

        bool operator==(const Iterator& other) const {
            if (this->range() != other.range()) {
                return false;
            }
            if (this->range() != nullptr) {
                return _outCur == other._outCur && _innerCur == other._innerCur;
            }
            return true;
        }
        bool operator!=(const Iterator& other) const {
            return !(*this == other);
        }

    private:
        explicit Iterator(const FlattenRange* range) :
                DebugIterator<FlattenRange>(range) {
            _outCur = IteratorAccess<reverse>::getBegin(this->range()->_base);
            _outEnd = IteratorAccess<reverse>::getEnd(this->range()->_base);

            if (_outCur == _outEnd) {
                this->reset();
            } else {
                _innerRange = std::make_shared<BaseInnerRange>(*_outCur);

                _innerCur = IteratorAccess<reverse>::getBegin(*_innerRange);
                _innerEnd = IteratorAccess<reverse>::getEnd(*_innerRange);

                postAdvance();
            }
        }

        void postAdvance() {
            assert(this->range() != nullptr);

            while (_outCur != _outEnd &&
                   _innerCur == _innerEnd) {
                ++_outCur;

                if (_outCur != _outEnd) {
                    _innerRange = std::make_shared<BaseInnerRange>(*_outCur);

                    _innerCur = IteratorAccess<reverse>::getBegin(*_innerRange);
                    _innerEnd = IteratorAccess<reverse>::getEnd(*_innerRange);
                }
            }

            if (_outCur == _outEnd) {
                this->reset();
            }
        }

    private:
        BaseOuterIterator _outCur;
        BaseOuterIterator _outEnd;

        std::shared_ptr<BaseInnerRange> _innerRange;

        BaseInnerIterator _innerCur;
        BaseInnerIterator _innerEnd;

    private:
        friend FlattenRange;
    };

    template <bool reverse>
    class Iterator<void, reverse> final : private DebugIterator<FlattenRange> {};

public:
    using value_type = typename BaseInnerRange::value_type;
    using size_type = std::size_t;

    using iterator = Iterator<BaseOuterIterator, false>;
    using reverse_iterator = typename std::conditional<has_reverse_iter, Iterator<BaseOuterRevIterator, has_reverse_iter>, void>::type;

    using const_iterator = iterator;
    using const_reverse_iterator = reverse_iterator;

    FlattenRange() = default;
    explicit FlattenRange(BaseOuterRange base) : _base(std::move(base)) {}

    const_iterator begin() const {
        return const_iterator(this);
    }
    const_iterator end() const {
        return const_iterator();
    }

    template <typename Q1 = BaseOuterRange,
              typename Q2 = BaseInnerRange,
              typename = typename std::enable_if<HasReverseIterator<Q1>::value>::type,
              typename = typename std::enable_if<HasReverseIterator<Q2>::value>::type>
    const_reverse_iterator rbegin() const {
        return const_reverse_iterator(this);
    }
    template <typename Q1 = BaseOuterRange,
              typename Q2 = BaseInnerRange,
              typename = typename std::enable_if<HasReverseIterator<Q1>::value>::type,
              typename = typename std::enable_if<HasReverseIterator<Q2>::value>::type>
    const_reverse_iterator rend() const {
        return const_reverse_iterator();
    }

    value_type front() const {
        auto b = this->begin();
        assert(b != this->end());
        return *b;
    }
    template <typename Q1 = BaseOuterRange,
              typename Q2 = BaseInnerRange,
              typename = typename std::enable_if<HasReverseIterator<Q1>::value>::type,
              typename = typename std::enable_if<HasReverseIterator<Q2>::value>::type>
    value_type back() const {
        auto b = this->rbegin();
        assert(b != this->rend());
        return *b;
    }

    size_type size() const {
        return std::accumulate(
            _base.begin(), _base.end(),
            static_cast<size_type>(0),
            [](size_type res, const BaseInnerRange& innerRange) {
                return res + checked_cast<size_type>(innerRange.size());
            });
    }

    bool empty() const {
        return size() == 0;
    }

private:
    BaseOuterRange _base;

private:
    template <typename BaseIterator, bool reverse>
    friend class Iterator;

    friend DebugIterator<FlattenRange>;
};

}  // namespace details

template <class BaseOuterRange>
details::FlattenRange<BaseOuterRange> flattenRange(BaseOuterRange base) {
    return details::FlattenRange<BaseOuterRange>(std::move(base));
}

namespace details {

struct FlattenRangeTag final {};

template <class BaseOuterRange>
auto operator|(BaseOuterRange base, FlattenRangeTag&&) ->
        decltype(flattenRange(std::move(base))) {
    return flattenRange(std::move(base));
}

}  // namespace details

inline details::FlattenRangeTag flatten() {
    return details::FlattenRangeTag{};
}

}  // namespace vpu
