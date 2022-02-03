// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cassert>

#include <type_traits>
#include <iterator>
#include <utility>
#include <memory>

#include <vpu/utils/range/helpers.hpp>
#include <vpu/utils/checked_cast.hpp>

namespace vpu {

namespace details {

template <typename T>
class GenericRangeWrapperBase {
public:
    class IteratorWrapperBase {
    public:
        using Ptr = std::shared_ptr<IteratorWrapperBase>;

        virtual ~IteratorWrapperBase() = default;

        virtual T get() const = 0;
        virtual void advance() = 0;
        virtual bool isEqual(const Ptr& other) const = 0;
    };

    using Ptr = std::shared_ptr<GenericRangeWrapperBase>;

    virtual ~GenericRangeWrapperBase() = default;

    virtual typename IteratorWrapperBase::Ptr begin() const = 0;
    virtual typename IteratorWrapperBase::Ptr end() const = 0;

    virtual T front() const = 0;

    virtual std::size_t size() const = 0;

    virtual bool empty() const = 0;
};

template <class BaseRange>
class GenericRangeWrapper final : public GenericRangeWrapperBase<typename BaseRange::value_type> {
    using BaseValueType = typename BaseRange::value_type;
    using Base = GenericRangeWrapperBase<BaseValueType>;
    using BaseIteratorWrapper = typename Base::IteratorWrapperBase;
    using BaseIteratorWrapperPtr = typename BaseIteratorWrapper::Ptr;

    template <class Iterator>
    class IteratorWrapper final : public Base::IteratorWrapperBase {
    public:
        explicit IteratorWrapper(Iterator iter) : _iter(std::move(iter)) {}

        BaseValueType get() const override {
            return *_iter;
        }

        void advance() override {
            ++_iter;
        }

        bool isEqual(const BaseIteratorWrapperPtr& other) const override {
            if (auto actualOther = dynamic_cast<IteratorWrapper*>(other.get())) {
                return _iter == actualOther->_iter;
            }
            return false;
        }

    private:
        Iterator _iter;
    };

public:
    explicit GenericRangeWrapper(BaseRange base) : _base(std::move(base)) {}

    typename Base::IteratorWrapperBase::Ptr begin() const override {
        return std::make_shared<IteratorWrapper<typename BaseRange::const_iterator>>(_base.begin());
    }

    typename Base::IteratorWrapperBase::Ptr end() const override {
        return std::make_shared<IteratorWrapper<typename BaseRange::const_iterator>>(_base.end());
    }

    BaseValueType front() const override {
        return _base.front();
    }

    std::size_t size() const override {
        return checked_cast<std::size_t>(_base.size());
    }

    bool empty() const override {
        return _base.empty();
    }

private:
    BaseRange _base;
};

}  // namespace details

template <class T>
class GenericRange final {
public:
    static constexpr bool has_reverse_iter = false;
    static constexpr bool has_random_access = false;
    static constexpr bool const_time_size = false;

private:
    using BaseWrapper = details::GenericRangeWrapperBase<T>;

    class Iterator final {
    public:
        using value_type = T;

        using pointer = void;
        using reference = void;

        using iterator_category = std::input_iterator_tag;
        using difference_type = std::ptrdiff_t;

        Iterator() = default;

        value_type operator*() const {
            return _wrapper->get();
        }

        Iterator& operator++() {
            _wrapper->advance();
            return *this;
        }

        bool operator==(const Iterator& other) const {
            return _wrapper->isEqual(other._wrapper);
        }
        bool operator!=(const Iterator& other) const {
            return !(*this == other);
        }

    private:
        explicit Iterator(typename BaseWrapper::IteratorWrapperBase::Ptr wrapper) :
                _wrapper(std::move(wrapper)) {
        }

    private:
        typename BaseWrapper::IteratorWrapperBase::Ptr _wrapper;

    private:
        friend GenericRange;
    };

public:
    using value_type = T;
    using size_type = std::size_t;

    using iterator = Iterator;

    using const_iterator = iterator;

    GenericRange() = default;

    template <class BaseRange>
    explicit GenericRange(BaseRange base) :
            _wrapper(std::make_shared<details::GenericRangeWrapper<BaseRange>>(std::move(base))) {
    }

    Iterator begin() const { return Iterator(_wrapper->begin()); }
    Iterator end() const { return Iterator(_wrapper->end()); }

    value_type front() const {
        return _wrapper->front();
    }

    size_type size() const {
        return _wrapper->size();
    }

    bool empty() const {
        return _wrapper->empty();
    }

private:
    typename BaseWrapper::Ptr _wrapper;
};

template <class BaseRange>
GenericRange<typename BaseRange::value_type> genericRange(BaseRange base) {
    return GenericRange<typename BaseRange::value_type>(std::move(base));
}

namespace details {

struct GenericRangeTag final {};

template <class BaseRange>
auto operator|(BaseRange base, GenericRangeTag&&) ->
        decltype(genericRange(std::move(base))) {
    return genericRange(std::move(base));
}

}  // namespace details

inline details::GenericRangeTag generic() {
    return details::GenericRangeTag{};
}

}  // namespace vpu
