// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cassert>

#include <functional>
#include <list>
#include <utility>

#include <vpu/utils/range/helpers.hpp>
#include <vpu/utils/range/container_range.hpp>
#include <vpu/utils/range/to_container.hpp>
#include <vpu/utils/optional.hpp>

namespace vpu {

template <typename T>
class ValueCache final {
public:
    ValueCache() = default;

    template <class F>
    explicit ValueCache(F&& f) {
        setInitializer(std::forward<F>(f));
    }

    ValueCache(const ValueCache&) = delete;
    ValueCache& operator=(const ValueCache&) = delete;

    ValueCache(ValueCache&&) = delete;
    ValueCache& operator=(ValueCache&&) = delete;

    template <class F>
    void setInitializer(F&& f) {
        _func = std::forward<F>(f);

        reset();
    }

    void reset() {
        _cache.reset();
    }

    T get() {
        if (!_cache.hasValue()) {
            assert(_func != nullptr);
            _cache = _func();
        }

        return _cache.get();
    }

private:
    std::function<T()> _func;
    Optional<T> _cache;
};

template <class Container>
class ContainerRangeCache final {
private:
    struct Holder final {
        Container cont;
        bool valid = false;
        int numUsers = 0;

        void addUser() {
            ++numUsers;
        }

        void removeUser() {
            assert(numUsers > 0);
            --numUsers;

            if (numUsers == 0 && !valid) {
                cont.clear();
            }
        }
    };

    class Range final : private details::DebugRange<Range> {
    public:
        static constexpr const bool has_reverse_iter = details::HasReverseIterator<Container>::value;
        static constexpr const bool has_random_access = details::HasRandomAccess<Container>::value;
        static constexpr const bool const_time_size = true;

    private:
        using ContIter = typename Container::const_iterator;
        using ContRevIter = typename details::GetReverseIterator<Container, has_reverse_iter>::reverse_iterator;

        template <class BaseIter, bool reverse>
        class Iterator final : private details::DebugIterator<Range> {
        public:
            using value_type = typename BaseIter::value_type;

            using pointer = typename BaseIter::pointer;
            using reference = typename BaseIter::reference;

            using iterator_category = std::input_iterator_tag;
            using difference_type = std::ptrdiff_t;

            Iterator() = default;

            ~Iterator() {
                if (_holder != nullptr) {
                    _holder->removeUser();
                }
            }

            Iterator(const Iterator& other) :
                    details::DebugIterator<Range>(other),
                    _holder(other._holder),
                    _cur(other._cur), _end(other._end) {
                if (_holder != nullptr) {
                    _holder->addUser();
                }
            }

            Iterator& operator=(const Iterator& other) {
                details::DebugIterator<Range>::operator=(other);
                if (&other != this) {
                    if (_holder != nullptr) {
                        _holder->removeUser();
                    }

                    _holder = other._holder;
                    _cur = other._cur;
                    _end = other._end;

                    if (_holder != nullptr) {
                        _holder->addUser();
                    }
                }
                return *this;
            }

            Iterator(Iterator&& other) :
                    details::DebugIterator<Range>(other),
                    _holder(other._holder),
                    _cur(std::move(other._cur)), _end(std::move(other._end)) {
                other._holder = nullptr;
            }

            Iterator& operator=(Iterator&& other) {
                details::DebugIterator<Range>::operator=(other);
                if (&other != this) {
                    if (_holder != nullptr) {
                        _holder->removeUser();
                    }

                    _holder = other._holder;
                    _cur = std::move(other._cur);
                    _end = std::move(other._end);

                    other._holder = nullptr;
                }
                return *this;
            }

            value_type operator*() const {
                assert(this->range() != nullptr);
                assert(_holder != nullptr);
                assert(_cur != _end);
                return *_cur;
            }

            Iterator& operator++() {
                assert(this->range() != nullptr);
                assert(_holder != nullptr);

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
                    assert(_holder != nullptr);
                    return _cur == other._cur;
                }
                return true;
            }

            bool operator!=(const Iterator& other) const {
                return !(*this == other);
            }

        private:
            explicit Iterator(const Range* range) :
                        details::DebugIterator<Range>(range) {
                _holder = this->range()->getHolder();
                assert(_holder != nullptr);

                _holder->addUser();

                _cur = details::IteratorAccess<reverse>::getBegin(_holder->cont);
                _end = details::IteratorAccess<reverse>::getEnd(_holder->cont);

                if (_cur == _end) {
                    this->reset();
                } else {
                    postAdvance();
                }
            }

            void reset() {
                details::DebugIterator<Range>::reset();

                if (_holder != nullptr) {
                    _holder->removeUser();
                    _holder = nullptr;
                }
            }

            void postAdvance() {
                assert(this->range() != nullptr);
                assert(_holder != nullptr);

                if (_cur == _end) {
                    this->reset();
                }
            }

        private:
            Holder* _holder = nullptr;

            BaseIter _cur;
            BaseIter _end;

        private:
            friend Range;
        };

    public:
        using value_type = typename Container::value_type;

        using iterator = Iterator<ContIter, false>;
        using reverse_iterator = typename std::conditional<has_reverse_iter, Iterator<ContRevIter, true>, void>::type;

        using const_iterator = iterator;
        using const_reverse_iterator = reverse_iterator;

        Range() = default;
        explicit Range(ContainerRangeCache* parent) : _parent(parent) {}

        iterator begin() const {
            return iterator(this);
        }
        iterator end() const {
            return iterator();
        }

        template <typename Q = Container, typename = typename std::enable_if<details::HasReverseIterator<Q>::value>::type>
        reverse_iterator rbegin() const {
            return reverse_iterator(this);
        }
        template <typename Q = Container, typename = typename std::enable_if<details::HasReverseIterator<Q>::value>::type>
        reverse_iterator rend() const {
            return reverse_iterator();
        }

        value_type front() const {
            auto h = getHolder();
            assert(h != nullptr);

            return details::getFrontImpl(h->cont);
        }
        template <typename Q = Container, typename = typename std::enable_if<details::HasReverseIterator<Q>::value>::type>
        value_type back() const {
            auto h = getHolder();
            assert(h != nullptr);

            return details::getBackImpl(h->cont);
        }

        size_t size() const {
            return getSize();
        }

        bool empty() const {
            return size() == 0;
        }

        template <typename Q = Container, typename = typename std::enable_if<details::HasRandomAccess<Q>::value>::type>
        value_type operator[](int ind) const {
            auto h = getHolder();
            assert(h != nullptr);

            assert(ind >= 0 && ind < h->cont.size());

            return h->cont[ind];
        }

        const Container& getContainer() const {
            auto h = getHolder();
            assert(h != nullptr);

            return h->cont;
        }

    private:
        Holder* getHolder() const {
            assert(_parent != nullptr);

            for (auto& h : _parent->_cache) {
                if (h.valid) {
                    return &h;
                }
            }

            for (auto& h : _parent->_cache) {
                if (h.numUsers == 0) {
                    assert(_parent->_func != nullptr);

                    h.cont = _parent->_func();
                    h.valid = true;

                    return &h;
                }
            }

            auto it = _parent->_cache.emplace(_parent->_cache.begin(), Holder{});

            auto& h = *it;

            h.cont = _parent->_func();
            h.valid = true;

            return &h;
        }

        size_t getSize() const {
            assert(_parent != nullptr);

            for (const auto& h : _parent->_cache) {
                if (h.valid) {
                    return checked_cast<size_t>(h.cont.size());
                }
            }

            for (auto& h : _parent->_cache) {
                if (h.numUsers == 0) {
                    assert(_parent->_func != nullptr);

                    h.cont = _parent->_func();
                    h.valid = true;

                    return checked_cast<size_t>(h.cont.size());
                }
            }

            auto it = _parent->_cache.emplace(_parent->_cache.begin(), Holder{});

            auto& h = *it;

            assert(_parent->_func != nullptr);

            h.cont = _parent->_func();
            h.valid = true;

            return checked_cast<size_t>(h.cont.size());
        }

    private:
        ContainerRangeCache* _parent = nullptr;

    private:
        template <class BaseIter, bool reverse>
        friend class Iterator;

        friend details::DebugIterator<Range>;
    };

    template <class F, bool NeedConvert> struct InitializerWrapper;
    template <class F> struct InitializerWrapper<F, false> {
        F f;
        Container operator()() const {
            return f();
        }
    };
    template <class F> struct InitializerWrapper<F, true> {
        F f;
        Container operator()() const {
            return f() | asContainer<Container>();
        }
    };

public:
    ContainerRangeCache() = default;

    template <class F>
    explicit ContainerRangeCache(F&& f) {
        setInitializer(std::forward<F>(f));
    }

    ContainerRangeCache(const ContainerRangeCache&) = delete;
    ContainerRangeCache& operator=(const ContainerRangeCache&) = delete;

    ContainerRangeCache(ContainerRangeCache&&) = default;
    ContainerRangeCache& operator=(ContainerRangeCache&&) = default;

    template <class F>
    void setInitializer(F&& f) {
        using FuncType = typename std::decay<F>::type;
        using ReturnType = typename std::decay<decltype(f())>::type;

        _func = InitializerWrapper<FuncType, !std::is_same<ReturnType, Container>::value>{std::forward<F>(f)};

        reset();
    }

    void reset() {
        for (auto it = _cache.begin(); it != _cache.end();) {
            if (it->numUsers == 0) {
                it = _cache.erase(it);
            } else {
                it->valid = false;
                ++it;
            }
        }
    }

    Range get() {
        return Range{this};
    }

private:
    std::function<Container()> _func;

    std::list<Holder> _cache;

private:
    friend Range;
};

}  // namespace vpu
