// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cassert>
#include <cstdint>

#include <limits>
#include <utility>
#include <algorithm>
#include <type_traits>
#include <functional>
#include <iterator>
#include <vector>
#include <list>
#include <set>
#include <unordered_set>
#include <map>
#include <unordered_map>
#include <memory>
#include <array>
#include <iostream>

#include <vpu/utils/numeric.hpp>
#include <vpu/utils/handle.hpp>
#include <vpu/utils/extra.hpp>

namespace vpu {

//
// SmallBufAllocator
//

template <typename T>
struct SmallBufElemMemory {
    static constexpr const size_t ElemSize = sizeof(T);

#ifdef _WIN32
    static constexpr const size_t ExtraSize = 16;
#else
    static constexpr const size_t ExtraSize = 0;
#endif

    static constexpr const size_t Align = alignof(T);

    typename std::aligned_storage<ElemSize + ExtraSize, Align>::type mem;
};

template <typename T, int _Capacity>
struct SmallBufHolder {
    using ElemMemory = SmallBufElemMemory<T>;

    static constexpr const size_t ElemSize = sizeof(ElemMemory);
    static constexpr const int Capacity = _Capacity;

    std::array<ElemMemory, Capacity> buf = {};
    bool bufLocked = false;
};

template <typename T, class BufHolder, class BaseAllocator = std::allocator<T>>
class SmallBufAllocator {
    using ElemMemory = typename BufHolder::ElemMemory;

    static constexpr const size_t ElemSize = BufHolder::ElemSize;
    static constexpr const int Capacity = BufHolder::Capacity;

    static_assert(sizeof(T) <= ElemSize, "sizeof(T) <= ElemSize");

public:
    using value_type = typename std::allocator_traits<BaseAllocator>::value_type;

    using pointer = typename std::allocator_traits<BaseAllocator>::pointer;
    using const_pointer = typename std::allocator_traits<BaseAllocator>::const_pointer;
    using void_pointer = typename std::allocator_traits<BaseAllocator>::void_pointer;
    using const_void_pointer = typename std::allocator_traits<BaseAllocator>::const_void_pointer;

    using size_type = typename std::allocator_traits<BaseAllocator>::size_type;
    using difference_type = typename std::allocator_traits<BaseAllocator>::difference_type;

    using propagate_on_container_copy_assignment = std::false_type;
    using propagate_on_container_move_assignment = std::false_type;
    using propagate_on_container_swap = std::false_type;

    template <typename T2> struct rebind {
        typedef SmallBufAllocator<
            T2, BufHolder,
            typename std::allocator_traits<BaseAllocator>::template rebind_alloc<T2>
        > other;
    };

    inline SmallBufAllocator() = default;
    inline explicit SmallBufAllocator(const BaseAllocator& baseAllocator) :
            _baseAllocator(baseAllocator) {
    }

    inline explicit SmallBufAllocator(BufHolder& h) :
            _buf(h.buf.data()), _bufLocked(&h.bufLocked) {
        *_bufLocked = false;
    }
    inline SmallBufAllocator(BufHolder& h, const BaseAllocator& baseAllocator) :
            _baseAllocator(baseAllocator),
            _buf(h.buf.data()), _bufLocked(&h.bufLocked) {
        *_bufLocked = false;
    }

    inline SmallBufAllocator(const SmallBufAllocator& other) :
            _baseAllocator(other._baseAllocator),
            _buf(other._buf), _bufLocked(other._bufLocked) {
    }
    inline SmallBufAllocator& operator=(const SmallBufAllocator& other) {
        if (&other != this) {
#ifndef NDEBUG
            if (_buf != nullptr && _bufLocked != nullptr) {
                assert(!*_bufLocked);
            }
#endif

            _baseAllocator = other._baseAllocator;
            _buf = other._buf;
            _bufLocked = other._bufLocked;
        }

        return *this;
    }

    template <typename T2, typename BufHolder2, class BaseAllocator2>
    inline SmallBufAllocator(const SmallBufAllocator<T2, BufHolder2, BaseAllocator2>& other) :
            _baseAllocator(other._baseAllocator),
            _buf(other._buf), _bufLocked(other._bufLocked) {
        static_assert(
            sizeof(T) <= SmallBufAllocator<T2, BufHolder2, BaseAllocator2>::ElemSize,
            "sizeof(T) <= SmallBufAllocator<T2, BufHolder2, BaseAllocator2>::ElemSize");
    }
    template <typename T2, typename BufHolder2, class BaseAllocator2>
    inline SmallBufAllocator& operator=(const SmallBufAllocator<T2, BufHolder2, BaseAllocator2>& other) {
        static_assert(
            sizeof(T) <= SmallBufAllocator<T2, BufHolder2, BaseAllocator2>::ElemSize,
            "sizeof(T) <= SmallBufAllocator<T2, BufHolder2, BaseAllocator2>::ElemSize");

        if (&other != this) {
#ifndef NDEBUG
            if (_buf != nullptr && _bufLocked != nullptr) {
                assert(!*_bufLocked);
            }
#endif

            _baseAllocator = other._baseAllocator;
            _buf = other._buf;
            _bufLocked = other._bufLocked;
        }

        return *this;
    }

    inline pointer allocate(size_type n, const_void_pointer hint = const_void_pointer()) {
        if (n <= Capacity && _buf != nullptr && _bufLocked != nullptr) {
            if (!*_bufLocked) {
                *_bufLocked = true;
                return static_cast<pointer>(_buf);
            }
        }

        return _baseAllocator.allocate(n, hint);
    }

    inline void deallocate(pointer ptr, size_type n) {
        if (_buf != nullptr && _bufLocked != nullptr) {
            if (ptr == static_cast<pointer>(_buf)) {
                assert(*_bufLocked);
                *_bufLocked = false;
                return;
            }
        }

        _baseAllocator.deallocate(ptr, n);
    }

    template <class U, class ...Args>
    inline void construct(U* ptr, Args&& ...args) {
        _baseAllocator.construct(ptr, std::forward<Args>(args)...);
    }

    template <class U>
    inline void destroy(U* ptr) {
        _baseAllocator.destroy(ptr);
    }

    inline void* getBuf() const { return _buf; }
    inline const BaseAllocator& getBaseAllocator() const { return _baseAllocator; }

private:
    BaseAllocator _baseAllocator;

    void* _buf = nullptr;
    bool* _bufLocked = nullptr;

    template <typename T2, typename BufHolder2, class BaseAllocator2>
    friend class SmallBufAllocator;
};

template <
    typename T1, typename BufHolder1, class BaseAllocator1,
    typename T2, typename BufHolder2, class BaseAllocator2
>
inline bool operator==(
        const SmallBufAllocator<T1, BufHolder1, BaseAllocator1>& a1,
        const SmallBufAllocator<T2, BufHolder2, BaseAllocator2>& a2) {
    return a1.getBuf() == a2.getBuf() && a1.getBaseAllocator() == a2.getBaseAllocator();
}
template <
    typename T1, typename BufHolder1, class BaseAllocator1,
    typename T2, typename BufHolder2, class BaseAllocator2
>
inline bool operator!=(
        const SmallBufAllocator<T1, BufHolder1, BaseAllocator1>& a1,
        const SmallBufAllocator<T2, BufHolder2, BaseAllocator2>& a2) {
    return a1.getBuf() != a2.getBuf() || a1.getBaseAllocator() != a2.getBaseAllocator();
}

//
// SmallVector
//

template <typename T, int Capacity = 8, class BaseAllocator = std::allocator<T>>
class SmallVector {
    using BufHolder = SmallBufHolder<T, Capacity>;
    using Alloc = SmallBufAllocator<T, BufHolder, BaseAllocator>;
    using BaseCont = std::vector<T, Alloc>;

public:
    using value_type = typename BaseCont::value_type;

    using size_type = typename BaseCont::size_type;

    using iterator = typename BaseCont::iterator;
    using const_iterator = typename BaseCont::const_iterator;

    inline SmallVector() : _allocator(_bufs), _base(_allocator) {
        _base.reserve(Capacity);
    }

    inline ~SmallVector() = default;

    inline explicit SmallVector(size_type count) : _allocator(_bufs), _base(count, _allocator) {}
    inline SmallVector(size_type count, const T& value) : _allocator(_bufs), _base(count, value, _allocator) {}
    inline SmallVector(std::initializer_list<T> init) : _allocator(_bufs), _base(init, _allocator) {}

    template <class InputIt>
    inline SmallVector(InputIt first, InputIt last) : _allocator(_bufs), _base(first, last, _allocator) {}

    inline SmallVector(const SmallVector& other) : _allocator(_bufs), _base(other._base, _allocator) {}
    inline SmallVector& operator=(const SmallVector& other) {
        if (&other != this) {
            _base = other._base;
        }
        return *this;
    }

    template <typename T2, int Capacity2, class BaseAllocator2>
    inline SmallVector(const SmallVector<T2, Capacity2, BaseAllocator2>& other) :  // NOLINT
            _allocator(_bufs), _base(other._base.begin(), other._base.end(), _allocator) {
    }
    template <typename T2, int Capacity2, class BaseAllocator2>
    inline SmallVector& operator=(const SmallVector<T2, Capacity2, BaseAllocator2>& other) {
        if (&other != this) {
            _base.assign(other._base.begin(), other._base.end());
        }
        return *this;
    }

    template <class Alloc2>
    inline SmallVector(const std::vector<T, Alloc2>& other) :  // NOLINT
            _allocator(_bufs), _base(other.begin(), other.end(), _allocator) {
    }
    template <class Alloc2>
    inline SmallVector& operator=(const std::vector<T, Alloc2>& other) {
        if (&other != this) {
            _base.assign(other.begin(), other.end());
        }
        return *this;
    }

    inline operator const BaseCont&() {
        return _base;
    }
    template <class Alloc2>
    inline operator std::vector<T, Alloc2>() {
        return std::vector<T, Alloc2>(_base.begin(), _base.end());
    }

    inline T& operator[](size_type pos) { return _base[pos]; }
    inline const T& operator[](size_type pos) const { return _base[pos]; }

    inline T& at(size_type pos) { return _base.at(pos); }
    inline const T& at(size_type pos) const { return _base.at(pos); }

    inline T& front() { return _base.front(); }
    inline const T& front() const { return _base.front(); }
    inline T& back() { return _base.back(); }
    inline const T& back() const { return _base.back(); }

    inline T* data() noexcept { return _base.data(); }
    inline const T* data() const noexcept { return _base.data(); }

    inline iterator begin() noexcept { return _base.begin(); }
    inline iterator end() noexcept { return _base.end(); }
    inline const_iterator begin() const noexcept { return _base.begin(); }
    inline const_iterator end() const noexcept { return _base.end(); }
    inline const_iterator cbegin() const noexcept { return _base.cbegin(); }
    inline const_iterator cend() const noexcept { return _base.cend(); }

    inline bool empty() const noexcept { return _base.empty(); }
    inline size_type size() const noexcept { return _base.size(); }

    inline void reserve(size_type cap) { _base.reserve(cap); }

    inline void clear() noexcept { _base.clear(); }

    inline void resize(size_type count) { _base.resize(count); }
    inline void resize(size_type count, const T& value) { _base.resize(count, value); }

    inline void push_back(const T& value) { _base.push_back(value); }
    inline void push_back(T&& value) { _base.push_back(value); }

    template <class... Args>
    inline void emplace_back(Args&&... args) { _base.emplace_back(std::forward<Args>(args)...); }

    inline void insert(iterator pos, const T& value) { _base.insert(pos, value); }
    inline void insert(iterator pos, T&& value) { _base.insert(pos, value); }
    inline void insert(iterator pos, size_type count, const T& value) { _base.insert(pos, count, value); }
    template <class InputIt>
    inline void insert(iterator pos, InputIt first, InputIt last) { _base.insert(pos, first, last); }
    inline void insert(iterator pos, std::initializer_list<T> ilist) { _base.insert(pos, ilist); }

    template <class... Args>
    inline iterator emplace(iterator pos, Args&&... args) { return _base.emplace(pos, std::forward<Args>(args)...); }

    inline void pop_back() { _base.pop_back(); }

    inline iterator erase(iterator pos) { return _base.erase(pos); }
    inline iterator erase(iterator first, iterator last) { return _base.erase(first, last); }

    inline void swap(SmallVector& other) { std::swap(*this, other); }

    inline bool operator==(const SmallVector& other) const { return _base == other._base; }
    inline bool operator!=(const SmallVector& other) const { return _base != other._base; }
    inline bool operator<(const SmallVector& other) const { return _base < other._base; }
    inline bool operator<=(const SmallVector& other) const { return _base <= other._base; }
    inline bool operator>(const SmallVector& other) const { return _base > other._base; }
    inline bool operator>=(const SmallVector& other) const { return _base >= other._base; }

private:
    template <typename T2, int Capacity2, class BaseAllocator2>
    friend class SmallVector;

    BufHolder _bufs;
    Alloc _allocator;
    BaseCont _base;
};

//
// IntrusivePtrList
//

template <class Base>
class IntrusivePtrListNode;

template <class Base>
class IntrusivePtrList;

template <class Base>
class IntrusivePtrListNode final {
public:
    inline explicit IntrusivePtrListNode(Base* owner) :
            _owner(owner) {
        assert(_owner != nullptr);
    }

    ~IntrusivePtrListNode();

    IntrusivePtrListNode(const IntrusivePtrListNode&) = delete;
    IntrusivePtrListNode& operator=(const IntrusivePtrListNode&) = delete;

private:
    inline Handle<Base> owner() const {
        return _owner->handle_from_this();
    }

    inline bool belongTo(const IntrusivePtrList<Base>* list) const {
        return _list == list;
    }
    inline void setList(IntrusivePtrList<Base>* list) {
        assert(_list == nullptr);
        _list = list;
    }

    inline bool hasIter(const typename IntrusivePtrList<Base>::Iterator* iter) const {
        return _iter == iter;
    }
    inline void setIter(typename IntrusivePtrList<Base>::Iterator* iter) {
        _iter = iter;
    }

    inline IntrusivePtrListNode* prevNode() const {
        return _prev;
    }
    inline IntrusivePtrListNode* nextNode() const {
        return _next;
    }

    void unlink();
    void linkBefore(IntrusivePtrListNode& nextNode);
    void linkAfter(IntrusivePtrListNode& prevNode);
    void updateFront(IntrusivePtrListNode& frontNode);

private:
    Base* _owner = nullptr;

    IntrusivePtrList<Base>* _list = nullptr;
    typename IntrusivePtrList<Base>::Iterator* _iter = nullptr;

    IntrusivePtrListNode* _prev = nullptr;
    IntrusivePtrListNode* _next = nullptr;

    friend class IntrusivePtrList<Base>;
};

template <class Base>
class IntrusivePtrList final {
public:
    class Iterator final {
    public:
        using iterator_category = std::input_iterator_tag;
        using value_type = Handle<Base>;
        using difference_type = std::ptrdiff_t;
        using pointer = const Handle<Base>*;
        using reference = const Handle<Base>&;

        inline Iterator() = default;

        inline Iterator(const Iterator&) = delete;
        inline Iterator& operator=(const Iterator&) = delete;

        inline explicit Iterator(IntrusivePtrListNode<Base> Base::* nodeField) :
                _nodeField(nodeField) {
            assert(_nodeField != nullptr);
        }

        Iterator(Iterator&& other);
        Iterator& operator=(Iterator&& other);

        Iterator(
                const std::shared_ptr<Base>& cur,
                IntrusivePtrListNode<Base> Base::* nodeField);

        ~Iterator();

        inline Handle<Base> operator*() const {
            return Handle<Base>(_cur);
        }

        inline Iterator& operator++() {
            if (!_skipNextAdvanced) {
                advance();
            }
            _skipNextAdvanced = false;
            return *this;
        }

        inline bool operator==(const Iterator& other) const { return _cur == other._cur; }
        inline bool operator!=(const Iterator& other) const { return _cur != other._cur; }

    private:
        inline void itemUnlinked() {
            advance();
            _skipNextAdvanced = true;
        }

        void advance();

    private:
        IntrusivePtrListNode<Base> Base::* _nodeField = nullptr;
        std::shared_ptr<Base> _cur;
        bool _skipNextAdvanced = false;

        friend class IntrusivePtrList;
        friend class IntrusivePtrListNode<Base>;
    };

    using value_type = Handle<Base>;
    using iterator = Iterator;
    using const_iterator = Iterator;

    inline explicit IntrusivePtrList(IntrusivePtrListNode<Base> Base::* nodeField) :
            _nodeField(nodeField) {
        assert(_nodeField != nullptr);
    }

    IntrusivePtrList(const IntrusivePtrList&) = delete;
    IntrusivePtrList& operator=(const IntrusivePtrList&) = delete;

    inline ~IntrusivePtrList() {
        try {
            clear();
        }
        catch (...) {
            std::cerr << "ERROR ~IntrusivePtrList(): can not clear data\n";
            std::abort();
        }
    }

    inline Iterator begin() const { return Iterator(_front.lock(), _nodeField); }
    inline Iterator end() const { return Iterator(_nodeField); }

    inline Iterator cbegin() const { return Iterator(_front.lock(), _nodeField); }
    inline Iterator cend() const { return Iterator(_nodeField); }

    inline size_t size() const { return _size; }
    inline bool empty() const { return _front == nullptr; }

    inline void clear() {
        while (!empty()) {
            pop_front();
        }
    }

    inline Handle<Base> front() const { return _front; }
    inline Handle<Base> back() const { return _back; }

    void push_back(const Handle<Base>& item);
    void push_back(const std::shared_ptr<Base>& item);

    void push_front(const Handle<Base>& item);
    void push_front(const std::shared_ptr<Base>& item);

    inline void erase(const Handle<Base>& item) {
        erase(item.get());
    }
    inline void erase(const std::shared_ptr<Base>& item) {
        erase(item.get());
    }
    inline void erase(const Iterator& it) {
        erase(it._cur.get());
    }

    inline void pop_front() {
        erase(_front);
    }
    inline void pop_back() {
        erase(_back);
    }

    inline bool has(const Handle<Base>& item) const {
        assert(!item.expired());

        const auto& itemNode = item.getPlain()->*_nodeField;
        return itemNode.belongTo(this);
    }
    inline bool has(const std::shared_ptr<Base>& item) const {
        const auto& itemNode = item.get()->*_nodeField;
        return itemNode.belongTo(this);
    }

private:
    void erase(Base* item);

private:
    IntrusivePtrListNode<Base> Base::* _nodeField = nullptr;

    Handle<Base> _front;
    Handle<Base> _back;

    size_t _size = 0;

    friend class IntrusivePtrListNode<Base>;
};

//
// Implementation
//

template <class Base>
inline IntrusivePtrListNode<Base>::~IntrusivePtrListNode() {
    try {
        if (_list != nullptr) {
            _list->erase(_owner);
            _list = nullptr;
        }
    }
    catch (...) {
        std::cerr << "ERROR ~IntrusivePtrListNode(): can not clear data\n";
        std::abort();
    }
}

template <class Base>
void IntrusivePtrListNode<Base>::unlink() {
    assert(_list != nullptr);

    if (_iter != nullptr) {
        _iter->itemUnlinked();
    }

    if (_prev != nullptr) {
        if (_prev->_next == this) {
            _prev->_next = _next;
        }
    }

    if (_next != nullptr) {
        if (_next->_prev == this) {
            _next->_prev = _prev;
        }
    }

    _list = nullptr;
    _iter = nullptr;
    _prev = nullptr;
    _next = nullptr;
}

template <class Base>
void IntrusivePtrListNode<Base>::linkBefore(IntrusivePtrListNode& nextNode) {
    assert(&nextNode != this);
    assert(_list == nullptr);
    assert(nextNode._list != nullptr);

    _prev = nextNode._prev;
    _next = &nextNode;
    nextNode._prev = this;
    if (_prev != nullptr) {
        _prev->_next = this;
    }

    _list = nextNode._list;
}

template <class Base>
void IntrusivePtrListNode<Base>::linkAfter(IntrusivePtrListNode& prevNode) {
    assert(&prevNode != this);
    assert(_list == nullptr);
    assert(prevNode._list != nullptr);

    _prev = &prevNode;
    _next = prevNode._next;
    prevNode._next = this;
    if (_next != nullptr) {
        _next->_prev = this;
    }

    _list = prevNode._list;
}

template <class Base>
void IntrusivePtrListNode<Base>::updateFront(IntrusivePtrListNode& frontNode) {
    assert(&frontNode != this);
    assert(_list != nullptr);
    assert(frontNode._list == _list);

    _prev = &frontNode;
    frontNode._next = this;
}

template <class Base>
inline IntrusivePtrList<Base>::Iterator::Iterator(Iterator&& other) {
    _nodeField = other._nodeField;
    _cur = std::move(other._cur);
    _skipNextAdvanced = other._skipNextAdvanced;

    if (_cur != nullptr) {
        assert(_nodeField != nullptr);

        auto& curNode = _cur.get()->*_nodeField;

        assert(curNode.hasIter(&other));
        curNode.setIter(this);
    }

    other._nodeField = nullptr;
    other._skipNextAdvanced = false;
}

template <class Base>
inline typename IntrusivePtrList<Base>::Iterator& IntrusivePtrList<Base>::Iterator::operator=(Iterator&& other) {
    if (this != &other) {
        if (_cur != nullptr) {
            auto& curNode = _cur.get()->*_nodeField;

            assert(curNode.hasIter(this));
            curNode.setIter(nullptr);
        }

        _nodeField = other._nodeField;
        _cur = std::move(other._cur);
        _skipNextAdvanced = other._skipNextAdvanced;

        if (_cur != nullptr) {
            assert(_nodeField != nullptr);

            auto& curNode = _cur.get()->*_nodeField;

            assert(curNode.hasIter(&other));
            curNode.setIter(this);
        }

        other._nodeField = nullptr;
        other._skipNextAdvanced = false;
    }
    return *this;
}

template <class Base>
inline IntrusivePtrList<Base>::Iterator::Iterator(
        const std::shared_ptr<Base>& cur,
        IntrusivePtrListNode<Base> Base::* nodeField) :
            _nodeField(nodeField),
            _cur(cur) {
    assert(_nodeField != nullptr);
    if (_cur != nullptr) {
        auto& curNode = _cur.get()->*_nodeField;

        assert(curNode.hasIter(nullptr));
        curNode.setIter(this);
    }
}

template <class Base>
inline IntrusivePtrList<Base>::Iterator::~Iterator() {
    if (_cur != nullptr) {
        auto& curNode = _cur.get()->*_nodeField;

        assert(curNode.hasIter(this));
        curNode.setIter(nullptr);
    }
}

template <class Base>
void IntrusivePtrList<Base>::Iterator::advance() {
    assert(_cur != nullptr);

    auto& curNode = _cur.get()->*_nodeField;
    assert(curNode.hasIter(this));

    curNode.setIter(nullptr);

    auto next = curNode.nextNode();
    if (next == nullptr) {
        _cur.reset();
    } else {
        auto nextOwner = next->owner();
        assert(!nextOwner.expired());

        auto& nextNode = nextOwner.get()->*_nodeField;
        assert(nextNode.hasIter(nullptr));

        nextNode.setIter(this);

        _cur = nextOwner.lock();
    }
}

template <class Base>
void IntrusivePtrList<Base>::push_back(const Handle<Base>& item) {
    IE_ASSERT(!item.expired());

    auto& itemNode = item.getPlain()->*_nodeField;

    if (_back == nullptr) {
        assert(_front == nullptr);

        _front = _back = item;
        itemNode.setList(this);
    } else {
        assert(_front != nullptr);

        auto& backNode = _back.get()->*_nodeField;
        itemNode.linkAfter(backNode);

        if (_front == _back) {
            itemNode.updateFront(backNode);
        }

        _back = item;
    }

    ++_size;
}

template <class Base>
void IntrusivePtrList<Base>::push_back(const std::shared_ptr<Base>& item) {
    auto& itemNode = item.get()->*_nodeField;

    if (_back == nullptr) {
        assert(_front == nullptr);

        _front = _back = item;
        itemNode.setList(this);
    } else {
        assert(_front != nullptr);

        auto& backNode = _back.get()->*_nodeField;
        itemNode.linkAfter(backNode);

        if (_front == _back) {
            itemNode.updateFront(backNode);
        }

        _back = item;
    }

    ++_size;
}

template <class Base>
void IntrusivePtrList<Base>::push_front(const Handle<Base>& item) {
    IE_ASSERT(!item.expired());

    auto& itemNode = item.getPlain()->*_nodeField;

    if (_front == nullptr) {
        assert(_back == nullptr);

        _front = _back = item;
        itemNode.setList(this);
    } else {
        assert(_back != nullptr);

        auto& frontNode = _front.get()->*_nodeField;
        itemNode.linkBefore(frontNode);

        _front = item;
    }

    ++_size;
}

template <class Base>
void IntrusivePtrList<Base>::push_front(const std::shared_ptr<Base>& item) {
    auto& itemNode = item.get()->*_nodeField;

    if (_front == nullptr) {
        assert(_back == nullptr);

        _front = _back = item;
        itemNode.setList(this);
    } else {
        assert(_back != nullptr);

        auto& frontNode = _front.get()->*_nodeField;
        itemNode.linkBefore(frontNode);

        _front = item;
    }

    ++_size;
}

template <class Base>
void IntrusivePtrList<Base>::erase(Base* item) {
    assert(item != nullptr);
    if (item == nullptr) {
        return;
    }
    assert(_size > 0);

    auto& itemNode = item->*_nodeField;
    assert(itemNode.belongTo(this));

    if (_front.getPlain() == item) {
        auto next = itemNode.nextNode();

        if (next == nullptr) {
            _front = nullptr;
        } else {
            assert(next->belongTo(this));

            auto nextOwner = next->owner();
            assert(!nextOwner.expired());

            _front = nextOwner;
        }
    }
    if (_back.getPlain() == item) {
        auto prev = itemNode.prevNode();

        if (prev == nullptr) {
            _back = nullptr;
        } else {
            assert(prev->belongTo(this));

            auto prevOwner = prev->owner();
            assert(!prevOwner.expired());

            _back = prevOwner;
        }
    }

    itemNode.unlink();

    --_size;
}

}  // namespace vpu
