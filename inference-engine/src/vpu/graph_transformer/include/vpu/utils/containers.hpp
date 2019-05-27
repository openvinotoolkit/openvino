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

#include <vpu/utils/numeric.hpp>
#include <vpu/utils/handle.hpp>

namespace vpu {

//
// Small containers
//

namespace impl {

class IntBufferBase {
public:
    virtual ~IntBufferBase() = default;

    virtual void* getData() const = 0;
    virtual int* getAvailable() const = 0;
};

template <typename T, size_t ExpandBytes, int Capacity>
class SmallBufAllocator {
    static_assert(Capacity > 0, "Capacity > 0");

public:
    struct ExpandedData final {
        static constexpr const size_t FINAL_BYTE_SIZE = alignVal<alignof(size_t)>(sizeof(T) + ExpandBytes);

        std::array<uint8_t, FINAL_BYTE_SIZE> data = {};

        ExpandedData() = default;

        ExpandedData(const ExpandedData&) = delete;
        ExpandedData& operator=(const ExpandedData&) = delete;

        ExpandedData(ExpandedData&&) = delete;
        ExpandedData& operator=(ExpandedData&&) = delete;
    };

    class IntBuffer final : public IntBufferBase {
    public:
        IntBuffer() {
            clear();
        }

        IntBuffer(const IntBuffer&) = delete;
        IntBuffer& operator=(const IntBuffer&) = delete;

        IntBuffer(IntBuffer&&) = delete;
        IntBuffer& operator=(IntBuffer&&) = delete;

        void clear() {
            for (int i = 0; i < Capacity; ++i) {
                _available[i] = Capacity - i;
            }
            _available[Capacity] = 0;
        }

        void* getData() const override {
            return _data.data();
        }
        int* getAvailable() const override {
            return _available.data();
        }

    private:
        mutable std::array<ExpandedData, Capacity> _data = {};
        mutable std::array<int, Capacity + 1> _available = {};
    };

public:
    using value_type = T;

    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;

    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    using propagate_on_container_copy_assignment = std::false_type;
    using propagate_on_container_move_assignment = std::false_type;
    using propagate_on_container_swap = std::false_type;

    template <typename T2> struct rebind {
        static_assert(sizeof(ExpandedData) >= sizeof(T2), "sizeof(ExpandedData) >= sizeof(T2)");

        typedef SmallBufAllocator<T2, sizeof(ExpandedData) - sizeof(T2), Capacity> other;
    };

    SmallBufAllocator() noexcept = delete;

    SmallBufAllocator(const SmallBufAllocator&) noexcept = default;
    SmallBufAllocator& operator=(const SmallBufAllocator&) noexcept = default;

    SmallBufAllocator(SmallBufAllocator&& other) noexcept : _intBuf(other._intBuf) {
        other._intBuf = nullptr;
    }
    SmallBufAllocator& operator=(SmallBufAllocator&& other) noexcept {
        if (&other != this) {
            _intBuf = other._intBuf;
            other._intBuf = nullptr;
        }
        return *this;
    }

    explicit SmallBufAllocator(IntBuffer& intBuf) noexcept : _intBuf(&intBuf) {}

    template <typename T2, size_t ExpandBytes2, int Capacity2>
    SmallBufAllocator(const SmallBufAllocator<T2, ExpandBytes2, Capacity2>& other) noexcept : _intBuf(other._intBuf) {
        static_assert(sizeof(ExpandedData) == sizeof(typename SmallBufAllocator<T2, ExpandBytes2, Capacity2>::ExpandedData),
                      "sizeof(ExpandedData) == sizeof(typename SmallBufAllocator<T2, ExpandBytes2, Capacity2>::ExpandedData)");
        static_assert(Capacity <= Capacity2, "Capacity <= Capacity2");
    }

    T* allocate(std::size_t n) {
        assert(_intBuf != nullptr);

        auto data = static_cast<ExpandedData*>(_intBuf->getData());
        auto available = _intBuf->getAvailable();

        if (n <= Capacity) {
            int pos = -1;
            int minAvailable = std::numeric_limits<int>::max();
            for (int i = 0; i < Capacity; ++i) {
                if (available[i] >= static_cast<int>(n) && available[i] < minAvailable) {
                    pos = i;
                    minAvailable = available[i];
                }
            }

            if (pos >= 0) {
                for (int i = pos - 1; (i >= 0) && available[i] > 0; --i) {
                    assert(available[i] > available[pos]);
                    available[i] -= available[pos];
                }

                std::fill_n(available + pos, n, 0);

                return reinterpret_cast<T*>(data + pos);
            }
        }

        return static_cast<T*>(::operator new (n * sizeof(T)));
    }

    void deallocate(T* ptr, std::size_t n) noexcept {
        assert(_intBuf != nullptr);

        auto data = static_cast<ExpandedData*>(_intBuf->getData());
        auto available = _intBuf->getAvailable();

        auto tempPtr = reinterpret_cast<ExpandedData*>(ptr);

        if (tempPtr < data || tempPtr >= data + Capacity) {
            ::operator delete(tempPtr);
        } else {
            auto pos = static_cast<int>(tempPtr - data);

            for (int i = static_cast<int>(static_cast<std::size_t>(pos) + n - 1); i >= pos; --i) {
                assert(available[i] == 0);
                available[i] = available[i + 1] + 1;
            }
            for (int i = pos; (i >= 0) && available[i] > 0; --i) {
                available[i] += available[i + 1];
            }
        }
    }

    T* allocate(std::size_t n, const void*) noexcept {
        return allocate(n);
    }

    template <class U, class ...Args>
    void construct(U* p, Args&& ...args) {
        ::new(p) U(std::forward<Args>(args)...);
    }

    template <class U>
    void destroy(U* p) noexcept {
        p->~U();
    }

    std::size_t max_size() const noexcept {
        return std::numeric_limits<std::size_t>::max() / sizeof(T);
    }

    const IntBufferBase* intBuf() const { return _intBuf; }

private:
    template <typename T2, size_t ExpandBytes2, int Capacity2>
    friend class SmallBufAllocator;

    const IntBufferBase* _intBuf = nullptr;
};

template <typename T1, size_t ExpandBytes1, int Capacity1, typename T2, size_t ExpandBytes2, int Capacity2>
bool operator==(const SmallBufAllocator<T1, ExpandBytes1, Capacity1>& a1, const SmallBufAllocator<T2, ExpandBytes2, Capacity2>& a2) noexcept {
    return a1.intBuf() == a2.intBuf();
}
template <typename T1, size_t ExpandBytes1, int Capacity1, typename T2, size_t ExpandBytes2, int Capacity2>
bool operator!=(const SmallBufAllocator<T1, ExpandBytes1, Capacity1>& a1, const SmallBufAllocator<T2, ExpandBytes2, Capacity2>& a2) noexcept {
    return a1.intBuf() != a2.intBuf();
}

}  // namespace impl

template <typename T, int Capacity>
class SmallVector {
#if defined(_WIN32)
    static constexpr const size_t ExpandBytes = 8;
#else
    static constexpr const size_t ExpandBytes = 0;
#endif

    using Alloc = impl::SmallBufAllocator<T, ExpandBytes, Capacity>;
    using BaseCont = std::vector<T, Alloc>;

public:
    using value_type = typename BaseCont::value_type;

    using iterator = typename BaseCont::iterator;
    using const_iterator = typename BaseCont::const_iterator;

    SmallVector() : _base(Alloc(_intBuf)) {
        _base.reserve(Capacity);
    }

    ~SmallVector() = default;

    explicit SmallVector(std::size_t count) : _base(count, Alloc(_intBuf)) {}
    SmallVector(std::size_t count, const T& value) : _base(count, value, Alloc(_intBuf)) {}
    SmallVector(std::initializer_list<T> init) : _base(init, Alloc(_intBuf)) {}

    template <class InputIt>
    SmallVector(InputIt first, InputIt last) : _base(first, last, Alloc(_intBuf)) {}

    SmallVector(const SmallVector& other) :
            _base(other._base, Alloc(_intBuf)) {
    }
    SmallVector& operator=(const SmallVector& other) {
        if (&other != this) {
            _base = other._base;
        }
        return *this;
    }

    template <typename T2, int Capacity2>
    SmallVector(const SmallVector<T2, Capacity2>& other) :  // NOLINT
            _base(other._base.begin(), other._base.end(), Alloc(_intBuf)) {
    }
    template <typename T2, int Capacity2>
    SmallVector& operator=(const SmallVector<T2, Capacity2>& other) {
        if (&other != this) {
            _base.assign(other._base.begin(), other._base.end());
        }
        return *this;
    }

    template <class Alloc2>
    SmallVector(const std::vector<T, Alloc2>& other) :  // NOLINT
            _base(other.begin(), other.end(), Alloc(_intBuf)) {
    }
    template <class Alloc2>
    SmallVector& operator=(const std::vector<T, Alloc2>& other) {
        if (&other != this) {
            _base.assign(other.begin(), other.end());
        }
        return *this;
    }

    operator const BaseCont&() {
        return _base;
    }
    template <class Alloc2>
    operator std::vector<T, Alloc2>() {
        return std::vector<T, Alloc2>(_base.begin(), _base.end());
    }

    T& operator[](std::size_t pos) { return _base[pos]; }
    const T& operator[](std::size_t pos) const { return _base[pos]; }

    T& at(std::size_t pos) { return _base.at(pos); }
    const T& at(std::size_t pos) const { return _base.at(pos); }

    T& front() { return _base.front(); }
    const T& front() const { return _base.front(); }
    T& back() { return _base.back(); }
    const T& back() const { return _base.back(); }

    T* data() noexcept { return _base.data(); }
    const T* data() const noexcept { return _base.data(); }

    iterator begin() noexcept { return _base.begin(); }
    iterator end() noexcept { return _base.end(); }
    const_iterator begin() const noexcept { return _base.begin(); }
    const_iterator end() const noexcept { return _base.end(); }
    const_iterator cbegin() const noexcept { return _base.cbegin(); }
    const_iterator cend() const noexcept { return _base.cend(); }

    bool empty() const noexcept { return _base.empty(); }
    std::size_t size() const noexcept { return _base.size(); }

    void reserve(std::size_t cap) { _base.reserve(cap); }

    void clear() noexcept { _base.clear(); }

    void resize(std::size_t count) { _base.resize(count); }
    void resize(std::size_t count, const T& value) { _base.resize(count, value); }

    void push_back(const T& value) { _base.push_back(value); }
    void push_back(T&& value) { _base.push_back(value); }

    template <class... Args>
    void emplace_back(Args&&... args) { _base.emplace_back(std::forward<Args>(args)...); }

    void insert(iterator pos, const T& value) { _base.insert(pos, value); }
    void insert(iterator pos, T&& value) { _base.insert(pos, value); }
    void insert(iterator pos, std::size_t count, const T& value) { _base.insert(pos, count, value); }
    template <class InputIt>
    void insert(iterator pos, InputIt first, InputIt last) { _base.insert(pos, first, last); }
    void insert(iterator pos, std::initializer_list<T> ilist) { _base.insert(pos, ilist); }

    template <class... Args>
    iterator emplace(iterator pos, Args&&... args) { return _base.emplace(pos, std::forward<Args>(args)...); }

    void pop_back() { _base.pop_back(); }

    iterator erase(iterator pos) { return _base.erase(pos); }
    iterator erase(iterator first, iterator last) { return _base.erase(first, last); }

    void swap(SmallVector& other) { std::swap(*this, other); }

    bool operator==(const SmallVector& other) const { return _base == other._base; }
    bool operator!=(const SmallVector& other) const { return _base != other._base; }
    bool operator<(const SmallVector& other) const { return _base < other._base; }
    bool operator<=(const SmallVector& other) const { return _base <= other._base; }
    bool operator>(const SmallVector& other) const { return _base > other._base; }
    bool operator>=(const SmallVector& other) const { return _base >= other._base; }

private:
    template <typename T2, int Capacity2>
    friend class SmallVector;

    typename Alloc::IntBuffer _intBuf;
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
