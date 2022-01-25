// Copyright (C) 2018-2022 Intel Corporation
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

namespace vpu {

template <class Base>
class IntrusiveHandleListNode;

template <class Base>
class IntrusiveHandleList;

template <class Base>
class IntrusiveHandleListNode final {
public:
    explicit IntrusiveHandleListNode(Base* owner) :
            _owner(owner) {
        assert(_owner != nullptr);
    }

    ~IntrusiveHandleListNode();

    IntrusiveHandleListNode(const IntrusiveHandleListNode&) = delete;
    IntrusiveHandleListNode& operator=(const IntrusiveHandleListNode&) = delete;

private:
    Handle<Base> owner() const {
        return _owner;
    }

    bool belongTo(const IntrusiveHandleList<Base>* list) const {
        return _list == list;
    }
    void setList(IntrusiveHandleList<Base>* list) {
        assert(_list == nullptr);
        _list = list;
    }

    bool hasIter(const typename IntrusiveHandleList<Base>::Iterator* iter) const {
        assert(iter != nullptr);
        return _iters.count(const_cast<typename IntrusiveHandleList<Base>::Iterator*>(iter)) != 0;
    }
    void addIter(typename IntrusiveHandleList<Base>::Iterator* iter) {
        assert(iter != nullptr);
        _iters.insert(iter);
    }
    void removeIter(typename IntrusiveHandleList<Base>::Iterator* iter) {
        assert(iter != nullptr);
        _iters.erase(iter);
    }

    IntrusiveHandleListNode* prevNode() const {
        return _prev;
    }
    IntrusiveHandleListNode* nextNode() const {
        return _next;
    }

    void unlink();
    void linkBefore(IntrusiveHandleListNode& nextNode);
    void linkAfter(IntrusiveHandleListNode& prevNode);
    void updateFront(IntrusiveHandleListNode& frontNode);

private:
    Base* _owner = nullptr;

    IntrusiveHandleList<Base>* _list = nullptr;
    std::unordered_set<typename IntrusiveHandleList<Base>::Iterator*> _iters;

    IntrusiveHandleListNode* _prev = nullptr;
    IntrusiveHandleListNode* _next = nullptr;

    friend class IntrusiveHandleList<Base>;
};

template <class Base>
class IntrusiveHandleList final {
    class Iterator final {
    public:
        using value_type = Handle<Base>;
        using pointer = value_type*;
        using reference = value_type&;

        using iterator_category = std::input_iterator_tag;
        using difference_type = std::ptrdiff_t;

        Iterator() = default;
        ~Iterator();

        Iterator(const Iterator&);
        Iterator& operator=(const Iterator&);

        Iterator(Iterator&& other);
        Iterator& operator=(Iterator&& other);

        Handle<Base> operator*() const {
            assert(_cur != nullptr);
            return _cur;
        }

        Iterator& operator++() {
            if (!_skipNextAdvanced) {
                advanceImpl();
            }
            _skipNextAdvanced = false;
            return *this;
        }

        bool operator==(const Iterator& other) const { return _cur == other._cur; }
        bool operator!=(const Iterator& other) const { return _cur != other._cur; }

    private:
        Iterator(
                bool reversed,
                IntrusiveHandleListNode<Base> Base::* nodeField) :
                    _reversed(reversed), _nodeField(nodeField) {
            assert(_nodeField != nullptr);
        }

        Iterator(
                bool reversed,
                Base* cur,
                IntrusiveHandleListNode<Base> Base::* nodeField);

        void itemUnlinked() {
            advanceImpl();
            _skipNextAdvanced = true;
        }

        void advanceImpl();

    private:
        bool _reversed = false;

        IntrusiveHandleListNode<Base> Base::* _nodeField = nullptr;
        Base* _cur = nullptr;

        bool _skipNextAdvanced = false;

    private:
        friend class IntrusiveHandleList;
        friend class IntrusiveHandleListNode<Base>;
    };

public:
    using value_type = Handle<Base>;

    using size_type = size_t;

    using iterator = Iterator;
    using reverse_iterator = Iterator;

    using const_iterator = iterator;
    using const_reverse_iterator = reverse_iterator;

    explicit IntrusiveHandleList(IntrusiveHandleListNode<Base> Base::* nodeField) :
            _nodeField(nodeField) {
        assert(_nodeField != nullptr);
    }

    IntrusiveHandleList(const IntrusiveHandleList&) = delete;
    IntrusiveHandleList& operator=(const IntrusiveHandleList&) = delete;

    IntrusiveHandleList(IntrusiveHandleList&&);
    IntrusiveHandleList& operator=(IntrusiveHandleList&&);

    ~IntrusiveHandleList() {
        try {
            clear();
        }
        catch (...) {
            std::cerr << "ERROR ~IntrusiveHandleList(): can not clear data\n";
            std::abort();
        }
    }

    Iterator begin() const { return Iterator(false, _front, _nodeField); }
    Iterator end() const { return Iterator(false, _nodeField); }

    Iterator cbegin() const { return Iterator(false, _front, _nodeField); }
    Iterator cend() const { return Iterator(false, _nodeField); }

    Iterator rbegin() const { return Iterator(true, _back, _nodeField); }
    Iterator rend() const { return Iterator(true, _nodeField); }

    Iterator crbegin() const { return Iterator(true, _back, _nodeField); }
    Iterator crend() const { return Iterator(true, _nodeField); }

    size_t size() const { return _size; }
    bool empty() const { return _front == nullptr; }

    void clear() {
        while (!empty()) {
            pop_front();
        }
    }

    Handle<Base> front() const { return _front; }
    Handle<Base> back() const { return _back; }

    void push_front(const Handle<Base>& item);
    void push_back(const Handle<Base>& item);

    void erase(const Handle<Base>& item) {
        erase(item.get());
    }
    void erase(const Iterator& it) {
        erase(it._cur);
    }

    void pop_front() {
        erase(_front);
    }
    void pop_back() {
        erase(_back);
    }

    bool has(const Handle<Base>& item) const {
        assert(!item.expired());

        const auto& itemNode = item.get()->*_nodeField;
        return itemNode.belongTo(this);
    }

private:
    void erase(Base* item);

private:
    IntrusiveHandleListNode<Base> Base::* _nodeField = nullptr;

    Base* _front = nullptr;
    Base* _back = nullptr;

    size_t _size = 0;

    friend class IntrusiveHandleListNode<Base>;
};

template <class Base>
IntrusiveHandleListNode<Base>::~IntrusiveHandleListNode() {
    try {
        if (_list != nullptr) {
            _list->erase(_owner);
            _list = nullptr;
        }
    }
    catch (...) {
        std::cerr << "ERROR ~IntrusiveHandleListNode(): can not clear data\n";
        std::abort();
    }
}

template <class Base>
void IntrusiveHandleListNode<Base>::unlink() {
    assert(_list != nullptr);
    _list = nullptr;

    while (!_iters.empty()) {
        (*_iters.begin())->itemUnlinked();
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

    _prev = nullptr;
    _next = nullptr;
}

template <class Base>
void IntrusiveHandleListNode<Base>::linkBefore(IntrusiveHandleListNode& nextNode) {
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
void IntrusiveHandleListNode<Base>::linkAfter(IntrusiveHandleListNode& prevNode) {
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
void IntrusiveHandleListNode<Base>::updateFront(IntrusiveHandleListNode& frontNode) {
    assert(&frontNode != this);
    assert(_list != nullptr);
    assert(frontNode._list == _list);

    _prev = &frontNode;
    frontNode._next = this;
}

template <class Base>
IntrusiveHandleList<Base>::Iterator::Iterator(
        bool reversed,
        Base* cur,
        IntrusiveHandleListNode<Base> Base::* nodeField) :
            _reversed(reversed),
            _nodeField(nodeField),
            _cur(cur) {
    assert(_nodeField != nullptr);
    if (_cur != nullptr) {
        auto& curNode = _cur->*_nodeField;

        assert(!curNode.hasIter(this));
        curNode.addIter(this);
    }
}

template <class Base>
IntrusiveHandleList<Base>::Iterator::Iterator(const Iterator& other) :
        _reversed(other._reversed),
        _nodeField(other._nodeField),
        _cur(other._cur),
        _skipNextAdvanced(other._skipNextAdvanced) {
    if (_cur != nullptr) {
        assert(_nodeField != nullptr);

        auto& curNode = _cur->*_nodeField;

        assert(curNode.hasIter(&other));
        curNode.addIter(this);
    }
}

template <class Base>
typename IntrusiveHandleList<Base>::Iterator& IntrusiveHandleList<Base>::Iterator::operator=(const Iterator& other) {
    if (this != &other) {
        if (_cur != nullptr) {
            assert(_nodeField != nullptr);

            auto& curNode = _cur->*_nodeField;

            assert(curNode.hasIter(this));
            curNode.removeIter(this);
        }

        _reversed = other._reversed;
        _nodeField = other._nodeField;
        _cur = other._cur;
        _skipNextAdvanced = other._skipNextAdvanced;

        if (_cur != nullptr) {
            assert(_nodeField != nullptr);

            auto& curNode = _cur->*_nodeField;

            assert(curNode.hasIter(&other));
            curNode.addIter(this);
        }
    }
    return *this;
}

template <class Base>
IntrusiveHandleList<Base>::Iterator::Iterator(Iterator&& other) :
        _reversed(other._reversed),
        _nodeField(other._nodeField),
        _cur(other._cur),
        _skipNextAdvanced(other._skipNextAdvanced) {
    if (_cur != nullptr) {
        assert(_nodeField != nullptr);

        auto& curNode = _cur->*_nodeField;

        assert(curNode.hasIter(&other));
        curNode.removeIter(&other);
        curNode.addIter(this);
    }

    other._nodeField = nullptr;
    other._cur = nullptr;
    other._skipNextAdvanced = false;
}

template <class Base>
typename IntrusiveHandleList<Base>::Iterator& IntrusiveHandleList<Base>::Iterator::operator=(Iterator&& other) {
    if (this != &other) {
        if (_cur != nullptr) {
            assert(_nodeField != nullptr);

            auto& curNode = _cur->*_nodeField;

            assert(curNode.hasIter(this));
            curNode.removeIter(this);
        }

        _reversed = other._reversed;
        _nodeField = other._nodeField;
        _cur = other._cur;
        _skipNextAdvanced = other._skipNextAdvanced;

        if (_cur != nullptr) {
            assert(_nodeField != nullptr);

            auto& curNode = _cur->*_nodeField;

            assert(curNode.hasIter(&other));
            curNode.removeIter(&other);
            curNode.addIter(this);
        }

        other._nodeField = nullptr;
        other._cur = nullptr;
        other._skipNextAdvanced = false;
    }
    return *this;
}

template <class Base>
IntrusiveHandleList<Base>::Iterator::~Iterator() {
    if (_cur != nullptr) {
        assert(_nodeField != nullptr);

        auto& curNode = _cur->*_nodeField;

        assert(curNode.hasIter(this));
        curNode.removeIter(this);
    }
}

template <class Base>
void IntrusiveHandleList<Base>::Iterator::advanceImpl() {
    assert(_cur != nullptr);

    auto& curNode = _cur->*_nodeField;
    assert(curNode.hasIter(this));
    curNode.removeIter(this);

    auto next = _reversed ? curNode.prevNode() : curNode.nextNode();
    if (next == nullptr) {
        _cur = nullptr;
    } else {
        auto nextOwner = next->owner();
        assert(!nextOwner.expired());

        _cur = nextOwner.get();

        auto& newCurNode = _cur->*_nodeField;
        assert(!newCurNode.hasIter(this));
        newCurNode.addIter(this);
    }
}

template <class Base>
IntrusiveHandleList<Base>::IntrusiveHandleList(IntrusiveHandleList&& other) {
    _nodeField = other._nodeField;
    assert(_nodeField != nullptr);

    while (!other.empty()) {
        const auto item = other.front();
        other.pop_front();
        push_back(item);
    }
}

template <class Base>
IntrusiveHandleList<Base>& IntrusiveHandleList<Base>::operator=(IntrusiveHandleList&& other) {
    _nodeField = other._nodeField;
    assert(_nodeField != nullptr);

    if (&other != this) {
        clear();

        while (!other.empty()) {
            const auto item = other.front();
            other.pop_front();
            push_back(item);
        }
    }

    return *this;
}

template <class Base>
void IntrusiveHandleList<Base>::push_back(const Handle<Base>& item) {
    IE_ASSERT(!item.expired());

    auto& itemNode = item.get()->*_nodeField;

    if (_back == nullptr) {
        assert(_front == nullptr);

        _front = _back = item.get();
        itemNode.setList(this);
    } else {
        assert(_front != nullptr);

        auto& backNode = _back->*_nodeField;
        itemNode.linkAfter(backNode);

        if (_front == _back) {
            itemNode.updateFront(backNode);
        }

        _back = item.get();
    }

    ++_size;
}

template <class Base>
void IntrusiveHandleList<Base>::push_front(const Handle<Base>& item) {
    IE_ASSERT(!item.expired());

    auto& itemNode = item.get()->*_nodeField;

    if (_front == nullptr) {
        assert(_back == nullptr);

        _front = _back = item.get();
        itemNode.setList(this);
    } else {
        assert(_back != nullptr);

        auto& frontNode = _front->*_nodeField;
        itemNode.linkBefore(frontNode);

        _front = item.get();
    }

    ++_size;
}

template <class Base>
void IntrusiveHandleList<Base>::erase(Base* item) {
    if (item == nullptr) {
        return;
    }

    assert(_size > 0);

    auto& itemNode = item->*_nodeField;
    assert(itemNode.belongTo(this));

    if (_front == item) {
        auto nextNode = itemNode.nextNode();

        if (nextNode == nullptr) {
            _front = nullptr;
        } else {
            assert(nextNode->belongTo(this));

            auto nextOwner = nextNode->owner();
            assert(!nextOwner.expired());

            _front = nextOwner.get();
        }
    }
    if (_back == item) {
        auto prevNode = itemNode.prevNode();

        if (prevNode == nullptr) {
            _back = nullptr;
        } else {
            assert(prevNode->belongTo(this));

            auto prevOwner = prevNode->owner();
            assert(!prevOwner.expired());

            _back = prevOwner.get();
        }
    }

    itemNode.unlink();

    --_size;
}

}  // namespace vpu
