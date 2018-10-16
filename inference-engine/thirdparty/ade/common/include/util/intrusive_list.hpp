// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#ifndef UTIL_INTRUSIVE_LIST_HPP
#define UTIL_INTRUSIVE_LIST_HPP

#include <utility>
#include <cstddef>
#include <iterator>

#include "util/type_traits.hpp"
#include "util/assert.hpp"

namespace util
{

/// Intrusive list node
/// It will unlink itself from any list during destruction
/// TODO: Add autoremove policy as template parameter?
class intrusive_list_node final
{
    intrusive_list_node* prev = nullptr;
    intrusive_list_node* next = nullptr;

public:
    template<typename T, intrusive_list_node T::* Elem>
    friend class intrusive_list;

    intrusive_list_node() = default;
    ~intrusive_list_node()
    {
        if (linked())
        {
            unlink();
        }
    }

    intrusive_list_node(const intrusive_list_node&) = delete;
    intrusive_list_node& operator=(const intrusive_list_node&) = delete;

    bool linked() const
    {
        ASSERT((nullptr == prev) == (nullptr == next));
        return nullptr != prev;
    }

    /// Link self before node
    void link_before(intrusive_list_node& node)
    {
        ASSERT(&node != this);
        ASSERT(!linked());
        ASSERT(node.linked());
        prev = node.prev;
        next = &node;
        node.prev->next = this;
        node.prev = this;
    }

    /// Link self after node
    void link_after(intrusive_list_node& node)
    {
        ASSERT(&node != this);
        ASSERT(!linked());
        ASSERT(node.linked());
        prev = &node;
        next = node.next;
        node.next->prev = this;
        node.next = this;
    }

    void unlink()
    {
        ASSERT(linked());
        ASSERT(this == prev->next);
        ASSERT(this == next->prev);
        prev->next = next;
        next->prev = prev;
        prev = nullptr;
        next = nullptr;
    }
};

/// Non-owning intrusive list container with std::list like interface
/// (except it don't have "size()" method)
template<typename T, intrusive_list_node T::* Elem>
class intrusive_list final
{
public:
    intrusive_list()
    {
        // Root is special case, linked to itself
        root.next = &root;
        root.prev = &root;
    }
    ~intrusive_list()
    {
        clear();
    }

    intrusive_list(const intrusive_list&) = delete;
    intrusive_list& operator=(const intrusive_list&) = delete;

    template<bool IsConst>
    struct iter final
    {
        using value_type = util::conditional_t<IsConst, const T, T>;
        using pointer = value_type*;
        using reference = value_type&;
        using iterator_category = std::bidirectional_iterator_tag;
        using difference_type = std::ptrdiff_t;

        friend class intrusive_list<T, Elem>;

        using node_t = util::conditional_t<IsConst, const intrusive_list_node*, intrusive_list_node*>;

        iter() = default;
        iter(node_t node): current(node)
        {
            ASSERT(nullptr != current);
        }
        iter(const iter&) = default;
        iter& operator=(const iter&) = default;

        reference& operator*()
        {
            ASSERT(valid());
            return get_object(*current);
        }

        inline iter& operator++() { to_next(); return *this; }
        inline iter& operator--() { to_prev(); return *this; }
        inline iter operator++(int) { auto tmp(*this); to_next(); return tmp; }
        inline iter operator--(int) { auto tmp(*this); to_prev(); return tmp; }

        template<bool C>
        inline bool operator==(const iter<C>& other) const { return current == other.current; }

        template<bool C>
        inline bool operator!=(const iter<C>& other) const { return current != other.current; }

    private:
        node_t current = nullptr;

        bool valid() const
        {
            return nullptr != current;
        }

        void to_next()
        {
            ASSERT(valid());
            current = current->next;
            ASSERT(valid());
        }

        void to_prev()
        {
            ASSERT(valid());
            current = current->prev;
            ASSERT(valid());
        }
    };

    using iterator = iter<false>;
    using const_iterator = iter<true>;

    iterator begin()
    {
        return iterator(root.next);
    }

    iterator end()
    {
        return iterator(&root);
    }

    const_iterator begin() const
    {
        return iterator(root.next);
    }

    const_iterator end() const
    {
        return iterator(&root);
    }

    bool empty() const
    {
        ASSERT((root.next == &root) == (root.prev == &root));
        return root.next == &root;
    }

    iterator insert(iterator pos, T& item)
    {
        ASSERT(pos.valid());
        auto& node = (item.*Elem);
        ASSERT(!node.linked());
        node.link_before(*(pos.current));
        return iterator(&node);
    }

    iterator erase(iterator pos)
    {
        ASSERT(pos.valid());
        iterator iter(pos.current->next);
        pos.current->unlink();
        return iter;
    }

    void clear()
    {
        // Unlink all nodes
        auto node = root.next;
        while (&root != node)
        {
            ASSERT(nullptr != node);
            auto nextNode = node->next;
            node->unlink();
            node = nextNode;
        }
    }

    void push_back(T& item)
    {
        auto& node = (item.*Elem);
        ASSERT(!node.linked());
        node.link_before(root);
    }

    void push_front(T& item)
    {
        auto& node = (item.*Elem);
        ASSERT(!node.linked());
        node.link_after(root);
    }

private:
    // All list nodes are cyclically linked through this root
    intrusive_list_node root;

    static T& get_object(intrusive_list_node& node)
    {
        return *reinterpret_cast<T*>(reinterpret_cast<char*>(&node) - get_offset());
    }

    static const T& get_object(const intrusive_list_node& node)
    {
        return *reinterpret_cast<const T*>(reinterpret_cast<const char*>(&node) - get_offset());
    }

    static ptrdiff_t get_offset()
    {
        return (((ptrdiff_t)&(reinterpret_cast<T*>(0x1000)->*Elem)) - 0x1000);
    }
};

}

#endif // UTIL_INTRUSIVE_LIST_HPP
