// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <assert.h>

#include <cstddef>
#include <deque>
#include <ostream>
#include <vector>

#include "ngraph/attribute_adapter.hpp"
#include "ngraph/ngraph_visibility.hpp"
#include "ngraph/node.hpp"

namespace ngraph {
class OrderElement {
public:
    using Ptr = std::shared_ptr<OrderElement>;
    OrderElement(Node* n) : node(n) {}

    OrderElement::Ptr input;
    OrderElement::Ptr output;
    Node* node;

    bool id_is_not_set() const {
        return m_id == -1;
    }

    int64_t get_id() const {
        if (!id_is_not_set()) {
            return m_id;
        }

        auto el = input;
        while (el) {
            if (el->id_is_not_set()) {
                el = el->input;
            } else {
                return el->get_id();
            }
        }
        return 0;
        // throw ngraph::ngraph_error("No element with id was found");
    }

    std::pair<int64_t, int64_t> get_id_with_depth() const {
        if (!id_is_not_set()) {
            return {m_id, 0};
        }

        int64_t depth = 0;
        auto el = input;
        while (el) {
            if (el->id_is_not_set()) {
                ++depth;
                el = el->input;
            } else {
                return {el->get_id(), depth};
            }
        }
        return {0, depth};
        // throw ngraph::ngraph_error("No element with id was found");
    }

    void set_id(int64_t id) {
        m_id = id;
    }

    void reset_id() {
        m_id = -1;
    }

private:
    int64_t m_id{-1};
};

class Order {
public:
    using Ptr = std::shared_ptr<Order>;

    Order() = default;

    void push_back(OrderElement::Ptr element) {
        if (!m_begin) {
            m_begin = m_end = element;
        } else {
            element->input = m_end;
            m_end->output = element;
            m_end = element;
        }
        ++m_size;
        m_need_reindexing = true;
    }

    void push_back(const Order& order) {
        auto element = order.m_begin;
        element->input = m_end;
        m_end->output = element;
        m_end = order.m_end;
        m_size += order.size();
        m_need_reindexing = true;
    }

    void insert_after(OrderElement::Ptr element, OrderElement::Ptr new_element) {
        auto output = element->output;
        element->output = new_element;
        new_element->input = element;
        new_element->output = output;
        if (output) {
            output->input = new_element;
        } else {
            m_end = new_element;
        }
        ++m_size;
        m_need_reindexing = true;
    }

    void insert_after(OrderElement::Ptr element, Order::Ptr order) {
        auto output = element->output;
        element->output = order->m_begin;
        order->m_begin->input = element;
        order->m_end->output = output;
        if (output) {
            output->input = order->m_end;
        } else {
            m_end = order->m_end;
        }
        m_size += order->size();
        m_need_reindexing = true;
    }

    void validate() {
        int64_t cnt{0};
        auto el = m_begin;
        while (el) {
            el = el->output;
            ++cnt;
        }
        assert(cnt == m_size);
    }

    void reset(Order::Ptr order);

    void remove(OrderElement::Ptr element);

    OrderElement::Ptr begin() const {
        return m_begin;
    }

    OrderElement::Ptr end() const {
        return m_end;
    }

    void reindexing() {
        if (!m_need_reindexing)
            return;
        int64_t id{0};
        auto el = m_begin;
        int64_t cnt{0};
        while (el) {
            el->set_id(id);
            id += 1;
            el = el->output;
            ++cnt;
        }
        m_need_reindexing = false;
        assert(cnt == m_size);
    }

    int64_t size() const {
        return m_size;
    }

    bool initialization_is_finished() const {
        return m_initialization_finished;
    }

    void finish_initialization() {
        m_initialization_finished = true;
        reindexing();
    }

private:
    OrderElement::Ptr m_begin{nullptr};
    OrderElement::Ptr m_end{nullptr};
    int64_t m_size{0};
    bool m_need_reindexing{false};
    bool m_initialization_finished{false};
};

//    NGRAPH_API
//    std::ostream& operator<<(std::ostream& s, const Strides& strides);
}  // namespace ngraph
