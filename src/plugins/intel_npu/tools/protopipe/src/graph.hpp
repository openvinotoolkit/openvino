//
// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <any>
#include <functional>
#include <memory>
#include <typeindex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "utils/error.hpp"

template <typename T>
class WeakHandle {
public:
    explicit WeakHandle(std::shared_ptr<T> obj): m_obj(obj) {
    }
    T* get() const {
        return m_obj.lock().get();
    }
    T* operator->() const {
        return get();
    }
    bool operator==(const WeakHandle& other) const {
        return get() == other.get();
    }

private:
    std::weak_ptr<T> m_obj;
};

namespace std {
template <typename T>
struct hash<WeakHandle<T>> {
    uint64_t operator()(const WeakHandle<T>& handle) const {
        return std::hash<T*>()(handle.get());
    }
};
}  // namespace std

class Graph;
class Node;
class Edge;

using NodeHandle = WeakHandle<Node>;
using EdgeHandle = WeakHandle<Edge>;
using Nodes = std::vector<NodeHandle>;
using Edges = std::vector<EdgeHandle>;
using NodeSet = std::unordered_set<NodeHandle>;
using EdgeSet = std::unordered_set<EdgeHandle>;

class Node {
    friend class Graph;
    using Ptr = std::shared_ptr<Node>;

public:
    Nodes srcNodes() const;
    Nodes dstNodes() const;
    Edges srcEdges() const;
    Edges dstEdges() const;

private:
    EdgeSet m_src_edges;
    EdgeSet m_dst_edges;
};

class Edge {
    friend class Graph;
    using Ptr = std::shared_ptr<Edge>;

public:
    Edge(NodeHandle src, NodeHandle dst): m_src(src), m_dst(dst) {
    }
    NodeHandle srcNode() const {
        return m_src;
    }
    NodeHandle dstNode() const {
        return m_dst;
    }

private:
    NodeHandle m_src;
    NodeHandle m_dst;
};

class Meta {
public:
    template <typename T>
    void set(T&& meta);
    template <typename T>
    const T& get() const;
    template <typename T>
    T& get();
    template <typename T>
    bool has() const;
    Meta& operator+=(const Meta& other);

private:
    using MetaStore = std::unordered_map<std::type_index, std::any>;
    MetaStore store;
};

template <typename T>
void Meta::set(T&& meta) {
    // NB: Check if there is no such meta yet.
    ASSERT(store.emplace(std::type_index(typeid(T)), std::forward<T>(meta)).second);
}

template <typename T>
bool Meta::has() const {
    auto it = store.find(std::type_index(typeid(T)));
    return it != store.end();
}

template <typename T>
const T& Meta::get() const {
    const auto it = store.find(std::type_index(typeid(T)));
    ASSERT(it != store.end());
    return *std::any_cast<T>(&it->second);
}

template <typename T>
T& Meta::get() {
    auto it = store.find(std::type_index(typeid(T)));
    ASSERT(it != store.end());
    return *std::any_cast<T>(&it->second);
}

class Graph {
public:
    NodeHandle create();
    void remove(NodeHandle nh);
    void remove(EdgeHandle eh);
    EdgeHandle link(NodeHandle src, NodeHandle dst);

    Meta& meta() {
        return m_graph_meta;
    }
    const Meta& meta() const {
        return m_graph_meta;
    }

    Meta& meta(NodeHandle handle);
    const Meta& meta(NodeHandle handle) const;
    Meta& meta(EdgeHandle handle);
    const Meta& meta(EdgeHandle handle) const;

    std::vector<NodeHandle> nodes() const;
    std::vector<NodeHandle> sorted() const;

private:
    template <typename T>
    struct MetaPtr {
        std::shared_ptr<T> ptr;
        Meta meta;
    };
    template <typename T>
    using MetaMap = std::unordered_map<T*, MetaPtr<T>>;

    Meta m_graph_meta;
    MetaMap<Node> m_nodes;
    MetaMap<Edge> m_edges;
};
