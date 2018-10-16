// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#ifndef EDGE_HPP
#define EDGE_HPP

#include <memory>
#include <string>

#include "handle.hpp"
#include "metadata.hpp"

namespace ade
{

class Graph;
class Node;
class Edge;
using EdgeHandle = Handle<Edge>;
using NodeHandle = Handle<Node>;

class Edge final : public std::enable_shared_from_this<Edge>
{
public:
    NodeHandle srcNode() const;
    NodeHandle dstNode() const;
private:
    friend class Graph;
    friend class Node;

    Edge(Node* prev, Node* next);
    ~Edge();
    Edge(const Edge&) = delete;
    Edge& operator=(const Edge&) = delete;

    Graph* getParent() const;

    void unlink();
    void resetPrevNode(Node* newNode);
    void resetNextNode(Node* newNode);

    Node* m_prevNode = nullptr;
    Node* m_nextNode = nullptr;
};

}

#endif // EDGE_HPP
