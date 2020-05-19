//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include <limits>
#include <list>
#include <sstream>

#include "ngraph/pass/pass.hpp"

namespace ngraph
{
    namespace pass
    {
        class MemoryLayout;
        class MemoryNode;
        class MemoryManager;
    }
}

class NGRAPH_API ngraph::pass::MemoryLayout : public FunctionPass
{
public:
    MemoryLayout(size_t alignment = 1, bool disable_memory_sharing = false);
    bool run_on_function(std::shared_ptr<ngraph::Function>) override;

private:
    size_t m_alignment;
    bool m_disable_memory_sharing;
};

class NGRAPH_API ngraph::pass::MemoryManager
{
public:
    enum class block_state
    {
        FREE,
        ALLOCATED
    };

    enum class allocation_scheme
    {
        FIRST_FIT,
        BEST_FIT,
        NO_REUSE
    };

    class node
    {
    public:
        node(size_t size, block_state state);

        bool is_free() const { return m_state == block_state::FREE; }
        size_t m_size;
        block_state m_state;
    };

    MemoryManager(size_t alignment = 1, bool disable_reuse = false);
    // memory_manager& alignment(size_t a);

    size_t allocate(size_t size);
    void free(size_t offset);

    void dump(std::ostream&);

    static size_t align(size_t x, size_t alignment);

    std::list<node>::iterator begin() { return m_node_list.begin(); }
    std::list<node>::iterator end() { return m_node_list.end(); }
    std::list<node>::const_iterator begin() const { return m_node_list.cbegin(); }
    std::list<node>::const_iterator end() const { return m_node_list.cend(); }
    const std::list<node>& get_node_list() const { return m_node_list; }
    size_t max_allocated() const { return m_max_allocated; }
private:
    size_t first_fit(size_t size);
    size_t best_fit(size_t size);
    size_t no_reuse_allocator(size_t size);

    std::list<node> m_node_list;
    size_t m_alignment;
    allocation_scheme m_scheme;
    size_t m_max_allocated;
};
