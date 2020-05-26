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

#include <exception>
#include <sstream>

#include "ngraph/log.hpp"
#include "ngraph/log.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/pass/liveness.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/memory_layout.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

pass::MemoryLayout::MemoryLayout(size_t alignment, bool disable_memory_sharing)
    : m_alignment(alignment)
    , m_disable_memory_sharing(disable_memory_sharing)
{
    if (m_alignment == 0)
    {
        throw invalid_argument("Memory alignment must be > 0");
    }
}

bool pass::MemoryLayout::run_on_function(shared_ptr<Function> function)
{
    MemoryManager mm(m_alignment, m_disable_memory_sharing);
    for (shared_ptr<Node> node : function->get_ordered_ops())
    {
        std::map<descriptor::Tensor*, descriptor::Tensor*> in_place_outputs;
        std::set<const descriptor::Tensor*> reused_inputs;

        if (auto op = std::dynamic_pointer_cast<op::Op>(node))
        {
            // concat and slice in_place_oi should be treated differently
            if (!is_type<op::Concat>(node) && !is_type<op::Slice>(node))
            {
                if (auto op_annotations = op->get_op_annotations())
                {
                    for (auto oi_pair : op_annotations->get_in_place_oi_pairs())
                    {
                        auto output = &node->output(oi_pair.output).get_tensor();
                        auto input = &node->get_input_tensor(oi_pair.input);
                        auto input_node = node->get_input_node_ptr(oi_pair.input);

                        // For destructive kernel, this should be the last use
                        // Non-destructive kernels can pass through if memory sharing is disabled
                        if ((node->liveness_free_list.count(input) != 0 ||
                             is_type<op::GetOutputElement>(node) ||
                             (m_disable_memory_sharing && !oi_pair.destructive &&
                              !is_type<op::v0::Parameter>(input_node) && !is_type<op::v0::Constant>(input_node))) &&
                            node->liveness_new_list.count(output) != 0)

                        {
                            NGRAPH_DEBUG << "Reusing " << input->get_name() << " for "
                                         << output->get_name();
                            in_place_outputs.insert({output, input});
                            reused_inputs.insert(input);
                        }
                    }
                }
            }
        }

        for (descriptor::Tensor* tensor : node->liveness_new_list)
        {
            size_t offset = in_place_outputs.count(tensor)
                                ? in_place_outputs.at(tensor)->get_pool_offset()
                                : mm.allocate(tensor->size());
            tensor->set_pool_offset(offset);
        }

        if (!m_disable_memory_sharing)
        {
            for (const descriptor::Tensor* tensor : node->liveness_free_list)
            {
                if (reused_inputs.count(tensor) == 0)
                {
                    mm.free(tensor->get_pool_offset());
                }
            }
        }
    }
    function->set_temporary_pool_size(mm.max_allocated());

    return false;
}

pass::MemoryManager::node::node(size_t size, block_state state)
    : m_size{size}
    , m_state{state}
{
}

pass::MemoryManager::MemoryManager(size_t alignment, bool disable_memory_reuse)
    : m_alignment{alignment}
    , m_scheme{disable_memory_reuse ? allocation_scheme::NO_REUSE : allocation_scheme::FIRST_FIT}
    , m_max_allocated{0}
{
    if (m_alignment == 0)
    {
        throw invalid_argument("Memory alignment must be > 0");
    }
    m_node_list.emplace_back(numeric_limits<size_t>::max(), block_state::FREE);
}

size_t pass::MemoryManager::allocate(size_t size)
{
    size_t rc = 0;
    switch (m_scheme)
    {
    case allocation_scheme::FIRST_FIT: rc = first_fit(size); break;
    case allocation_scheme::BEST_FIT: rc = best_fit(size); break;
    case allocation_scheme::NO_REUSE: rc = no_reuse_allocator(size); break;
    }
    return rc;
}

size_t pass::MemoryManager::no_reuse_allocator(size_t size)
{
    size_t offset = m_max_allocated;
    m_max_allocated += align(size, m_alignment);
    return offset;
}

size_t pass::MemoryManager::best_fit(size_t size)
{
    size = align(size, m_alignment);
    size_t offset = 0;
    size_t min_delta = numeric_limits<size_t>::max();
    auto best_fit = m_node_list.end();
    size_t best_offset = offset;
    for (auto it = m_node_list.begin(); it != m_node_list.end(); ++it)
    {
        if (it->m_state == block_state::FREE && it->m_size >= size)
        {
            size_t delta = it->m_size - size;
            if (delta < min_delta)
            {
                min_delta = delta;
                best_fit = it;
                best_offset = offset;
            }
        }
        offset += it->m_size;
    }

    if (best_fit == m_node_list.end())
    {
        throw bad_alloc();
    }

    if (min_delta == 0)
    {
        // exact fit
        best_fit->m_state = block_state::ALLOCATED;
    }
    else
    {
        m_node_list.insert(best_fit, node{size, block_state::ALLOCATED});
        best_fit->m_size -= size;
    }
    m_max_allocated = max(m_max_allocated, best_offset + size);

    return best_offset;
}

size_t pass::MemoryManager::first_fit(size_t size)
{
    size = align(size, m_alignment);
    size_t offset = 0;
    bool found = false;
    for (auto it = m_node_list.begin(); it != m_node_list.end(); ++it)
    {
        if (it->m_state == block_state::FREE && it->m_size >= size)
        {
            if (it->m_size > size)
            {
                m_node_list.insert(it, node{size, block_state::ALLOCATED});
                it->m_size -= size;
            }
            else
            {
                // exact fit
                it->m_state = block_state::ALLOCATED;
            }

            found = true;
            break;
        }
        offset += it->m_size;
    }
    if (!found)
    {
        throw bad_alloc();
    }
    m_max_allocated = max(m_max_allocated, offset + size);

    return offset;
}

void pass::MemoryManager::free(size_t offset)
{
    size_t search_offset = 0;
    bool found = false;
    for (auto it = m_node_list.begin(); it != m_node_list.end(); ++it)
    {
        if (offset == search_offset)
        {
            list<node>::iterator it_next = next(it);
            if (it == m_node_list.begin())
            {
                // free the first node in the list
                it->m_state = block_state::FREE;
            }
            else
            {
                // node has predecessor
                list<node>::iterator it_prev = prev(it);
                if (it_prev->m_state == block_state::FREE)
                {
                    it->m_size += it_prev->m_size;
                    m_node_list.erase(it_prev);
                }
            }
            if (it_next != m_node_list.end() && it_next->m_state == block_state::FREE)
            {
                // join this node with next
                it->m_size += it_next->m_size;
                m_node_list.erase(it_next);
            }
            it->m_state = block_state::FREE;
            found = true;
            break;
        }
        search_offset += it->m_size;
    }
    if (!found)
    {
        throw runtime_error("bad free");
    }
}

void pass::MemoryManager::dump(ostream& out)
{
    for (const node& n : m_node_list)
    {
        out << "size=" << n.m_size << ", ";
        out << (n.m_state == block_state::FREE ? "FREE" : "ALLOCATED");
        out << "\n";
    }
}

size_t pass::MemoryManager::align(size_t size, size_t alignment)
{
    if (alignment == 0)
    {
        throw invalid_argument("alignment must be > 0");
    }
    if (size == 0)
    {
        size = alignment;
    }
    else
    {
        auto remainder = size % alignment;
        if (remainder > 0)
        {
            size += (alignment - remainder);
        }
    }
    return size;
}
