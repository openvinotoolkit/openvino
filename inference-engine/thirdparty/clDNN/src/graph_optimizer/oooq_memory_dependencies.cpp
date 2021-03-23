/*
// Copyright (c) 2018-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "pass_manager.h"
#include "program_node.h"
#include "layout_optimizer.h"
#include "program_impl.h"
#include "program_helpers.h"
#include "cldnn_itt.h"
#include <vector>
#include <memory>
#include <list>
#include <map>
#include <set>
#include <algorithm>

using namespace cldnn;

namespace {

class bits_64 {
public:
    explicit bits_64(size_t size, bool set = false) : storage((size / 64) + 1, (set ? ~0ULL : 0ULL)) {}
    bool is_set(size_t idx) const {
        size_t storage_idx = idx >> 6;
        uint64_t mask = 1ULL << (idx & 0x3F);
        return storage[storage_idx] & mask;
    }
    void set(size_t idx) {
        size_t storage_idx = idx >> 6;
        uint64_t mask = 1ULL << (idx & 0x3F);
        storage[storage_idx] |= mask;
    }
    bool _or(const bits_64& that) {
        bool changed = false;
        size_t sz = std::min(storage.size(), that.storage.size());
        for (size_t i = 0; i < sz; i++) {
            uint64_t myval = storage[i];
            uint64_t thatval = myval | that.storage[i];
            bool local_change = myval != thatval;
            changed |= local_change;
            if (local_change)
                storage[i] = thatval;
        }
        return changed;
    }
#if 0
    void dump(std::ostream& s, size_t cols) {
        size_t idx = 0;
        size_t rows = (storage.size() * 64) / cols;

        s << storage.size() << " items" << std::endl;
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                s << is_set(idx);
                idx++;
            }
            s << std::endl;
        }
    }
#endif

protected:
    std::vector<uint64_t> storage;
};

}  // namespace

void oooq_memory_dependencies::run(program_impl& p) {
    OV_ITT_SCOPED_TASK(itt::domains::CLDNN, "CLDNN::pass::OooqMemoryDependencies");
    // For oooq memory dependencies nodes A and B can't share memory if
    // processing_num(A) < processing_num(B) and there is no path from A to B.
    // Assuming precalculation of reachability this function has complexity O(N^2 log N).

    // First create transitive closure of the graph,
    // giving us mapping of node to set of all users that can be reached from this node.
    auto& processing_order = p.get_processing_order();

    // maps program nodes to bimap vector ids
    auto user_map = std::map<program_node*, unsigned int>();
    unsigned int i = 0;
    for (auto node : processing_order) {
        user_map[node] = i++;
    }

    unsigned int num_nodes = static_cast<unsigned int>(user_map.size());

    // full cross ref [node<->node] bitmap.
    // every node has a bit array assigned to it
    // users or the node are marked with 1 bit in this array
    std::vector<bits_64> user_bitmap(num_nodes, bits_64(num_nodes));

    // init bitmaps from direct node users
    for (const auto& node : user_map) {
        for (const auto& user : node.first->get_users()) {
            user_bitmap[node.second].set(user_map.at(user));
        }
    }

    // Iteratively extend the users set by adding closure over existing users untill no change occurs.
    bool changed = true;
    while (changed) {
        changed = false;
        for (unsigned int n = 0; n < num_nodes; n++) {
            auto& users = user_bitmap[n];

            // iterate over all users
            for (unsigned int user_id = 0; user_id < num_nodes; user_id++) {
                // if we have this user set, then add its sub-users to the map
                if (users.is_set(user_id)) {
                    changed |= users._or(user_bitmap[user_id]);
                }
            }
        }
    }

    // Connection query:
    auto are_connected = [&](unsigned int A, unsigned int B) {
        return user_bitmap[A].is_set(B);
    };

    unsigned int A = 0;
    auto itr_A = processing_order.begin();

    while (itr_A != processing_order.end()) {
        unsigned int B = ++A;
        auto itr_B = ++itr_A;
        while (itr_B != processing_order.end()) {
            if (!are_connected(A, B)) {
                add_memory_dependency(*itr_A, *itr_B);
                add_memory_dependency(*itr_B, *itr_A);
            }
            itr_B++;
            B++;
        }
    }
}
