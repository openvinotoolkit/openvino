/*
// Copyright (c) 2016 Intel Corporation
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

#include "primitive_gpu_base.h"
#include <list>

namespace cldnn {
namespace gpu {

bool is_user_cpu(const program_node* user) {
    if (user->can_be_optimized()) {
        auto users = user->get_users();
        for (const auto& u : users) {
            if (is_user_cpu(u)) {
                return true;
            }
        }
        return false;
    }
    return user->get_selected_impl()->is_cpu();
}

bool is_any_user_cpu(const std::list<const program_node*>& users) {
    for (const auto& user : users) {
        if (is_user_cpu(user))
            return true;
    }
    return false;
}
}  // namespace gpu
}  // namespace cldnn