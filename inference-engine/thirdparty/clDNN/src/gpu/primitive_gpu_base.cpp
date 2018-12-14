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

#include "detection_output_inst.h"
#include "proposal_inst.h"
#include "prior_box_inst.h"

namespace cldnn {
    namespace gpu {

        bool is_any_user_cpu(const std::list<const program_node*>& users)
        {
            for (const auto& user : users)
            {
                if (user->type() == detection_output::type_id() ||
                    user->type() == prior_box::type_id() ||
                    user->type() == proposal::type_id())
                {
                    return true;
                }
            }
            return false;
        }
    }
}