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

#include "ngraph/type/element_type.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            void gather_tree(const char* step_ids,
                             const char* parent_ids,
                             const char* max_seq_len,
                             const char* end_token,
                             char* out,
                             const Shape& step_ids_shape,
                             const Shape& parent_ids_shape,
                             const Shape& max_seq_len_shape,
                             const Shape& end_token_shape,
                             const element::Type& type);
        }
    }
}