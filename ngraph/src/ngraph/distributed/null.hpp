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

#include <cstdio>
#include <string>

#include "ngraph/distributed.hpp"

namespace ngraph
{
    namespace distributed
    {
        class Null : public DistributedInterface
        {
            const std::string& get_name() const override;
            int get_size() override;
            int get_rank() override;
            void all_reduce(void* in,
                            void* out,
                            element::Type_t element_type,
                            reduction::Type reduce_type,
                            size_t count) override;

            void broadcast(void* in,
                           element::Type_t element_type,
                           size_t count,
                           int root_id) override;

            void recv(void* in, element::Type_t element_type, size_t count, int src_id) override;

            void send(const void* in,
                      element::Type_t element_type,
                      size_t count,
                      int dest_id) override;

        protected:
            std::string m_name{"NULL"};
        };
    }
}
