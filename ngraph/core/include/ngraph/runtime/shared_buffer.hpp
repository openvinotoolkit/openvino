//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
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

#include <cstddef>

#include "ngraph/runtime/aligned_buffer.hpp"

namespace ngraph
{
    namespace runtime
    {
        /// \brief SharedBuffer class to store pointer to pre-acclocated buffer.
        template <typename T>
        class SharedBuffer : public ngraph::runtime::AlignedBuffer
        {
        public:
            SharedBuffer(char* data, size_t size, T& shared_object)
                : _shared_object(shared_object)
            {
                m_allocated_buffer = data;
                m_aligned_buffer = data;
                m_byte_size = size;
            }

            virtual ~SharedBuffer()
            {
                m_aligned_buffer = nullptr;
                m_allocated_buffer = nullptr;
                m_byte_size = 0;
            }

        private:
            T _shared_object;
        };
    }
}
