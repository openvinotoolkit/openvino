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

#include <vector>

#include "ngraph/except.hpp"
#include "ngraph/ngraph_visibility.hpp"

namespace ngraph
{
    namespace op
    {
        namespace util
        {
            struct oi_pair
            {
                size_t output;
                size_t input;
                bool destructive;
            };

            /// \brief Base class for annotations added to graph ops
            class NGRAPH_API OpAnnotations
            {
            public:
                virtual ~OpAnnotations() = default;

                void add_in_place_oi_pair(const struct oi_pair& oi)
                {
                    for (auto e : m_in_place_oi_pairs)
                    {
                        if (e.input == oi.input || e.output == oi.output)
                        {
                            throw ngraph_error("In_place hint conflicts with an existing entry");
                        }
                    }
                    m_in_place_oi_pairs.emplace_back(oi);
                }

                const std::vector<struct oi_pair>& get_in_place_oi_pairs() const
                {
                    return m_in_place_oi_pairs;
                }
                bool is_cacheable() const { return m_cacheable; }
                void set_cacheable(bool val) { m_cacheable = val; }

            private:
                // map of output-input pairs for which in-place computation is valid
                std::vector<struct oi_pair> m_in_place_oi_pairs;

                bool m_cacheable = false;
            };
        }
    }
}
