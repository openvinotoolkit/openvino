// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cfenv>

#include "ngraph/check.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            /// Object of this class will temporary set floating-point rounding mode and
            /// restor previouse one during destruction (end of life/scope)
            class FeRound final
            {
            public:
                explicit FeRound(int new_round_mode)
                    : previous_round_mode(std::fegetround())
                {
                    NGRAPH_CHECK(previous_round_mode != -1, "Don't know FLT_ROUNDS mode");
                    const auto set_status = std::fesetround(new_round_mode);
                    NGRAPH_CHECK(set_status == 0, "Can't set FLT_ROUNDS mode");
                }

                FeRound(const FeRound&) = delete;
                FeRound& operator=(const FeRound&) = delete;
                FeRound(FeRound&&) = delete;
                FeRound& operator=(FeRound&&) = delete;

                ~FeRound() { std::fesetround(previous_round_mode); }

            private:
                int previous_round_mode;
            };
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
