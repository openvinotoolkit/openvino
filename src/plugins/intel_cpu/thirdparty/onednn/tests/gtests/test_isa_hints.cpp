/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <map>
#include <set>
#include <utility>

#include "tests/test_isa_common.hpp"

#include "gtest/gtest.h"

#include "oneapi/dnnl/dnnl.hpp"
#include "src/cpu/x64/cpu_isa_traits.hpp"

namespace dnnl {

TEST(isa_hints_test_t, TestISAHints) {
    using impl::cpu::x64::cpu_isa_t;
    auto hints = cpu_isa_hints::prefer_ymm;

    // Use soft version of mayiuse that allows resetting the cpu_isa_hints
    const bool test_flag = true;

    std::map<cpu_isa_t, bool> compat_before_hint;

    for (auto isa : cpu_isa_all()) {
        const auto &internal_isa_set = masked_internal_cpu_isa(isa);
        for (auto internal_isa : internal_isa_set) {
            compat_before_hint[internal_isa] = mayiuse(internal_isa, test_flag);
        }
    }

    std::map<std::pair<cpu_isa_t, cpu_isa_t>, std::pair<bool, bool>>
            masked_compat_before_hint;
    for (const auto &isa_pair : hints_masked_internal_cpu_isa(hints)) {
        masked_compat_before_hint[isa_pair]
                = {mayiuse(isa_pair.first, test_flag),
                        mayiuse(isa_pair.second, test_flag)};
    }

    status st = set_cpu_isa_hints(hints);
    // status::unimplemented if the feature was disabled at compile time
    if (st == status::unimplemented) return;

    ASSERT_TRUE(st == status::success);

    for (auto isa : cpu_isa_all()) {
        const auto &internal_isa_set = masked_internal_cpu_isa(isa);
        for (auto internal_isa : internal_isa_set) {
            // ISA specific hint will not change the non-hint-complying ISA
            ASSERT_TRUE(compat_before_hint[internal_isa]
                    == mayiuse(internal_isa, test_flag));
        }
    }

    for (const auto &isa_pair : hints_masked_internal_cpu_isa(hints)) {
        auto compat_pair = masked_compat_before_hint[isa_pair];
        // isa_pair = {isa_no_hint, isa_hints} is a pair of two ISA that are
        // only distinguished w.r.t. the CPU ISA hints. Also compat_pair
        // verifies ISA use before the CPU_ISA hints is applied i.e.
        //
        // {compat_pair.frst, compat_pair.second} :=
        //     { mayiuse(isa_no_hints), mayiuse(isa_hints) }

        // CPU_ISA_HINT will not affect the availability of isa_no_hint
        ASSERT_TRUE(mayiuse(isa_pair.first, test_flag) == compat_pair.first);
        // Without proper CPU_ISA_HINT isa_hints will not be available
        ASSERT_FALSE(compat_pair.second);

        // With proper CPU_ISA_HINT isa_hints is available if and only if
        // isa_no_hints is available
        ASSERT_TRUE(mayiuse(isa_pair.first, test_flag)
                == mayiuse(isa_pair.second, test_flag));
    }
}

} // namespace dnnl
