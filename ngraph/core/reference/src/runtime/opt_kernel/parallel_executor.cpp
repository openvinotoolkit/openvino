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

#include "ngraph/runtime/opt_kernel/parallel_executor.hpp"
#include "ngraph/env_util.hpp"

using namespace ngraph;

bool runtime::detail::forced_single_threaded_execution()
{
    return getenv_bool("REF_SINGLE_THREADED", false);
}

uint64_t runtime::detail::parallelism_threshold()
{
    const int32_t DEFAULT_THRESHOLD = 1000000;
    const auto t = getenv_int("REF_THRESHOLD", DEFAULT_THRESHOLD);
    if (t < 0)
    {
        return DEFAULT_THRESHOLD;
    }
    else
    {
        return t;
    }
}

size_t runtime::detail::parallel_tasks_number()
{
    const size_t DEFAULT_TASKS_NUMBER = 4;
    const size_t MAX_TASKS_NUMBER = 128;
    const auto c = getenv_int("REF_TASKS_NUMBER", DEFAULT_TASKS_NUMBER);
    if (c < 1)
    {
        return DEFAULT_TASKS_NUMBER;
    }
    else if (c > MAX_TASKS_NUMBER)
    {
        return MAX_TASKS_NUMBER;
    }
    else
    {
        return static_cast<size_t>(c);
    }
}
