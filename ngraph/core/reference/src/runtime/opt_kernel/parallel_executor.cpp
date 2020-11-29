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

bool runtime::parallel::forced_single_threaded_execution()
{
    return getenv_bool("REF_SINGLE_THREADED", false);
}

size_t runtime::parallel::parallel_tasks_count()
{
    const auto c = getenv_int("REF_TASKS_NUMBER", 4);
    if (c < 1)
    {
        return 4;
    }
    else
    {
        return static_cast<size_t>(c);
    }
}
