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

#include <memory>
#include <unordered_map>

namespace ngraph
{
    class Function;

    namespace autodiff
    {
        /// \brief Returns a FunctionSpec for the backprop derivative of its argument.
        /// \param f is f(X_i...)
        /// \returns f'(X_i..., c) where f'(x_i, ..., c)_j is backprop for X_j
        std::shared_ptr<Function> backprop_function(const std::shared_ptr<Function>& f);
    }
}
