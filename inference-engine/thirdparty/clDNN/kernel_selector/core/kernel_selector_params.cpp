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

#include "kernel_selector_params.h"
#include "kernel_selector_common.h"
#include <sstream>
 
namespace kernel_selector {

    std::string Params::to_string() const
    {
        std::stringstream s;
        s << toString(kType);
        return s.str();
    }

    std::string base_params::to_string() const
    {
        std::stringstream s;
        s << Params::to_string() << "_";
        s << toString(activationParams) << "_";
        s << toString(activationFunc) << "_";

        for (auto input : inputs)
        {
            s << toString(input) << "_";
        }
        s << toString(output);

        return s.str();
    }

}