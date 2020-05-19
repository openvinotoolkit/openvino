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

#include "ngraph/serializer.hpp"

std::string ngraph::serialize(std::shared_ptr<ngraph::Function> func, size_t indent)
{
    throw std::runtime_error("serializer disabled in build");
}

void ngraph::serialize(const std::string& path,
                       std::shared_ptr<ngraph::Function> func,
                       size_t indent)
{
    throw std::runtime_error("serializer disabled in build");
}

void ngraph::serialize(std::ostream& out, std::shared_ptr<ngraph::Function> func, size_t indent)
{
    throw std::runtime_error("serializer disabled in build");
}

std::shared_ptr<ngraph::Function> ngraph::deserialize(std::istream& in)
{
    throw std::runtime_error("serializer disabled in build");
}

std::shared_ptr<ngraph::Function> ngraph::deserialize(const std::string& str)
{
    throw std::runtime_error("serializer disabled in build");
}

void ngraph::set_serialize_output_shapes(bool enable)
{
    throw std::runtime_error("serializer disabled in build");
}
