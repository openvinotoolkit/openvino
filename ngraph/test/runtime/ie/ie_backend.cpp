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

#include "ie_backend.hpp"

#include "backend_manager.hpp"
#include "ie_backend_visibility.hpp"
#include "ie_executable.hpp"
#include "ie_tensor.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/opsets/opset.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

runtime::ie::IE_Backend::IE_Backend(const string& configuration_string)
{
    string config = configuration_string;
    // Get device name, after colon if present: IE:CPU -> CPU
    auto separator = config.find(":");
    if (separator != config.npos)
    {
        config = config.substr(separator + 1);
    }
    m_device = config;
}

shared_ptr<runtime::Executable> runtime::ie::IE_Backend::compile(shared_ptr<Function> func, bool)
{
    return make_shared<IE_Executable>(func, m_device);
}

bool runtime::ie::IE_Backend::is_supported(const Node& node) const
{
    const auto& opset = get_opset1();
    return opset.contains_op_type(&node);
}

shared_ptr<runtime::Tensor>
    runtime::ie::IE_Backend::create_dynamic_tensor(const element::Type& type,
                                                   const PartialShape& shape)
{
    return make_shared<IETensor>(type, shape);
}

shared_ptr<runtime::Tensor> runtime::ie::IE_Backend::create_tensor()
{
    throw runtime_error("IE_Backend::create_tensor() not supported");
}

shared_ptr<runtime::Tensor>
    runtime::ie::IE_Backend::create_tensor(const element::Type& element_type, const Shape& shape)
{
    return make_shared<IETensor>(element_type, shape);
}

shared_ptr<runtime::Tensor> runtime::ie::IE_Backend::create_tensor(
    const element::Type& element_type, const Shape& shape, void* data)
{
    shared_ptr<runtime::Tensor> tensor = make_shared<IETensor>(element_type, shape);
    if (tensor == nullptr)
        throw runtime_error("Cannot create IETensor!");
    tensor->write(data, shape_size(shape) * element_type.size());
    return tensor;
}

extern "C" IE_BACKEND_API void ngraph_register_ie_backend()
{
    runtime::BackendManager::register_backend(
        "IE", [](const string& config) { return make_shared<runtime::ie::IE_Backend>(config); });
}
