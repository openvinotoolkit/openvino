// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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
