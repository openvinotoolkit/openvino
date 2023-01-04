// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "int_backend.hpp"

#include "int_executable.hpp"
#include "ngraph/except.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

runtime::interpreter::INTBackend::INTBackend() {}

runtime::interpreter::INTBackend::INTBackend(const vector<string>& unsupported_op_name_list)
    : m_unsupported_op_name_list{unsupported_op_name_list.begin(), unsupported_op_name_list.end()} {}

shared_ptr<runtime::Tensor> runtime::interpreter::INTBackend::create_tensor() {
    return make_shared<runtime::HostTensor>();
}

shared_ptr<runtime::Tensor> runtime::interpreter::INTBackend::create_tensor(const element::Type& type,
                                                                            const Shape& shape) {
    return make_shared<runtime::HostTensor>(type, shape);
}

shared_ptr<runtime::Tensor> runtime::interpreter::INTBackend::create_dynamic_tensor(const element::Type& type,
                                                                                    const PartialShape& pshape) {
    return make_shared<runtime::HostTensor>(type, pshape);
}

shared_ptr<runtime::Tensor> runtime::interpreter::INTBackend::create_tensor(const element::Type& type,
                                                                            const Shape& shape,
                                                                            void* memory_pointer) {
    return make_shared<runtime::HostTensor>(type, shape, memory_pointer);
}

shared_ptr<runtime::Executable> runtime::interpreter::INTBackend::compile(shared_ptr<Function> function,
                                                                          bool enable_performance_collection) {
    return make_shared<INTExecutable>(function, enable_performance_collection);
}

bool runtime::interpreter::INTBackend::is_supported(const Node& node) const {
    return m_unsupported_op_name_list.find(node.description()) == m_unsupported_op_name_list.end();
}

bool runtime::interpreter::INTBackend::set_config(const map<string, string>& config, string& error) {
    bool rc = false;
    auto it = config.find("test_echo");
    error = "";
    if (it != config.end()) {
        error = it->second;
        rc = true;
    }
    return rc;
}
