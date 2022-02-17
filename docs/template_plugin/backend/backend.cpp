// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backend.hpp"

#include <sstream>

#include "int_backend.hpp"
#include "ngraph/file_util.hpp"
#include "ngraph/util.hpp"
#include "openvino/util/file_util.hpp"

using namespace std;
using namespace ngraph;

runtime::Backend::~Backend() {}

std::shared_ptr<runtime::Backend> runtime::Backend::create() {
    auto inner_backend = make_shared<interpreter::INTBackend>();

    return inner_backend;
}

std::shared_ptr<ngraph::runtime::Tensor> runtime::Backend::create_dynamic_tensor(
    const ngraph::element::Type& /* element_type */,
    const PartialShape& /* shape */) {
    throw std::invalid_argument("This backend does not support dynamic tensors");
}

bool runtime::Backend::is_supported(const Node& /* node */) const {
    // The default behavior is that a backend does not support any ops. If this is not the case
    // then override this method and enhance.
    return false;
}

std::shared_ptr<runtime::Executable> runtime::Backend::load(istream& /* input_stream */) {
    throw runtime_error("load operation unimplemented.");
}

bool runtime::Backend::set_config(const map<string, string>& /* config */, string& error) {
    error = "set_config not supported";
    return false;
}
