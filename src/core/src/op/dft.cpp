//*****************************************************************************
// Copyright 2017-2022 Intel Corporation
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

#include "openvino/op/dft.hpp"

#include <algorithm>
#include <memory>

#include "itt.hpp"

namespace ov {
op::v7::DFT::DFT(const Output<Node>& data, const Output<Node>& axes) : FFTBase(data, axes) {
    constructor_validate_and_infer_types();
}

op::v7::DFT::DFT(const Output<Node>& data, const Output<Node>& axes, const Output<Node>& signal_size)
    : FFTBase(data, axes, signal_size) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> op::v7::DFT::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v7_DFT_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    NODE_VALIDATION_CHECK(this, new_args.size() == 2 || new_args.size() == 3, "Number of inputs must be 2 or 3");

    if (new_args.size() == 2) {
        return std::make_shared<op::v7::DFT>(new_args.at(0), new_args.at(1));
    }

    return std::make_shared<op::v7::DFT>(new_args.at(0), new_args.at(1), new_args.at(2));
}
}  // namespace ov
