// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <ngraph/opsets/opset5.hpp>

namespace ov
{
    namespace runtime
    {
        namespace reference
        {
            using custom_evaluate_function =
                std::function<void(const std::shared_ptr<ov::Function>& function,
                                   const HostTensorVector& inputs,
                                   HostTensorVector& outputs)>;
            void tensor_iterator(uint64_t num_iterations,
                                 const std::shared_ptr<Function>& body,
                                 const op::util::OutputDescriptionVector& out_descs,
                                 const op::util::InputDescriptionVector& input_descs,
                                 const HostTensorVector& out,
                                 const HostTensorVector& args,
                                 const custom_evaluate_function& evaluate = nullptr);
        } // namespace reference
    }     // namespace runtime
} // namespace ov
