// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <ngraph/opsets/opset5.hpp>

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            void loop(const std::shared_ptr<Function>& body,
                      const op::util::OutputDescriptionVector& out_descs,
                      const op::util::InputDescriptionVector& input_descs,
                      const opset5::Loop::SpecialBodyPorts& special_ports,
                      const HostTensorVector& out,
                      const HostTensorVector& args);
        }
    } // namespace runtime
} // namespace ngraph
