// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace ov {
namespace intel_gna {
namespace pass {
namespace pwl {
class Function {
public:
    virtual ~Function() = default;
    virtual double get_value(double x) const = 0;
    virtual double get_first_derivative(double x) const = 0;
};

}  // namespace pwl
}  // namespace pass
}  // namespace intel_gna
}  // namespace ov