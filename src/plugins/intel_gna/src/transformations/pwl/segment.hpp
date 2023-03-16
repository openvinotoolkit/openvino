// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace ov {
namespace intel_gna {
namespace pass {
namespace pwl {
struct Segment {
    Segment() = default;
    Segment(double im, double ib, double ialpha, double ibeta = 0) : m(im), b(ib), alpha(ialpha), beta(ibeta) {}
    double m;
    double b;
    double alpha;
    double beta;
};  // struct Segment

}  // namespace pwl
}  // namespace pass
}  // namespace intel_gna
}  // namespace ov