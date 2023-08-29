// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace ov {
namespace test {
namespace utils {

enum ComparisonTypes {
    EQUAL,
    NOT_EQUAL,
    IS_FINITE,
    IS_INF,
    IS_NAN,
    LESS,
    LESS_EQUAL,
    GREATER,
    GREATER_EQUAL
};

enum ConversionTypes {
    CONVERT,
    CONVERT_LIKE
};

enum ReductionType {
    Mean,
    Max,
    Min,
    Prod,
    Sum,
    LogicalOr,
    LogicalAnd,
    L1,
    L2
};

}  // namespace utils
}  // namespace test
}  // namespace ov
