// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/attribute_adapter.hpp"

#include <vector>

#include "ngraph/axis_set.hpp"
#include "ngraph/coordinate.hpp"
#include "ngraph/coordinate_diff.hpp"
#include "ngraph/partial_shape.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/strides.hpp"
#include "ngraph/type.hpp"
#include "ngraph/type/element_type.hpp"
#include "openvino/core/any.hpp"

using namespace std;
using namespace ngraph;

namespace ov {
BWDCMP_RTTI_DEFINITION(AttributeAdapter<float>);
BWDCMP_RTTI_DEFINITION(AttributeAdapter<double>);
BWDCMP_RTTI_DEFINITION(AttributeAdapter<string>);
BWDCMP_RTTI_DEFINITION(AttributeAdapter<bool>);
BWDCMP_RTTI_DEFINITION(AttributeAdapter<int8_t>);
BWDCMP_RTTI_DEFINITION(AttributeAdapter<int16_t>);
BWDCMP_RTTI_DEFINITION(AttributeAdapter<int32_t>);
BWDCMP_RTTI_DEFINITION(AttributeAdapter<int64_t>);
BWDCMP_RTTI_DEFINITION(AttributeAdapter<uint8_t>);
BWDCMP_RTTI_DEFINITION(AttributeAdapter<uint16_t>);
BWDCMP_RTTI_DEFINITION(AttributeAdapter<uint32_t>);
BWDCMP_RTTI_DEFINITION(AttributeAdapter<uint64_t>);
#ifdef __APPLE__
// size_t is not uint_64t on OSX
BWDCMP_RTTI_DEFINITION(AttributeAdapter<size_t>);
BWDCMP_RTTI_DEFINITION(AttributeAdapter<vector<size_t>>);
#endif
BWDCMP_RTTI_DEFINITION(AttributeAdapter<vector<int8_t>>);
BWDCMP_RTTI_DEFINITION(AttributeAdapter<vector<int16_t>>);
BWDCMP_RTTI_DEFINITION(AttributeAdapter<vector<int32_t>>);
BWDCMP_RTTI_DEFINITION(AttributeAdapter<vector<int64_t>>);
BWDCMP_RTTI_DEFINITION(AttributeAdapter<vector<uint8_t>>);
BWDCMP_RTTI_DEFINITION(AttributeAdapter<vector<uint16_t>>);
BWDCMP_RTTI_DEFINITION(AttributeAdapter<vector<uint32_t>>);
BWDCMP_RTTI_DEFINITION(AttributeAdapter<vector<uint64_t>>);
BWDCMP_RTTI_DEFINITION(AttributeAdapter<vector<float>>);
BWDCMP_RTTI_DEFINITION(AttributeAdapter<vector<double>>);
BWDCMP_RTTI_DEFINITION(AttributeAdapter<vector<string>>);
}  // namespace ov
