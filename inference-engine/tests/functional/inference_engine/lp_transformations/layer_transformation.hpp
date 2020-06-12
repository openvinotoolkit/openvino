// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include "common_test_utils/test_common.hpp"
#include <string>
#include <sstream>
#include <fstream>
#include <memory>
#include <queue>
#include <map>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include <transformations/utils/utils.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/convert_opset1_to_legacy/conv_bias_fusion.hpp>
#include <ngraph/pass/visualize_tree.hpp>

#include "../transformations/ngraph_test_utils.hpp"
#include "ngraph_functions/low_precision_transformations/depth_to_space_function.hpp"

using namespace testing;

typedef std::tuple<
    ngraph::element::Type,
    ngraph::Shape> DepthToSpaceTestsParams;

class LayerTransformation : public CommonTestUtils::TestsCommon {
public:
};
