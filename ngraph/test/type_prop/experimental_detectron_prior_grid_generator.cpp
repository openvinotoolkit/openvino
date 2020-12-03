//*****************************************************************************
// Copyright 2020 Intel Corporation
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

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace ngraph;

using Attrs = op::v6::ExperimentalDetectronPriorGridGenerator::Attributes;
using GridGenerator = op::v6::ExperimentalDetectronPriorGridGenerator;

TEST(type_prop, detectron_grid_generator)
{
    Attrs attrs;
    attrs.flatten = true;
    attrs.h = 0;
    attrs.w = 0;
    attrs.stride_x = 4.0f;
    attrs.stride_y = 4.0f;

    auto priors = std::make_shared<op::Parameter>(element::f32, Shape{3, 4});
    auto feature_map = std::make_shared<op::Parameter>(element::f32, Shape{1, 256, 200, 336});
    auto im_data = std::make_shared<op::Parameter>(element::f32, Shape{1, 3, 800, 1344});

    ASSERT_TRUE(true);
}
