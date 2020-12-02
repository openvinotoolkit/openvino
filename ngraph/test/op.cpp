//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "ngraph/graph_util.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/variant.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace std;
using namespace ngraph;

TEST(op, is_op)
{
    auto arg0 = make_shared<op::Parameter>(element::Type_t::f32, Shape{1});
    ASSERT_NE(nullptr, arg0);
    EXPECT_TRUE(op::is_parameter(arg0));
}

TEST(op, is_parameter)
{
    auto arg0 = make_shared<op::Parameter>(element::Type_t::f32, Shape{1});
    ASSERT_NE(nullptr, arg0);
    auto t0 = make_shared<op::Add>(arg0, arg0);
    ASSERT_NE(nullptr, t0);
    EXPECT_FALSE(op::is_parameter(t0));
}

TEST(op, provenance_tag)
{
    auto node = make_shared<op::Parameter>(element::Type_t::f32, Shape{1});
    auto tag1 = "parameter node";
    auto tag2 = "f32 node";
    node->add_provenance_tag(tag1);
    node->add_provenance_tag(tag2);

    node->remove_provenance_tag(tag1);

    auto tags = node->get_provenance_tags();
    ASSERT_TRUE(tags.find(tag1) == tags.end());
    ASSERT_TRUE(tags.find(tag2) != tags.end());
}

struct Ship
{
    std::string name;
    int16_t x;
    int16_t y;
};

namespace ngraph
{
    template <>
    class VariantWrapper<Ship> : public VariantImpl<Ship>
    {
    public:
        static constexpr VariantTypeInfo type_info{"Variant::Ship", 0};
        const VariantTypeInfo& get_type_info() const override { return type_info; }
        VariantWrapper(const value_type& value)
            : VariantImpl<value_type>(value)
        {
        }
    };

    constexpr VariantTypeInfo VariantWrapper<Ship>::type_info;
}

TEST(op, variant)
{
    shared_ptr<Variant> var_std_string = make_shared<VariantWrapper<std::string>>("My string");
    ASSERT_TRUE((is_type<VariantWrapper<std::string>>(var_std_string)));
    EXPECT_EQ((as_type_ptr<VariantWrapper<std::string>>(var_std_string)->get()), "My string");

    shared_ptr<Variant> var_int64_t = make_shared<VariantWrapper<int64_t>>(27);
    ASSERT_TRUE((is_type<VariantWrapper<int64_t>>(var_int64_t)));
    EXPECT_FALSE((is_type<VariantWrapper<std::string>>(var_int64_t)));
    EXPECT_EQ((as_type_ptr<VariantWrapper<int64_t>>(var_int64_t)->get()), 27);

    shared_ptr<Variant> var_ship = make_shared<VariantWrapper<Ship>>(Ship{"Lollipop", 3, 4});
    ASSERT_TRUE((is_type<VariantWrapper<Ship>>(var_ship)));
    Ship& ship = as_type_ptr<VariantWrapper<Ship>>(var_ship)->get();
    EXPECT_EQ(ship.name, "Lollipop");
    EXPECT_EQ(ship.x, 3);
    EXPECT_EQ(ship.y, 4);

    auto node = make_shared<op::Parameter>(element::Type_t::f32, Shape{1});
    node->get_rt_info()["A"] = var_ship;
    auto node_var_ship = node->get_rt_info().at("A");
    ASSERT_TRUE((is_type<VariantWrapper<Ship>>(node_var_ship)));
    Ship& node_ship = as_type_ptr<VariantWrapper<Ship>>(node_var_ship)->get();
    EXPECT_EQ(&node_ship, &ship);
}
