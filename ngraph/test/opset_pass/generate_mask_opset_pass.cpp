#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/opset0_downgrade.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(opset_transform, opset1_generate_mask_downgrade_pass)
{
    Shape scalar{};
    const unsigned int seed = 777;
    auto training = op::Constant::create(element::f32, Shape{}, {1});
    auto result_shape = op::Constant::create<int64_t>(element::i64, Shape{2}, {1, 128});
    auto gen_mask =
        make_shared<op::v1::GenerateMask>(training, result_shape, element::f32, seed, 0.5, false);
    auto gen_mask2 =
        make_shared<op::v1::GenerateMask>(training, result_shape, element::f32, seed, 0.5, false);
    auto f = make_shared<Function>(NodeVector{gen_mask, gen_mask2}, ParameterVector{});

    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<pass::Opset0Downgrade>();
    pass_manager.run_passes(f);

    auto generate_mask_v0 = as_type_ptr<op::v0::GenerateMask>(
        f->get_results().at(0)->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(generate_mask_v0);
    EXPECT_EQ(generate_mask_v0->get_mask_shape(), (Shape{1, 128}));
}
