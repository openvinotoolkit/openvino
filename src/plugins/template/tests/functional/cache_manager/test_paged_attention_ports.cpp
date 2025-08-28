#include <gtest/gtest.h>
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/paged_attention.hpp"
#include "openvino/core/model.hpp"

using namespace ov;

TEST(CacheManagerGraph, PagedAttentionPortsContract) {
    auto q  = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 8});
    auto k  = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 8});
    auto v  = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 8});
    auto kc = std::make_shared<op::v0::Parameter>(element::f32, Shape{2, 1, 16, 4});
    auto vc = std::make_shared<op::v0::Parameter>(element::f32, Shape{2, 1, 16, 4});
    auto pl = std::make_shared<op::v0::Parameter>(element::i32, Shape{1});
    auto sb = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});
    auto bi = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});
    auto bib= std::make_shared<op::v0::Parameter>(element::i32, Shape{2});
    auto sc = std::make_shared<op::v0::Parameter>(element::f32, Shape{});
    auto sw = std::make_shared<op::v0::Parameter>(element::i32, Shape{});
    auto as = std::make_shared<op::v0::Parameter>(element::f32, Shape{1});
    auto mcl= std::make_shared<op::v0::Parameter>(element::i32, Shape{});

    auto pat = std::make_shared<op::PagedAttentionExtension>(
        q, k, v, kc, vc, pl, sb, bi, bib, sc, sw, as, mcl);

    ASSERT_EQ(pat->get_output_size(), 2);
    auto scores_out = pat->output(1);

    auto res_scores = std::make_shared<op::v0::Result>(scores_out);
    auto res_vals   = std::make_shared<op::v0::Result>(pat->output(0));
    auto model = std::make_shared<Model>(ResultVector{res_scores, res_vals},
                                         ParameterVector{q,k,v,kc,vc,pl,sb,bi,bib,sc,sw,as,mcl});

    auto in3 = pat->input_value(3).get_node_shared_ptr();
    auto in4 = pat->input_value(4).get_node_shared_ptr();
    ASSERT_TRUE(std::dynamic_pointer_cast<op::v0::Parameter>(in3) != nullptr);
    ASSERT_TRUE(std::dynamic_pointer_cast<op::v0::Parameter>(in4) != nullptr);

    auto src = res_scores->input_value(0);
    ASSERT_EQ(src.get_index(), 1u);
}
