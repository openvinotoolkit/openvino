// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "transformations/fp16_compression/disable_fp16_comp_rope_angles.hpp"

#include <cmath>
#include <map>
#include <memory>
#include <sstream>
#include <tuple>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/if.hpp"
#include "openvino/op/loop.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/sin.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/multi_subgraph_base.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/convert_precision.hpp"
#include "transformations/rt_info/disable_precision_conversion.hpp"

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v3 = ov::op::v3;

namespace {

// How the exporters build the angles from the integer positions
enum class AngleForm {
    MatMulTranspose,       // Llama / Qwen3: inv_freq[1, D/2, 1] @ positions[1, 1, L] -> Transpose
    MatMulPositionsFirst,  // torch.outer: Unsqueeze(positions.float())[1, L, 1] @ inv_freq[1, 1, D/2]
    Multiply,              // transformers main: positions[..., None].float() * inv_freq
    ModernBert,            // torch.export of ModernBERT: Broadcast -> Reshape -> MatMul -> Reshape -> Transpose
};

constexpr size_t half_dim = 4;

std::vector<float> inv_freq() {
    std::vector<float> values(half_dim);
    for (size_t i = 0; i < half_dim; i++) {
        values[i] = 1.0f / std::pow(10000.0f, static_cast<float>(2 * i) / static_cast<float>(2 * half_dim));
    }
    return values;
}

std::shared_ptr<v0::Constant> i64_constant(const std::vector<int64_t>& values) {
    return v0::Constant::create(ov::element::i64, ov::Shape{values.size()}, values);
}

// position_ids -> Unsqueeze -> Convert(f32) -> angles -> [Concat(x, x)] -> Sin, Cos
struct RopeAngles {
    std::shared_ptr<v0::Parameter> position_ids;
    std::shared_ptr<v0::Parameter> q;
    std::vector<std::shared_ptr<ov::Node>> chain;  // the nodes the pass keeps in f32, Sin and Cos excluded
    std::shared_ptr<v0::Sin> sin;
    std::shared_ptr<v0::Cos> cos;
};

RopeAngles make_angles(AngleForm form, bool with_concat, const ov::PartialShape& positions_shape) {
    RopeAngles r;
    r.position_ids = std::make_shared<v0::Parameter>(ov::element::i64, positions_shape);
    const auto unsqueeze_axis = form == AngleForm::MatMulTranspose ? 1 : -1;
    std::shared_ptr<ov::Node> convert;
    if (form == AngleForm::MatMulPositionsFirst) {
        // torch.outer: the float positions are unsqueezed
        convert = std::make_shared<v0::Convert>(r.position_ids, ov::element::f32);
        r.chain.push_back(convert);
        convert = std::make_shared<v0::Unsqueeze>(convert, i64_constant({unsqueeze_axis}));
    } else {
        auto positions = std::make_shared<v0::Unsqueeze>(r.position_ids, i64_constant({unsqueeze_axis}));
        convert = std::make_shared<v0::Convert>(positions, ov::element::f32);
    }
    r.chain.push_back(convert);
    std::shared_ptr<ov::Node> angles;
    switch (form) {
    case AngleForm::MatMulTranspose: {
        auto freq = v0::Constant::create(ov::element::f32, ov::Shape{1, half_dim, 1}, inv_freq());
        auto matmul = std::make_shared<v0::MatMul>(freq, convert);
        angles = std::make_shared<v1::Transpose>(matmul, i64_constant({0, 2, 1}));
        r.chain.insert(r.chain.end(), {matmul, angles});
        break;
    }
    case AngleForm::MatMulPositionsFirst: {
        auto freq = v0::Constant::create(ov::element::f32, ov::Shape{1, 1, half_dim}, inv_freq());
        angles = std::make_shared<v0::MatMul>(convert, freq);
        r.chain.push_back(angles);
        break;
    }
    case AngleForm::Multiply: {
        auto freq = v0::Constant::create(ov::element::f32, ov::Shape{1, 1, half_dim}, inv_freq());
        angles = std::make_shared<v1::Multiply>(convert, freq);
        r.chain.push_back(angles);
        break;
    }
    case AngleForm::ModernBert: {
        auto broadcast =
            std::make_shared<v3::Broadcast>(convert, i64_constant({1, 1, 1}), ov::op::BroadcastType::BIDIRECTIONAL);
        auto reshaped = std::make_shared<v1::Reshape>(broadcast, i64_constant({1, 1, -1}), false);
        auto freq = v0::Constant::create(ov::element::f32, ov::Shape{1, half_dim, 1}, inv_freq());
        auto matmul = std::make_shared<v0::MatMul>(freq, reshaped);
        auto matmul_reshaped =
            std::make_shared<v1::Reshape>(matmul, i64_constant({1, static_cast<int64_t>(half_dim), -1}), false);
        angles = std::make_shared<v1::Transpose>(matmul_reshaped, i64_constant({0, 2, 1}));
        r.chain.insert(r.chain.end(), {broadcast, reshaped, matmul, matmul_reshaped, angles});
        break;
    }
    }
    if (with_concat) {
        angles = std::make_shared<v0::Concat>(ov::OutputVector{angles, angles}, -1);
        r.chain.push_back(angles);
    }
    r.sin = std::make_shared<v0::Sin>(angles);
    r.cos = std::make_shared<v0::Cos>(angles);
    const auto dim = static_cast<int64_t>(with_concat ? 2 * half_dim : half_dim);
    r.q = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{1, positions_shape[1], dim});
    return r;
}

void mark_chain(const RopeAngles& r) {
    for (const auto& node : r.chain) {
        ov::disable_conversion(node, ov::element::f16);
    }
    ov::disable_conversion(r.sin, ov::element::f16);
    ov::disable_conversion(r.cos, ov::element::f16);
}

// q * cos + q * sin
std::shared_ptr<ov::Model> make_rotation(const RopeAngles& r,
                                         const ov::Output<ov::Node>& cos,
                                         const ov::Output<ov::Node>& sin) {
    auto rotated =
        std::make_shared<v1::Add>(std::make_shared<v1::Multiply>(r.q, cos), std::make_shared<v1::Multiply>(r.q, sin));
    return std::make_shared<ov::Model>(ov::OutputVector{rotated}, ov::ParameterVector{r.position_ids, r.q});
}

std::shared_ptr<ov::Model> make_model(AngleForm form, bool with_concat, const ov::PartialShape& positions_shape) {
    const auto r = make_angles(form, with_concat, positions_shape);
    return make_rotation(r, r.cos, r.sin);
}

std::shared_ptr<ov::Model> make_model_ref(AngleForm form, bool with_concat, const ov::PartialShape& positions_shape) {
    const auto r = make_angles(form, with_concat, positions_shape);
    mark_chain(r);
    return make_rotation(r,
                         std::make_shared<v0::Convert>(r.cos, ov::element::f32),
                         std::make_shared<v0::Convert>(r.sin, ov::element::f32));
}

// Loop that does not allow transformations in its body, as snippets::op::Subgraph
class LoopWithoutBodyTransformations : public ov::op::v5::Loop {
public:
    LoopWithoutBodyTransformations(const ov::Output<ov::Node>& trip_count, const ov::Output<ov::Node>& condition)
        : Loop(trip_count, condition) {
        m_transformations_allowed = false;
    }
};

// The model as the body of a Loop with one iteration; the model inputs are invariant inputs of the Loop
std::shared_ptr<ov::Model> wrap_in_loop(const std::shared_ptr<ov::Model>& body_model,
                                        bool transformations_allowed = true) {
    ov::ResultVector body_results{
        std::make_shared<v0::Result>(v0::Constant::create(ov::element::boolean, ov::Shape{}, {true}))};
    const auto& results = body_model->get_results();
    body_results.insert(body_results.end(), results.begin(), results.end());
    const auto body = std::make_shared<ov::Model>(body_results, body_model->get_parameters());

    const auto trip_count = v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
    const auto condition = v0::Constant::create(ov::element::boolean, ov::Shape{}, {true});
    std::shared_ptr<ov::op::v5::Loop> loop;
    if (transformations_allowed) {
        loop = std::make_shared<ov::op::v5::Loop>(trip_count, condition);
    } else {
        loop = std::make_shared<LoopWithoutBodyTransformations>(trip_count, condition);
    }
    loop->set_function(body);
    loop->set_special_body_ports({-1, 0});
    ov::ParameterVector parameters;
    for (const auto& body_parameter : body->get_parameters()) {
        parameters.push_back(
            std::make_shared<v0::Parameter>(body_parameter->get_element_type(), body_parameter->get_partial_shape()));
        loop->set_invariant_input(body_parameter, parameters.back());
    }
    ov::OutputVector outputs;
    for (size_t i = 1; i < body_results.size(); ++i) {
        outputs.push_back(loop->get_iter_value(body_results[i], -1));
    }
    loop->validate_and_infer_types();  // before the outer Results are created on the Loop outputs
    return std::make_shared<ov::Model>(outputs, parameters);
}

class DisableFP16CompForRopeAnglesTest : public TransformationTestsF {
public:
    void SetUp() override {
        TransformationTestsF::SetUp();
        manager.register_pass<ov::pass::DisableFP16CompForRopeAngles>();
    }

    void TearDown() override {
        TransformationTestsF::TearDown();
        // the fixture checks that the reference marks are in the model; check the converse too, so that a mark the
        // reference does not expect fails the test
        if (model && model_ref) {
            EXPECT_EQ(count_marks(model), count_marks(model_ref));
        }
    }

    // number of nodes kept from f16 conversion, per op type, the bodies of If/Loop/TensorIterator included
    static std::map<std::string, size_t> count_marks(const std::shared_ptr<ov::Model>& m) {
        std::map<std::string, size_t> counts;
        for (const auto& node : m->get_ordered_ops()) {
            if (ov::is_conversion_disabled(node, ov::element::f16)) {
                counts[node->get_type_name()]++;
            }
            if (const auto multi_subgraph = ov::as_type_ptr<ov::op::util::MultiSubGraphOp>(node)) {
                for (const auto& body : multi_subgraph->get_functions()) {
                    for (const auto& [type, count] : count_marks(body)) {
                        counts[type] += count;
                    }
                }
            }
        }
        return counts;
    }
};

using RopeAnglesParams = std::tuple<AngleForm, bool, ov::PartialShape>;

class DisableFP16CompForRopeAnglesParamTest : public DisableFP16CompForRopeAnglesTest,
                                              public testing::WithParamInterface<RopeAnglesParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<RopeAnglesParams>& info) {
        const auto& [form, with_concat, positions_shape] = info.param;
        const char* forms[] = {"MatMulTranspose", "MatMulPositionsFirst", "Multiply", "ModernBert"};
        std::ostringstream name;
        name << forms[static_cast<int>(form)] << "_" << (with_concat ? "Concat" : "NoConcat") << "_"
             << (positions_shape.is_static() ? "static" : "dynamic");
        return name.str();
    }
};

}  // namespace

// The angle chain is kept in f32 and bounded by a Convert after Sin/Cos, so that the rotation keeps f16.
TEST_P(DisableFP16CompForRopeAnglesParamTest, MarksAngleChainAndInsertsBoundary) {
    const auto& [form, with_concat, positions_shape] = GetParam();
    model = make_model(form, with_concat, positions_shape);
    model_ref = make_model_ref(form, with_concat, positions_shape);
}

INSTANTIATE_TEST_SUITE_P(TransformationTests,
                         DisableFP16CompForRopeAnglesParamTest,
                         testing::Combine(testing::Values(AngleForm::MatMulTranspose,
                                                          AngleForm::MatMulPositionsFirst,
                                                          AngleForm::Multiply,
                                                          AngleForm::ModernBert),
                                          testing::Values(true, false),
                                          testing::Values(ov::PartialShape{1, 16}, ov::PartialShape{1, -1})),
                         DisableFP16CompForRopeAnglesParamTest::getTestCaseName);

// Running the pass twice gives the same model as running it once: Sin/Cos already feed the boundary Convert.
TEST_F(DisableFP16CompForRopeAnglesTest, IsIdempotent) {
    model = make_model(AngleForm::MatMulTranspose, true, ov::PartialShape{1, -1});
    manager.register_pass<ov::pass::DisableFP16CompForRopeAngles>();
    model_ref = make_model_ref(AngleForm::MatMulTranspose, true, ov::PartialShape{1, -1});
}

// One position Convert shared by two tables (e.g. local and global RoPE): both chains are kept in f32.
TEST_F(DisableFP16CompForRopeAnglesTest, SharedPositionConvert) {
    const auto freq = [](float base) {
        return v0::Constant::create(ov::element::f32, ov::Shape{1, 1, half_dim}, std::vector<float>(half_dim, base));
    };
    {
        auto position_ids = std::make_shared<v0::Parameter>(ov::element::i64, ov::PartialShape{1, -1});
        auto unsqueeze = std::make_shared<v0::Unsqueeze>(position_ids, i64_constant({-1}));
        auto convert = std::make_shared<v0::Convert>(unsqueeze, ov::element::f32);
        ov::OutputVector outputs;
        for (float base : {1.0f, 0.5f}) {
            auto multiply = std::make_shared<v1::Multiply>(convert, freq(base));
            outputs.push_back(
                std::make_shared<v1::Add>(std::make_shared<v0::Sin>(multiply), std::make_shared<v0::Cos>(multiply)));
        }
        model = std::make_shared<ov::Model>(outputs, ov::ParameterVector{position_ids});
    }
    {
        auto position_ids = std::make_shared<v0::Parameter>(ov::element::i64, ov::PartialShape{1, -1});
        auto unsqueeze = std::make_shared<v0::Unsqueeze>(position_ids, i64_constant({-1}));
        auto convert = std::make_shared<v0::Convert>(unsqueeze, ov::element::f32);
        ov::disable_conversion(convert, ov::element::f16);
        ov::OutputVector outputs;
        for (float base : {1.0f, 0.5f}) {
            auto multiply = std::make_shared<v1::Multiply>(convert, freq(base));
            auto sin = std::make_shared<v0::Sin>(multiply);
            auto cos = std::make_shared<v0::Cos>(multiply);
            for (const auto& node : ov::NodeVector{multiply, sin, cos}) {
                ov::disable_conversion(node, ov::element::f16);
            }
            outputs.push_back(std::make_shared<v1::Add>(std::make_shared<v0::Convert>(sin, ov::element::f32),
                                                        std::make_shared<v0::Convert>(cos, ov::element::f32)));
        }
        model_ref = std::make_shared<ov::Model>(outputs, ov::ParameterVector{position_ids});
    }
}

// Two tables share the position Convert and one angle Multiply has another consumer: that chain is dropped, and then
// the other one, whose shared Convert would drag the dropped chain into f32. The model is left untouched.
TEST_F(DisableFP16CompForRopeAnglesTest, SkipsSharedPositionsOfRejectedChain) {
    auto position_ids = std::make_shared<v0::Parameter>(ov::element::i64, ov::PartialShape{1, -1});
    auto convert = std::make_shared<v0::Convert>(std::make_shared<v0::Unsqueeze>(position_ids, i64_constant({-1})),
                                                 ov::element::f32);
    ov::OutputVector outputs;
    for (float base : {1.0f, 0.5f}) {
        auto multiply = std::make_shared<v1::Multiply>(
            convert,
            v0::Constant::create(ov::element::f32, ov::Shape{1, 1, half_dim}, std::vector<float>(half_dim, base)));
        outputs.push_back(
            std::make_shared<v1::Add>(std::make_shared<v0::Sin>(multiply), std::make_shared<v0::Cos>(multiply)));
        if (base == 0.5f) {
            outputs.push_back(
                std::make_shared<v1::Multiply>(multiply, v0::Constant::create(ov::element::f32, {}, {2.0f})));
        }
    }
    model = std::make_shared<ov::Model>(outputs, ov::ParameterVector{position_ids});
}

// Sin/Cos feeding Results directly: the chain is marked, no Convert is inserted in front of a Result.
TEST_F(DisableFP16CompForRopeAnglesTest, NoBoundaryBeforeResult) {
    {
        const auto r = make_angles(AngleForm::Multiply, true, ov::PartialShape{1, -1});
        model = std::make_shared<ov::Model>(ov::OutputVector{r.sin, r.cos}, ov::ParameterVector{r.position_ids});
    }
    {
        const auto r = make_angles(AngleForm::Multiply, true, ov::PartialShape{1, -1});
        mark_chain(r);
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{r.sin, r.cos}, ov::ParameterVector{r.position_ids});
    }
}

// A Sin/Cos consumer already kept in f32 reads the table directly.
TEST_F(DisableFP16CompForRopeAnglesTest, NoBoundaryBeforeKeptConsumer) {
    {
        const auto r = make_angles(AngleForm::Multiply, true, ov::PartialShape{1, -1});
        auto kept = std::make_shared<v1::Multiply>(r.q, r.cos);
        ov::disable_conversion(kept, ov::element::f16);
        auto rotated = std::make_shared<v1::Add>(kept, std::make_shared<v1::Multiply>(r.q, r.sin));
        model = std::make_shared<ov::Model>(ov::OutputVector{rotated}, ov::ParameterVector{r.position_ids, r.q});
    }
    {
        const auto r = make_angles(AngleForm::Multiply, true, ov::PartialShape{1, -1});
        mark_chain(r);
        auto kept = std::make_shared<v1::Multiply>(r.q, r.cos);
        ov::disable_conversion(kept, ov::element::f16);
        auto sin = std::make_shared<v0::Convert>(r.sin, ov::element::f32);
        auto rotated = std::make_shared<v1::Add>(kept, std::make_shared<v1::Multiply>(r.q, sin));
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{rotated}, ov::ParameterVector{r.position_ids, r.q});
    }
}

// Float positions converted to f32: not integer positions, left untouched.
TEST_F(DisableFP16CompForRopeAnglesTest, SkipsFloatPositions) {
    auto positions = std::make_shared<v0::Parameter>(ov::element::f16, ov::PartialShape{1, -1, 1});
    auto q = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1, half_dim});
    auto convert = std::make_shared<v0::Convert>(positions, ov::element::f32);
    auto angles = std::make_shared<v1::Multiply>(
        convert,
        v0::Constant::create(ov::element::f32, ov::Shape{1, 1, half_dim}, std::vector<float>(half_dim, 0.5f)));
    auto rotated = std::make_shared<v1::Add>(std::make_shared<v1::Multiply>(q, std::make_shared<v0::Cos>(angles)),
                                             std::make_shared<v1::Multiply>(q, std::make_shared<v0::Sin>(angles)));
    model = std::make_shared<ov::Model>(ov::OutputVector{rotated}, ov::ParameterVector{positions, q});
}

// Only a Sin (no Cos sibling): not a RoPE table, left untouched.
TEST_F(DisableFP16CompForRopeAnglesTest, SkipsSinOnly) {
    auto position_ids = std::make_shared<v0::Parameter>(ov::element::i64, ov::PartialShape{1, -1});
    auto q = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1, half_dim});
    auto convert = std::make_shared<v0::Convert>(std::make_shared<v0::Unsqueeze>(position_ids, i64_constant({-1})),
                                                 ov::element::f32);
    auto angles = std::make_shared<v1::Multiply>(
        convert,
        v0::Constant::create(ov::element::f32, ov::Shape{1, 1, half_dim}, std::vector<float>(half_dim, 0.5f)));
    auto rotated = std::make_shared<v1::Multiply>(q, std::make_shared<v0::Sin>(angles));
    model = std::make_shared<ov::Model>(ov::OutputVector{rotated}, ov::ParameterVector{position_ids, q});
}

// The positions Convert also feeds an unrelated op: marking it would drag that op into f32, left untouched.
TEST_F(DisableFP16CompForRopeAnglesTest, SkipsPositionsWithOtherUsers) {
    const auto r = make_angles(AngleForm::Multiply, true, ov::PartialShape{1, -1});
    auto other = std::make_shared<v1::Multiply>(r.chain.front(), v0::Constant::create(ov::element::f32, {}, {2.0f}));
    auto rotated = make_rotation(r, r.cos, r.sin)->get_results()[0]->input_value(0);
    model = std::make_shared<ov::Model>(ov::OutputVector{rotated, other}, ov::ParameterVector{r.position_ids, r.q});
}

// The angle MatMul also feeds a non-trigonometric op: left untouched.
TEST_F(DisableFP16CompForRopeAnglesTest, SkipsAngleWithOtherUsers) {
    const auto r = make_angles(AngleForm::MatMulTranspose, true, ov::PartialShape{1, -1});
    auto other = std::make_shared<v1::Multiply>(r.chain[1], v0::Constant::create(ov::element::f32, {}, {2.0f}));
    auto rotated = make_rotation(r, r.cos, r.sin)->get_results()[0]->input_value(0);
    model = std::make_shared<ov::Model>(ov::OutputVector{rotated, other}, ov::ParameterVector{r.position_ids, r.q});
}

// Concat of two different tensors is not the duplicated RoPE table: left untouched.
TEST_F(DisableFP16CompForRopeAnglesTest, SkipsConcatOfDifferentInputs) {
    auto position_ids = std::make_shared<v0::Parameter>(ov::element::i64, ov::PartialShape{1, -1});
    auto other = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1, half_dim});
    auto convert = std::make_shared<v0::Convert>(std::make_shared<v0::Unsqueeze>(position_ids, i64_constant({-1})),
                                                 ov::element::f32);
    auto angles = std::make_shared<v1::Multiply>(
        convert,
        v0::Constant::create(ov::element::f32, ov::Shape{1, 1, half_dim}, std::vector<float>(half_dim, 0.5f)));
    auto concat = std::make_shared<v0::Concat>(ov::OutputVector{angles, other}, -1);
    auto add = std::make_shared<v1::Add>(std::make_shared<v0::Sin>(concat), std::make_shared<v0::Cos>(concat));
    model = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{position_ids, other});
}

// The table concatenates two angle tensors built from distinct position Converts: not the duplicated RoPE table,
// left untouched (marking only the matched half would keep the other half in f16).
TEST_F(DisableFP16CompForRopeAnglesTest, SkipsConcatOfDistinctAngles) {
    auto position_ids = std::make_shared<v0::Parameter>(ov::element::i64, ov::PartialShape{1, -1});
    auto q = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1, 2 * half_dim});
    auto positions = std::make_shared<v0::Unsqueeze>(position_ids, i64_constant({-1}));
    auto freq = v0::Constant::create(ov::element::f32, ov::Shape{1, 1, half_dim}, std::vector<float>(half_dim, 0.5f));
    auto first = std::make_shared<v1::Multiply>(std::make_shared<v0::Convert>(positions, ov::element::f32), freq);
    auto second = std::make_shared<v1::Multiply>(std::make_shared<v0::Convert>(positions, ov::element::f32), freq);
    auto concat = std::make_shared<v0::Concat>(ov::OutputVector{first, second}, -1);
    auto rotated = std::make_shared<v1::Add>(std::make_shared<v1::Multiply>(q, std::make_shared<v0::Cos>(concat)),
                                             std::make_shared<v1::Multiply>(q, std::make_shared<v0::Sin>(concat)));
    model = std::make_shared<ov::Model>(ov::OutputVector{rotated}, ov::ParameterVector{position_ids, q});
}

// Integer constant positions are folded at compile time, not a runtime table: left untouched.
TEST_F(DisableFP16CompForRopeAnglesTest, SkipsConstantPositions) {
    auto q = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{1, 16, half_dim});
    auto positions = v0::Constant::create(ov::element::i64, ov::Shape{1, 16, 1}, std::vector<int64_t>(16, 3));
    auto convert = std::make_shared<v0::Convert>(positions, ov::element::f32);
    auto angles = std::make_shared<v1::Multiply>(
        convert,
        v0::Constant::create(ov::element::f32, ov::Shape{1, 1, half_dim}, std::vector<float>(half_dim, 0.5f)));
    auto rotated = std::make_shared<v1::Add>(std::make_shared<v1::Multiply>(q, std::make_shared<v0::Cos>(angles)),
                                             std::make_shared<v1::Multiply>(q, std::make_shared<v0::Sin>(angles)));
    model = std::make_shared<ov::Model>(ov::OutputVector{rotated}, ov::ParameterVector{q});
}

// The angle chains in the body of a Loop, nested or not, are kept in f32 there; the Loop itself is not marked.
TEST_F(DisableFP16CompForRopeAnglesTest, MarksAngleChainInLoopBody) {
    model = wrap_in_loop(make_model(AngleForm::MatMulTranspose, true, ov::PartialShape{1, -1}));
    model_ref = wrap_in_loop(make_model_ref(AngleForm::MatMulTranspose, true, ov::PartialShape{1, -1}));
}

TEST_F(DisableFP16CompForRopeAnglesTest, MarksAngleChainInNestedLoopBody) {
    model = wrap_in_loop(wrap_in_loop(make_model(AngleForm::Multiply, true, ov::PartialShape{1, -1})));
    model_ref = wrap_in_loop(wrap_in_loop(make_model_ref(AngleForm::Multiply, true, ov::PartialShape{1, -1})));
}

// A Loop that does not allow transformations in its body (as snippets::op::Subgraph): left untouched.
TEST_F(DisableFP16CompForRopeAnglesTest, SkipsBodyWithoutTransformations) {
    model = wrap_in_loop(make_model(AngleForm::Multiply, true, ov::PartialShape{1, -1}), false);
}

// Positions converted to f32 outside of a Loop reach its body as an f32 Parameter: the body chain is not matched.
TEST_F(DisableFP16CompForRopeAnglesTest, SkipsBodyWithPositionsConvertedOutside) {
    auto positions = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1, 1});
    auto q = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1, half_dim});
    auto angles = std::make_shared<v1::Multiply>(
        positions,
        v0::Constant::create(ov::element::f32, ov::Shape{1, 1, half_dim}, std::vector<float>(half_dim, 0.5f)));
    auto rotated = std::make_shared<v1::Add>(std::make_shared<v1::Multiply>(q, std::make_shared<v0::Cos>(angles)),
                                             std::make_shared<v1::Multiply>(q, std::make_shared<v0::Sin>(angles)));
    model = wrap_in_loop(std::make_shared<ov::Model>(ov::OutputVector{rotated}, ov::ParameterVector{positions, q}));
}

// The angle chains in both branches of an If are kept in f32. The fixture does not compare If bodies, so each branch is
// compared with its reference here.
TEST(TransformationTests, DisableFP16CompForRopeAnglesMarksAngleChainsInIfBranches) {
    const auto make_if = [](const std::shared_ptr<ov::Model>& then_body, const std::shared_ptr<ov::Model>& else_body) {
        auto if_op = std::make_shared<ov::op::v8::If>(v0::Constant::create(ov::element::boolean, ov::Shape{1}, {true}));
        if_op->set_then_body(then_body);
        if_op->set_else_body(else_body);
        ov::ParameterVector parameters;
        for (size_t i = 0; i < then_body->get_parameters().size(); ++i) {
            const auto& then_parameter = then_body->get_parameters()[i];
            parameters.push_back(std::make_shared<v0::Parameter>(then_parameter->get_element_type(),
                                                                 then_parameter->get_partial_shape()));
            if_op->set_input(parameters.back(), then_parameter, else_body->get_parameters()[i]);
        }
        const auto output = if_op->set_output(then_body->get_results()[0], else_body->get_results()[0]);
        return std::make_shared<ov::Model>(ov::OutputVector{output}, parameters);
    };
    const auto then_body = make_model(AngleForm::MatMulTranspose, true, ov::PartialShape{1, -1});
    const auto else_body = make_model(AngleForm::Multiply, true, ov::PartialShape{1, -1});
    const auto model = make_if(then_body, else_body);
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::DisableFP16CompForRopeAngles>();
    manager.run_passes(model);

    const auto comparator = FunctionsComparator::with_default().enable(FunctionsComparator::RUNTIME_KEYS);
    for (const auto& [body, body_ref] :
         {std::make_pair(then_body, make_model_ref(AngleForm::MatMulTranspose, true, ov::PartialShape{1, -1})),
          std::make_pair(else_body, make_model_ref(AngleForm::Multiply, true, ov::PartialShape{1, -1}))}) {
        const auto result = comparator(body, body_ref);
        EXPECT_TRUE(result.valid) << result.message;
        EXPECT_EQ(DisableFP16CompForRopeAnglesTest::count_marks(body),
                  DisableFP16CompForRopeAnglesTest::count_marks(body_ref));
    }
}

// A change in a body is reported, and running again changes nothing.
TEST(TransformationTests, DisableFP16CompForRopeAnglesReportsBodyChanges) {
    const auto model = wrap_in_loop(make_model(AngleForm::Multiply, true, ov::PartialShape{1, -1}));
    ov::pass::DisableFP16CompForRopeAngles pass;
    EXPECT_TRUE(pass.run_on_model(model));
    EXPECT_FALSE(pass.run_on_model(model));
}

// Followed by ConvertPrecision(f32 -> f16) as in the CPU plugin, at top level and in a Loop body: the angles stay in
// f32 and the rotation in f16. ConvertPrecision replaces the boundary Convert (in a body, it is left as an f16 -> f16
// Convert that EliminateConvert removes later), so the precisions are checked rather than a reference model.
TEST(TransformationTests, DisableFP16CompForRopeAnglesWithConvertPrecision) {
    for (const auto in_loop : {false, true}) {
        for (const auto form : {AngleForm::MatMulTranspose,
                                AngleForm::MatMulPositionsFirst,
                                AngleForm::Multiply,
                                AngleForm::ModernBert}) {
            SCOPED_TRACE(std::to_string(static_cast<int>(form)) + (in_loop ? " in Loop" : ""));
            const auto r = make_angles(form, true, ov::PartialShape{1, -1});
            auto model = make_rotation(r, r.cos, r.sin);
            // the rotation, taken before the passes: ConvertPrecision puts a Convert between it and the f32 Result
            const auto rotated = model->get_results()[0]->get_input_node_shared_ptr(0);
            if (in_loop) {
                model = wrap_in_loop(model);
            }
            ov::pass::Manager manager;
            manager.register_pass<ov::pass::DisableFP16CompForRopeAngles>();
            manager.register_pass<ov::pass::ConvertPrecision>(precisions_map{{ov::element::f32, ov::element::f16}},
                                                              type_to_fuse_map{},
                                                              true,
                                                              false);
            manager.run_passes(model);

            for (const auto& node : r.chain) {
                EXPECT_EQ(node->get_output_element_type(0), ov::element::f32) << node;
            }
            for (const auto& trig : ov::NodeVector{r.sin, r.cos}) {
                EXPECT_EQ(trig->get_output_element_type(0), ov::element::f32) << trig;
                for (const auto& input : trig->output(0).get_target_inputs()) {
                    const auto consumer = input.get_node();
                    ASSERT_TRUE(ov::is_type<v0::Convert>(consumer)) << consumer;
                    EXPECT_EQ(consumer->get_output_element_type(0), ov::element::f16) << consumer;
                }
            }
            EXPECT_TRUE(ov::is_type<v1::Add>(rotated)) << rotated;
            EXPECT_EQ(rotated->get_output_element_type(0), ov::element::f16) << rotated;
        }
    }
}

// Cos also feeds 1 / Maximum(1 - cos, eps), which ConvertPrecision keeps in f32: that consumer reads the f32 cos
// directly (in f16, 1 - cos is 0 for the small angles), while the rotation still reads the cos/sin in f16.
TEST(TransformationTests, DisableFP16CompForRopeAnglesWithConvertPrecision_PrecisionSensitiveConsumer) {
    const auto r = make_angles(AngleForm::Multiply, true, ov::PartialShape{1, -1});
    // the rotation, taken before the passes: ConvertPrecision puts a Convert between it and the f32 Result
    const auto rotated = make_rotation(r, r.cos, r.sin)->get_results()[0]->get_input_node_shared_ptr(0);
    auto one_minus_cos = std::make_shared<v1::Subtract>(v0::Constant::create(ov::element::f32, {}, {1.0f}), r.cos);
    auto clamped = std::make_shared<v1::Maximum>(one_minus_cos, v0::Constant::create(ov::element::f32, {}, {1.0e-12f}));
    auto inverse = std::make_shared<v1::Divide>(v0::Constant::create(ov::element::f32, {}, {1.0f}), clamped);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{rotated->output(0), inverse},
                                             ov::ParameterVector{r.position_ids, r.q});
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::DisableFP16CompForRopeAngles>();
    manager.register_pass<ov::pass::ConvertPrecision>(precisions_map{{ov::element::f32, ov::element::f16}},
                                                      type_to_fuse_map{},
                                                      true,
                                                      false);
    manager.run_passes(model);

    // no f16 round trip between Cos and the eps-guarded division
    EXPECT_EQ(one_minus_cos->input_value(1), r.cos->output(0));
    for (const auto& node : ov::NodeVector{r.cos, one_minus_cos, clamped, inverse}) {
        EXPECT_EQ(node->get_output_element_type(0), ov::element::f32) << node;
    }
    // the other consumers of Sin/Cos are the Converts to f16 in front of the rotation, which keeps f16
    for (const auto& trig : ov::NodeVector{r.sin, r.cos}) {
        for (const auto& input : trig->output(0).get_target_inputs()) {
            const auto consumer = input.get_node();
            if (consumer == one_minus_cos.get()) {
                continue;
            }
            ASSERT_TRUE(ov::is_type<v0::Convert>(consumer)) << consumer;
            EXPECT_EQ(consumer->get_output_element_type(0), ov::element::f16) << consumer;
        }
    }
    EXPECT_TRUE(ov::is_type<v1::Add>(rotated)) << rotated;
    EXPECT_EQ(rotated->get_output_element_type(0), ov::element::f16) << rotated;
}
