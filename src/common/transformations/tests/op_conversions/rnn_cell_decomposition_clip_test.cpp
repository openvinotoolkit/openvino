// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gru_cell.hpp"
#include "openvino/op/lstm_cell.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/rnn_cell.hpp"
#include "openvino/op/util/rnn_cell_base.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/op_conversions/gru_cell_decomposition.hpp"
#include "transformations/op_conversions/lstm_cell_decomposition.hpp"
#include "transformations/op_conversions/rnn_cell_decomposition.hpp"

using namespace ov;
using namespace testing;

namespace v0 = ov::op::v0;
namespace v3 = ov::op::v3;
namespace v4 = ov::op::v4;

namespace {
constexpr size_t batch = 2;
constexpr size_t input_size = 3;
constexpr size_t hidden_size = 4;

TEST(RNNClipUtilsTest, ClassifiesClipValues) {
    using ov::op::util::classify_rnn_clip;
    using ov::op::util::RNNClipMode;

    const auto infinity = std::numeric_limits<float>::infinity();

    EXPECT_EQ(RNNClipMode::NONE, classify_rnn_clip(0.f));
    EXPECT_EQ(RNNClipMode::NONE, classify_rnn_clip(infinity));
    EXPECT_EQ(RNNClipMode::CLAMP, classify_rnn_clip(1.f));
    EXPECT_EQ(RNNClipMode::INVALID, classify_rnn_clip(-1.f));
    EXPECT_EQ(RNNClipMode::INVALID, classify_rnn_clip(-infinity));
    EXPECT_EQ(RNNClipMode::INVALID, classify_rnn_clip(std::numeric_limits<float>::quiet_NaN()));
}

TEST(RNNClipUtilsTest, ComparesClipValuesByMeaning) {
    using ov::op::util::are_clips_equal;

    const auto infinity = std::numeric_limits<float>::infinity();
    const auto nan = std::numeric_limits<float>::quiet_NaN();

    EXPECT_TRUE(are_clips_equal(0.f, infinity));
    EXPECT_TRUE(are_clips_equal(infinity, 0.f));
    EXPECT_TRUE(are_clips_equal(1.f, 1.f));
    EXPECT_TRUE(are_clips_equal(-1.f, -1.f));
    EXPECT_FALSE(are_clips_equal(0.f, 1.f));
    EXPECT_FALSE(are_clips_equal(0.f, -1.f));
    EXPECT_FALSE(are_clips_equal(nan, nan));
}

// Each *CellDecomposition pass inserts a Clamp only when `clip` actually requests clipping. `clip == 0` and
// `clip == inf` both mean "no clipping" per the RNN/GRU/LSTM specs, so neither value may produce a Clamp.
struct CellDecompositionClipParams {
    std::string name;
    std::function<std::shared_ptr<ov::Model>(float clip)> makeModel;
    std::function<void(ov::pass::Manager&)> registerPass;
};

std::shared_ptr<ov::Model> makeRNNCellModel(float clip) {
    const auto X = std::make_shared<v0::Parameter>(element::f32, Shape{batch, input_size});
    const auto H = std::make_shared<v0::Parameter>(element::f32, Shape{batch, hidden_size});
    const auto W = v0::Constant::create(element::f32, Shape{hidden_size, input_size}, {1.f});
    const auto R = v0::Constant::create(element::f32, Shape{hidden_size, hidden_size}, {1.f});
    const auto B = v0::Constant::create(element::f32, Shape{hidden_size}, {0.f});

    const auto cell = std::make_shared<v0::RNNCell>(X,
                                                    H,
                                                    W,
                                                    R,
                                                    B,
                                                    hidden_size,
                                                    std::vector<std::string>{"tanh"},
                                                    std::vector<float>{},
                                                    std::vector<float>{},
                                                    clip);
    return std::make_shared<ov::Model>(cell->outputs(), ParameterVector{X, H});
}

std::shared_ptr<ov::Model> makeGRUCellModel(float clip) {
    const auto X = std::make_shared<v0::Parameter>(element::f32, Shape{batch, input_size});
    const auto H = std::make_shared<v0::Parameter>(element::f32, Shape{batch, hidden_size});
    const auto W = v0::Constant::create(element::f32, Shape{3 * hidden_size, input_size}, {1.f});
    const auto R = v0::Constant::create(element::f32, Shape{3 * hidden_size, hidden_size}, {1.f});
    const auto B = v0::Constant::create(element::f32, Shape{3 * hidden_size}, {0.f});

    const auto cell = std::make_shared<v3::GRUCell>(X,
                                                    H,
                                                    W,
                                                    R,
                                                    B,
                                                    hidden_size,
                                                    std::vector<std::string>{"sigmoid", "tanh"},
                                                    std::vector<float>{},
                                                    std::vector<float>{},
                                                    clip,
                                                    /*linear_before_reset=*/false);
    return std::make_shared<ov::Model>(cell->outputs(), ParameterVector{X, H});
}

std::shared_ptr<ov::Model> makeLSTMCellModel(float clip) {
    const auto X = std::make_shared<v0::Parameter>(element::f32, Shape{batch, input_size});
    const auto H = std::make_shared<v0::Parameter>(element::f32, Shape{batch, hidden_size});
    const auto C = std::make_shared<v0::Parameter>(element::f32, Shape{batch, hidden_size});
    const auto W = v0::Constant::create(element::f32, Shape{4 * hidden_size, input_size}, {1.f});
    const auto R = v0::Constant::create(element::f32, Shape{4 * hidden_size, hidden_size}, {1.f});
    const auto B = v0::Constant::create(element::f32, Shape{4 * hidden_size}, {0.f});

    const auto cell = std::make_shared<v4::LSTMCell>(X,
                                                     H,
                                                     C,
                                                     W,
                                                     R,
                                                     B,
                                                     hidden_size,
                                                     std::vector<std::string>{"sigmoid", "tanh", "tanh"},
                                                     std::vector<float>{},
                                                     std::vector<float>{},
                                                     clip);
    return std::make_shared<ov::Model>(cell->outputs(), ParameterVector{X, H, C});
}

const std::vector<CellDecompositionClipParams> cell_params{
    {"RNNCell",
     makeRNNCellModel,
     [](ov::pass::Manager& m) {
         m.register_pass<ov::pass::RNNCellDecomposition>();
     }},
    {"GRUCell",
     makeGRUCellModel,
     [](ov::pass::Manager& m) {
         m.register_pass<ov::pass::GRUCellDecomposition>();
     }},
    {"LSTMCell",
     makeLSTMCellModel,
     [](ov::pass::Manager& m) {
         m.register_pass<ov::pass::LSTMCellDecomposition>();
     }},
};
}  // namespace

class CellDecompositionClipTests : public TransformationTestsF, public WithParamInterface<CellDecompositionClipParams> {
public:
    static std::string getTestCaseName(const TestParamInfo<CellDecompositionClipParams>& info) {
        return info.param.name;
    }
};

// clip == inf must decompose to exactly the same graph as clip == 0, i.e. without any Clamp.
TEST_P(CellDecompositionClipTests, ClipInfDecomposesLikeClipZero) {
    const auto& p = GetParam();
    model = p.makeModel(std::numeric_limits<float>::infinity());
    p.registerPass(manager);

    // model_ref is the already-decomposed clip == 0 graph, so a Clamp appearing for inf fails the comparison.
    ov::pass::Manager ref_manager;
    p.registerPass(ref_manager);
    model_ref = p.makeModel(0.f);
    ref_manager.run_passes(model_ref);
    ASSERT_EQ(count_ops_of_type<v0::Clamp>(model_ref), 0);
}

// A finite clip must still insert Clamp, so the check above is not vacuously true.
TEST_P(CellDecompositionClipTests, FiniteClipInsertsClamp) {
    const auto& p = GetParam();
    model = p.makeModel(3.5f);

    ov::pass::Manager clip_manager;
    p.registerPass(clip_manager);
    clip_manager.run_passes(model);
    EXPECT_GT(count_ops_of_type<v0::Clamp>(model), 0);

    // The fixture would otherwise compare `model` against a clone taken after the pass already ran.
    test_skipped = true;
}

// Neither a negative nor a NaN clip describes usable bounds `[-clip, clip]`: Clamp rejects `min > max`, so a pass
// that created one would throw during decomposition instead of leaving the graph unclipped. Both must be skipped.
TEST_P(CellDecompositionClipTests, InvalidClipInsertsNoClamp) {
    const auto& p = GetParam();
    // This test drives its own models, so the fixture has nothing to compare. Set before the assertions below so a
    // failure surfaces on its own instead of being masked by an uninitialized-model error from TearDown().
    test_skipped = true;

    for (const float clip : {-1.f, -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
        auto invalid_clip_model = p.makeModel(clip);

        ov::pass::Manager invalid_clip_manager;
        p.registerPass(invalid_clip_manager);
        ASSERT_NO_THROW(invalid_clip_manager.run_passes(invalid_clip_model)) << "clip = " << clip;
        EXPECT_EQ(count_ops_of_type<v0::Clamp>(invalid_clip_model), 0) << "clip = " << clip;
    }
}

INSTANTIATE_TEST_SUITE_P(TransformationTests,
                         CellDecompositionClipTests,
                         ValuesIn(cell_params),
                         CellDecompositionClipTests::getTestCaseName);
