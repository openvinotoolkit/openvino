// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/topk_renormalize_to_softmax_after_topk_fusion.hpp"

#include <gtest/gtest.h>

#include <memory>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/topk.hpp"

using namespace testing;
using namespace ov;

namespace {

std::shared_ptr<Model> build_pattern_v8(const PartialShape& shape,
                                        int64_t softmax_axis,
                                        int64_t topk_axis,
                                        int64_t reduce_axis,
                                        size_t softmax_consumers = 1,
                                        size_t topk_values_consumers = 2,
                                        bool reduce_keep_dims = true) {
    auto data = std::make_shared<op::v0::Parameter>(element::f32, shape);
    auto softmax = std::make_shared<op::v8::Softmax>(data, softmax_axis);
    auto k = op::v0::Constant::create(element::i64, Shape{}, {2});
    auto topk = std::make_shared<op::v11::TopK>(softmax,
                                                k,
                                                topk_axis,
                                                op::v11::TopK::Mode::MAX,
                                                op::v11::TopK::SortType::SORT_VALUES);
    auto red_axes = op::v0::Constant::create(element::i64, Shape{1}, {reduce_axis});
    auto reduce = std::make_shared<op::v1::ReduceSum>(topk->output(0), red_axes, reduce_keep_dims);
    auto divide = std::make_shared<op::v1::Divide>(topk->output(0), reduce);

    ResultVector results{std::make_shared<op::v0::Result>(divide), std::make_shared<op::v0::Result>(topk->output(1))};

    if (softmax_consumers > 1) {
        results.push_back(std::make_shared<op::v0::Result>(softmax));
    }
    if (topk_values_consumers > 2) {
        results.push_back(std::make_shared<op::v0::Result>(topk->output(0)));
    }
    return std::make_shared<Model>(results, ParameterVector{data});
}

std::shared_ptr<Model> build_ref_v8(const PartialShape& shape, int64_t topk_axis, int64_t softmax_axis) {
    auto data = std::make_shared<op::v0::Parameter>(element::f32, shape);
    auto k = op::v0::Constant::create(element::i64, Shape{}, {2});
    auto topk = std::make_shared<op::v11::TopK>(data,
                                                k,
                                                topk_axis,
                                                op::v11::TopK::Mode::MAX,
                                                op::v11::TopK::SortType::SORT_VALUES);
    auto softmax = std::make_shared<op::v8::Softmax>(topk->output(0), softmax_axis);
    return std::make_shared<Model>(
        ResultVector{std::make_shared<op::v0::Result>(softmax), std::make_shared<op::v0::Result>(topk->output(1))},
        ParameterVector{data});
}

}  // namespace

TEST_F(TransformationTestsF, TopkRenormalizeToSoftmaxAfterTopkFusion_BasicV8) {
    PartialShape shape{2, 8};
    model = build_pattern_v8(shape, 1, 1, 1);
    manager.register_pass<pass::TopkRenormalizeToSoftmaxAfterTopkFusion>();
    model_ref = build_ref_v8(shape, 1, 1);
}

TEST_F(TransformationTestsF, TopkRenormalizeToSoftmaxAfterTopkFusion_NegativeSoftmaxAxisV8) {
    PartialShape shape{2, 8};
    model = build_pattern_v8(shape, -1, 1, -1);
    manager.register_pass<pass::TopkRenormalizeToSoftmaxAfterTopkFusion>();
    model_ref = build_ref_v8(shape, 1, -1);
}

TEST_F(TransformationTestsF, TopkRenormalizeToSoftmaxAfterTopkFusion_V1Softmax) {
    PartialShape shape{4, 6};
    {
        auto data = std::make_shared<op::v0::Parameter>(element::f32, shape);
        auto softmax = std::make_shared<op::v1::Softmax>(data, 1);
        auto k = op::v0::Constant::create(element::i64, Shape{}, {2});
        auto topk =
            std::make_shared<op::v1::TopK>(softmax, k, 1, op::v1::TopK::Mode::MAX, op::v1::TopK::SortType::SORT_VALUES);
        auto red_axes = op::v0::Constant::create(element::i64, Shape{1}, {1});
        auto reduce = std::make_shared<op::v1::ReduceSum>(topk->output(0), red_axes, true);
        auto divide = std::make_shared<op::v1::Divide>(topk->output(0), reduce);
        model = std::make_shared<Model>(
            ResultVector{std::make_shared<op::v0::Result>(divide), std::make_shared<op::v0::Result>(topk->output(1))},
            ParameterVector{data});
        manager.register_pass<pass::TopkRenormalizeToSoftmaxAfterTopkFusion>();
    }
    {
        auto data = std::make_shared<op::v0::Parameter>(element::f32, shape);
        auto k = op::v0::Constant::create(element::i64, Shape{}, {2});
        auto topk =
            std::make_shared<op::v1::TopK>(data, k, 1, op::v1::TopK::Mode::MAX, op::v1::TopK::SortType::SORT_VALUES);
        auto softmax = std::make_shared<op::v1::Softmax>(topk->output(0), 1);
        model_ref = std::make_shared<Model>(
            ResultVector{std::make_shared<op::v0::Result>(softmax), std::make_shared<op::v0::Result>(topk->output(1))},
            ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, TopkRenormalizeToSoftmaxAfterTopkFusion_NegativeKeepDimsFalse) {
    PartialShape shape{2, 8};
    model = build_pattern_v8(shape, 1, 1, 1, 1, 2, /*reduce_keep_dims=*/false);
    manager.register_pass<pass::TopkRenormalizeToSoftmaxAfterTopkFusion>();
}

TEST_F(TransformationTestsF, TopkRenormalizeToSoftmaxAfterTopkFusion_NegativeAxesMismatch) {
    PartialShape shape{2, 8};
    model = build_pattern_v8(shape, /*softmax_axis=*/1, /*topk_axis=*/1, /*reduce_axis=*/0);
    manager.register_pass<pass::TopkRenormalizeToSoftmaxAfterTopkFusion>();
}

TEST_F(TransformationTestsF, TopkRenormalizeToSoftmaxAfterTopkFusion_NegativeSoftmaxMultiConsumer) {
    PartialShape shape{2, 8};
    model = build_pattern_v8(shape, 1, 1, 1, /*softmax_consumers=*/2);
    manager.register_pass<pass::TopkRenormalizeToSoftmaxAfterTopkFusion>();
}

TEST_F(TransformationTestsF, TopkRenormalizeToSoftmaxAfterTopkFusion_NegativeTopkValuesExtraConsumer) {
    PartialShape shape{2, 8};
    model = build_pattern_v8(shape, 1, 1, 1, 1, /*topk_values_consumers=*/3);
    manager.register_pass<pass::TopkRenormalizeToSoftmaxAfterTopkFusion>();
}

TEST_F(TransformationTestsF, TopkRenormalizeToSoftmaxAfterTopkFusion_ExtraIndicesConsumer) {
    // Extra consumers on TopK.indices (e.g. MoE Gather) must not block the pass.
    PartialShape shape{2, 8};
    {
        auto data = std::make_shared<op::v0::Parameter>(element::f32, shape);
        auto softmax = std::make_shared<op::v8::Softmax>(data, 1);
        auto k = op::v0::Constant::create(element::i64, Shape{}, {2});
        auto topk = std::make_shared<op::v11::TopK>(softmax,
                                                    k,
                                                    1,
                                                    op::v11::TopK::Mode::MAX,
                                                    op::v11::TopK::SortType::SORT_VALUES);
        auto red_axes = op::v0::Constant::create(element::i64, Shape{1}, {1});
        auto reduce = std::make_shared<op::v1::ReduceSum>(topk->output(0), red_axes, true);
        auto divide = std::make_shared<op::v1::Divide>(topk->output(0), reduce);
        model = std::make_shared<Model>(ResultVector{std::make_shared<op::v0::Result>(divide),
                                                     std::make_shared<op::v0::Result>(topk->output(1)),
                                                     std::make_shared<op::v0::Result>(topk->output(1))},
                                        ParameterVector{data});
        manager.register_pass<pass::TopkRenormalizeToSoftmaxAfterTopkFusion>();
    }
    {
        auto data = std::make_shared<op::v0::Parameter>(element::f32, shape);
        auto k = op::v0::Constant::create(element::i64, Shape{}, {2});
        auto topk =
            std::make_shared<op::v11::TopK>(data, k, 1, op::v11::TopK::Mode::MAX, op::v11::TopK::SortType::SORT_VALUES);
        auto softmax = std::make_shared<op::v8::Softmax>(topk->output(0), 1);
        model_ref = std::make_shared<Model>(ResultVector{std::make_shared<op::v0::Result>(softmax),
                                                         std::make_shared<op::v0::Result>(topk->output(1)),
                                                         std::make_shared<op::v0::Result>(topk->output(1))},
                                            ParameterVector{data});
    }
}
