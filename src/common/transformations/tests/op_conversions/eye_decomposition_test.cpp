// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/eye_decomposition.hpp"

#include <memory>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "gtest/gtest.h"
#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/opsets/opset9.hpp"
using namespace ov;
using namespace testing;

/** Helper to get access EyeDecomposition protected methods. */
class EyeDecompositionWrapper : public ov::pass::EyeDecomposition {
public:
    std::shared_ptr<ov::Node> exp_eye(const ov::Output<ov::Node>& height,
                                      const ov::Output<ov::Node>& width,
                                      const ov::Output<ov::Node>& k,
                                      ov::element::Type dtype) {
        const auto zero_int = ov::opset9::Constant::create(ov::element::i64, ov::Shape{1}, {0});
        const auto zero = ov::opset9::Constant::create(dtype, ov::Shape{1}, {0});
        const auto one = ov::opset9::Constant::create(dtype, ov::Shape{1}, {1});

        const auto k_neg = std::make_shared<ov::opset9::Negative>(k);
        const auto k_axis = std::make_shared<ov::opset9::Concat>(ov::OutputVector{k_neg, k}, 0);

        const auto eye_shape = std::make_shared<ov::opset9::Concat>(ov::OutputVector{height, width}, 0);

        // Calculate eye zero padding and internal square eye size.
        const auto pad_start =
            std::make_shared<ov::opset9::Minimum>(eye_shape, std::make_shared<ov::opset9::Maximum>(zero_int, k_axis));
        const auto shape_pad_diff = std::make_shared<ov::opset9::Subtract>(eye_shape, pad_start);
        const auto eye_size = std::make_shared<ov::opset9::ReduceMin>(shape_pad_diff, zero_int, true);
        const auto pad_end = std::make_shared<ov::opset9::Subtract>(shape_pad_diff, eye_size);

        // Make 1d-eye as eye_size times of (1, zeros(eye_size)), trimmed at end by eye_size elements.
        const auto zeros = std::make_shared<ov::opset9::Tile>(zero, eye_size);
        const auto one_followed_by_zeros = std::make_shared<ov::opset9::Concat>(ov::OutputVector{one, zeros}, 0);
        const auto eye_1d =
            std::make_shared<ov::opset9::Pad>(std::make_shared<ov::opset9::Tile>(one_followed_by_zeros, eye_size),
                                              zero_int,
                                              std::make_shared<ov::opset9::Negative>(eye_size),
                                              ov::op::PadMode::CONSTANT);
        // Reshape 1d-eye to 2d-eye
        const auto eye_2d = std::make_shared<ov::opset9::Reshape>(
            eye_1d,
            std::make_shared<ov::opset9::Concat>(ov::OutputVector{eye_size, eye_size}, 0),
            false);

        // Pad Eye to get final shape
        return std::make_shared<ov::opset9::Pad>(eye_2d, pad_start, pad_end, ov::op::PadMode::CONSTANT);
    }

    std::shared_ptr<ov::Node> exp_eye(const ov::Output<ov::Node>& height,
                                      const ov::Output<ov::Node>& width,
                                      const ov::Output<ov::Node>& k,
                                      const ov::Output<ov::Node>& batch,
                                      ov::element::Type dtype) {
        const auto eye = exp_eye(height, width, k, dtype);
        const auto eye_tile = std::make_shared<ov::opset9::Constant>(ov::element::i64, ov::Shape{2}, 1);

        // `batch_repeats` repeat eye matrix as tile only in higher dimensions than 1 by number(s) in batch parameter.
        const auto batch_repeats = std::make_shared<ov::opset9::Concat>(ov::OutputVector{batch, eye_tile}, 0);

        return std::make_shared<ov::opset9::Tile>(eye, batch_repeats);
    }
};

class FakeEye : public ov::op::Op {
public:
    FakeEye() = default;

    FakeEye(const ov::Output<ov::Node>& num_rows,
            const ov::Output<ov::Node>& num_columns,
            const ov::Output<ov::Node>& diagonal_index,
            const ov::Output<ov::Node>& batch_shape,
            const ov::element::Type& out_type)
        : Op({num_rows, num_columns, diagonal_index, batch_shape}) {
        constructor_validate_and_infer_types();
    }

    FakeEye(const ov::Output<ov::Node>& num_rows,
            const ov::Output<ov::Node>& num_columns,
            const ov::Output<ov::Node>& diagonal_index,
            const ov::element::Type& out_type)
        : Op({num_rows, num_columns, diagonal_index}) {
        constructor_validate_and_infer_types();
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override {
        check_new_args_count(this, new_args);
        if (new_args.size() == 3) {
            return std::make_shared<FakeEye>(new_args[0], new_args[1], new_args[2], ov::element::f32);
        } else if (new_args.size() == 4) {
            return std::make_shared<FakeEye>(new_args[0], new_args[1], new_args[2], new_args[3], ov::element::f32);
        } else {
            OPENVINO_THROW("FakeEye has incorrect input number: ", new_args.size());
        }
    }
};

class EyeTransformationTests : public TransformationTestsF {
protected:
    EyeDecompositionWrapper eye_decomposition_wrapper;

    ov::element::Type dtype;
    size_t h, w;
    int shift;

    void SetUp() override {
        TransformationTestsF::SetUp();

        dtype = ov::element::f32;
        h = 4;
        w = 4;
    }

    template <class TEye>
    std::shared_ptr<TEye> make_test_eye(const ov::Output<ov::Node>& k) const {
        auto height = ov::opset9::Constant::create(ov::element::i64, ov::Shape{1}, {h});
        auto width = ov::opset9::Constant::create(ov::element::i64, ov::Shape{1}, {w});

        return std::make_shared<TEye>(height, width, k, dtype);
    }

    template <class TEye>
    std::shared_ptr<TEye> make_test_eye() const {
        auto k = ov::opset9::Constant::create(ov::element::i64, ov::Shape{1}, {shift});

        return make_test_eye<TEye>(k);
    }

    template <class TEye>
    std::shared_ptr<TEye> make_test_eye_batch(const ov::Output<ov::Node>& batch) const {
        auto height = ov::opset9::Constant::create(ov::element::i64, ov::Shape{1}, {h});
        auto width = ov::opset9::Constant::create(ov::element::i64, ov::Shape{1}, {w});
        auto k = ov::opset9::Constant::create(ov::element::i64, ov::Shape{1}, {shift});

        return std::make_shared<TEye>(height, width, k, batch, dtype);
    }
};

/** \brief Diagonal shift is not `Constant`, there should be no decompose. */
TEST_F(EyeTransformationTests, shift_is_not_const) {
    {
        auto data = std::make_shared<ov::opset9::Parameter>(dtype, ov::Shape{h, w});
        auto k = std::make_shared<ov::opset9::Parameter>(ov::element::i64, ov::Shape{1});
        auto node = make_test_eye<ov::opset9::Eye>(k);

        model = std::make_shared<ov::Model>(ov::NodeVector{node}, ov::ParameterVector{data, k});

        manager.register_pass<ov::pass::EyeDecomposition>();
    }
}

/** \brief Batch size is not `Constant`, there should be no decompose. */
TEST_F(EyeTransformationTests, batch_is_not_const) {
    {
        auto data = std::make_shared<ov::opset9::Parameter>(dtype, ov::Shape{h, w});
        auto batch = std::make_shared<ov::opset9::Parameter>(ov::element::i64, ov::Shape{2});
        auto node = make_test_eye_batch<ov::opset9::Eye>(batch);

        model = std::make_shared<ov::Model>(ov::NodeVector{node}, ov::ParameterVector{data, batch});

        manager.register_pass<ov::pass::EyeDecomposition>();
    }
}

/** \brief Use fake eye as not supported op type, there should be no decompose. */
TEST_F(EyeTransformationTests, use_fake_eye) {
    {
        auto data = std::make_shared<ov::opset9::Parameter>(dtype, ov::Shape{h, w});
        auto node = make_test_eye<FakeEye>();

        model = std::make_shared<ov::Model>(ov::NodeVector{node}, ov::ParameterVector{data});

        manager.register_pass<ov::pass::EyeDecomposition>();
    }
}

using EyeTestParameters = std::tuple<ov::element::Type,           // Eye element type
                                     std::tuple<size_t, size_t>,  // Eye dimensions (height, width)
                                     int                          // diagonal shift
                                     >;

class EyeTransformationTestsP : public EyeTransformationTests, public WithParamInterface<EyeTestParameters> {
protected:
    void SetUp() override {
        TransformationTestsF::SetUp();

        std::tuple<size_t, size_t> dim;
        std::tie(dtype, dim, shift) = GetParam();
        std::tie(h, w) = dim;
    }
};

INSTANTIATE_TEST_SUITE_P(eye_no_diagonal_shift,
                         EyeTransformationTestsP,
                         Combine(Values(ov::element::i32, ov::element::f32, ov::element::u8),
                                 Combine(Range<size_t>(0, 10, 2), Range<size_t>(0, 10, 2)),
                                 Values(0)),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(square_eye_diagonal_shift_within_dim,
                         EyeTransformationTestsP,
                         Combine(Values(ov::element::i32, ov::element::f32),
                                 Values(std::make_tuple(4, 4)),
                                 Range(-4, 5)),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(rectangular_narrow_eye_diagonal_shift_within_dim,
                         EyeTransformationTestsP,
                         Combine(Values(ov::element::i32, ov::element::f32),
                                 Values(std::make_tuple(7, 3)),
                                 Range(-7, 4)),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(rectangular_wide_eye_diagonal_shift_within_dim,
                         EyeTransformationTestsP,
                         Combine(Values(ov::element::i32, ov::element::f32),
                                 Values(std::make_tuple(2, 4)),
                                 Range(-2, 5)),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(eye_diagonal_shift_outside_dim,
                         EyeTransformationTestsP,
                         Combine(Values(ov::element::f32),
                                 Combine(Range<size_t>(6, 10, 2), Range<size_t>(6, 10, 2)),
                                 Values(-30, -11, 11, 25)),
                         PrintToStringParamName());

/** \brief Test eye decomposition for different data types, dimension and diagonal shift. */
TEST_P(EyeTransformationTestsP, eye_decompose) {
    {
        auto data = std::make_shared<ov::opset9::Parameter>(dtype, ov::Shape{h, w});
        auto node = make_test_eye<ov::opset9::Eye>();

        model = std::make_shared<ov::Model>(ov::NodeVector{node}, ov::ParameterVector{data});

        manager.register_pass<ov::pass::EyeDecomposition>();
    }

    {
        auto data = std::make_shared<ov::opset9::Parameter>(dtype, ov::Shape{h, w});
        auto height = ov::opset9::Constant::create(ov::element::i64, ov::Shape{1}, {h});
        auto width = ov::opset9::Constant::create(ov::element::i64, ov::Shape{1}, {w});
        auto k = ov::opset9::Constant::create(ov::element::i64, ov::Shape{1}, {shift});

        auto node = eye_decomposition_wrapper.exp_eye(height, width, k, dtype);
        model_ref = std::make_shared<ov::Model>(ov::NodeVector{node}, ov::ParameterVector{data});
    }

    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

class BatchEyeTransformationTests : public EyeTransformationTests, public WithParamInterface<std::vector<size_t>> {};

INSTANTIATE_TEST_SUITE_P(batch_size,
                         BatchEyeTransformationTests,
                         Values(std::vector<size_t>{1},
                                std::vector<size_t>{2},
                                std::vector<size_t>{2, 1},
                                std::vector<size_t>{2, 3},
                                std::vector<size_t>{3, 5, 1},
                                std::vector<size_t>{3, 5, 4}),
                         PrintToStringParamName());

/** \brief Test eye decomposition for batch sizes and values. */
TEST_P(BatchEyeTransformationTests, eye_decompose) {
    {
        auto data = std::make_shared<ov::opset9::Parameter>(dtype, ov::Shape{h, w});
        auto batch = ov::opset9::Constant::create(ov::element::i64, ov::Shape{GetParam().size()}, GetParam());
        auto node = make_test_eye_batch<ov::opset9::Eye>(batch);

        model = std::make_shared<ov::Model>(ov::NodeVector{node}, ov::ParameterVector{data});

        manager.register_pass<ov::pass::EyeDecomposition>();
    }

    {
        auto data = std::make_shared<ov::opset9::Parameter>(dtype, ov::Shape{h, w});
        auto height = ov::opset9::Constant::create(ov::element::i64, ov::Shape{1}, {h});
        auto width = ov::opset9::Constant::create(ov::element::i64, ov::Shape{1}, {w});
        auto k = ov::opset9::Constant::create(ov::element::i64, ov::Shape{1}, {shift});
        auto batch = ov::opset9::Constant::create(ov::element::i64, ov::Shape{GetParam().size()}, GetParam());

        auto node = eye_decomposition_wrapper.exp_eye(height, width, k, batch, dtype);
        model_ref = std::make_shared<ov::Model>(ov::NodeVector{node}, ov::ParameterVector{data});
    }

    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}
