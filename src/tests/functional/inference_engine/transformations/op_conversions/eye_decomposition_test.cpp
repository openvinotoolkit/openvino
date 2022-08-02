// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <string>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "transformations/op_conversions/eye_decomposition.hpp"

using namespace testing;

/** Helper to get access EyeDecomposition protected methods. */
class EyeDecompositionWrapper : public ov::pass::EyeDecomposition {
public:
    std::shared_ptr<ov::Node> exp_eye(const ov::Output<ov::Node>& height,
                                      const ov::Output<ov::Node>& width,
                                      const ov::Output<ov::Node>& k,
                                      const ov::Output<ov::Node>& batch,
                                      ov::element::Type dtype) {
        return make_eye_model(height, width, k, dtype);
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
            throw ov::Exception("FakeEye has incorrect input number: " + std::to_string(new_args.size()));
        }
    }
};

class EyeTransformationTests : public TransformationTestsF {
protected:
    EyeDecompositionWrapper eye_decomposition_wrapper;

    ov::element::Type dtype;
    int h, w, shift;

    void SetUp() override {
        TransformationTestsF::SetUp();

        dtype = ov::element::f32;
        h = 4;
        w = 4;
    }

    template <class TEye>
    std::shared_ptr<TEye> make_test_eye(const ov::Output<ov::Node>& k) const {
        auto height = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {h});
        auto width = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {w});

        return std::make_shared<TEye>(height, width, k, dtype);
    }

    template <class TEye>
    std::shared_ptr<TEye> make_test_eye() const {
        auto k = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {shift});

        return make_test_eye<TEye>(k);
    }
};

/** \brief Diagonal shift is not `Constant`, there should be no decompose. */
TEST_F(EyeTransformationTests, shift_is_not_const) {
    {
        auto data = std::make_shared<ov::op::v0::Parameter>(dtype, ov::Shape{h, w});
        auto k = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1});
        auto node = make_test_eye<ov::op::v9::Eye>(k);

        model = std::make_shared<ov::Model>(ov::NodeVector{node}, ov::ParameterVector{data, k});

        manager.register_pass<ov::pass::EyeDecomposition>();
    }

    {
        auto data = std::make_shared<ov::op::v0::Parameter>(dtype, ov::Shape{h, w});
        auto k = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1});
        auto node = make_test_eye<ov::op::v9::Eye>(k);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{node}, ov::ParameterVector{data, k});
    }
}

/** \brief Use fake eye as not supported op type, there should be no decompose. */
TEST_F(EyeTransformationTests, use_fake_eye) {
    {
        auto data = std::make_shared<ov::op::v0::Parameter>(dtype, ov::Shape{h, w});
        auto node = make_test_eye<FakeEye>();

        model = std::make_shared<ov::Model>(ov::NodeVector{node}, ov::ParameterVector{data});

        manager.register_pass<ov::pass::EyeDecomposition>();
    }

    {
        auto data = std::make_shared<ov::op::v0::Parameter>(dtype, ov::Shape{h, w});
        auto node = make_test_eye<FakeEye>();

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{node}, ov::ParameterVector{data});
    }
}

/** \brief Diagnol shift value is not supported type, there should be no decompose. */
TEST_F(EyeTransformationTests, diagonal_shift_is_not_supported_type) {
    {
        auto data = std::make_shared<ov::op::v0::Parameter>(dtype, ov::Shape{h, w});
        auto k = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {1});
        auto node = make_test_eye<ov::op::v9::Eye>(k);

        model = std::make_shared<ov::Model>(ov::NodeVector{node}, ov::ParameterVector{data});

        manager.register_pass<ov::pass::EyeDecomposition>();
    }

    {
        auto data = std::make_shared<ov::op::v0::Parameter>(dtype, ov::Shape{4, 4});
        auto k = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {1});
        auto node = make_test_eye<ov::op::v9::Eye>(k);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{node}, ov::ParameterVector{data});
    }
}

using EyeTestParameters = std::tuple<ov::element::Type,     // Eye element type
                                     std::tuple<int, int>,  // Eye dimensions (height, width)
                                     int                    // diagonal shift
                                     >;

class EyeTransformationTestsP : public EyeTransformationTests, public WithParamInterface<EyeTestParameters> {
protected:
    void SetUp() override {
        TransformationTestsF::SetUp();

        std::tuple<int, int> dim;
        std::tie(dtype, dim, shift) = GetParam();
        std::tie(h, w) = dim;
    }
};

INSTANTIATE_TEST_SUITE_P(eye_no_diagonal_shift,
                         EyeTransformationTestsP,
                         Combine(Values(ov::element::i32, ov::element::f32, ov::element::u8),
                                 Combine(Range(0, 10, 2), Range(0, 10, 2)),
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
                                 Combine(Range(6, 10, 2), Range(6, 10, 2)),
                                 Values(-30, -11, 11, 25)),
                         PrintToStringParamName());

/** \brief Test eye decomposition for different data types, dimension and diagonal shift. */
TEST_P(EyeTransformationTestsP, eye_decompose) {
    {
        auto data = std::make_shared<ov::op::v0::Parameter>(dtype, ov::Shape{h, w});
        auto node = make_test_eye<ov::op::v9::Eye>();

        model = std::make_shared<ov::Model>(ov::NodeVector{node}, ov::ParameterVector{data});

        manager.register_pass<ov::pass::EyeDecomposition>();
    }

    {
        auto data = std::make_shared<ov::op::v0::Parameter>(dtype, ov::Shape{h, w});
        auto height = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {h});
        auto width = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {w});
        auto k = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {shift});

        auto node = eye_decomposition_wrapper.exp_eye(height, width, k, {}, dtype);
        model_ref = std::make_shared<ov::Model>(ov::NodeVector{node}, ov::ParameterVector{data});
    }

    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}
