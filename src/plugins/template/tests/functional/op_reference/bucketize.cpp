// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/bucketize.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"

using namespace reference_tests;
using namespace ov;

struct BucketizeParams {
    template <class IT, class BT, class OT>
    BucketizeParams(const element::Type& input_type,
                    const Shape& input_shape,
                    const std::vector<IT>& input,
                    const element::Type& bucket_type,
                    const Shape& bucket_shape,
                    const std::vector<BT>& buckets,
                    bool with_right_bound,
                    const element::Type& output_type,
                    const std::vector<OT>& expected_output)
        : input_type(input_type),
          input_shape(input_shape),
          input(CreateTensor(input_shape, input_type, input)),
          bucket_type(bucket_type),
          bucket_shape(bucket_shape),
          buckets(CreateTensor(bucket_shape, bucket_type, buckets)),
          with_right_bound(with_right_bound),
          output_type(output_type),
          expected_output(CreateTensor(input_shape, output_type, expected_output)) {}

    element::Type input_type;
    Shape input_shape;
    ov::Tensor input;
    element::Type bucket_type;
    Shape bucket_shape;
    ov::Tensor buckets;
    bool with_right_bound;
    element::Type output_type;
    ov::Tensor expected_output;
};

class ReferenceBucketizeLayerTest : public testing::TestWithParam<BucketizeParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        const auto& params = GetParam();
        function = CreateFunction(params.input_type,
                                  params.input_shape,
                                  params.bucket_type,
                                  params.bucket_shape,
                                  params.with_right_bound,
                                  params.output_type);
        inputData = {params.input, params.buckets};
        refOutData = {params.expected_output};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<BucketizeParams>& obj) {
        const auto& param = obj.param;
        std::ostringstream result;
        result << "input_type=" << param.input_type << "_";
        result << "input_shape=" << param.input_shape << "_";
        result << "bucket_type=" << param.bucket_type << "_";
        result << "bucket_shape=" << param.bucket_shape << "_";
        result << "with_right_bound=" << param.with_right_bound << "_";
        result << "output_type=" << param.output_type;
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const element::Type& input_type,
                                                 const Shape& input_shape,
                                                 const element::Type& bucket_type,
                                                 const Shape& bucket_shape,
                                                 const bool with_right_bound,
                                                 const element::Type& output_type) {
        auto data = std::make_shared<op::v0::Parameter>(input_type, input_shape);
        auto buckets = std::make_shared<op::v0::Parameter>(bucket_type, bucket_shape);
        return std::make_shared<Model>(
            std::make_shared<op::v3::Bucketize>(data, buckets, output_type, with_right_bound),
            ParameterVector{data, buckets});
    }
};

TEST_P(ReferenceBucketizeLayerTest, CompareWithHardcodedRefs) {
    Exec();
}

INSTANTIATE_TEST_SUITE_P(smoke_Bucketize_With_Hardcoded_Refs,
                         ReferenceBucketizeLayerTest,
                         ::testing::Values(
                             // fp32, int32, with_right_bound
                             BucketizeParams(element::f32,
                                             Shape{10, 1},
                                             std::vector<float>{8.f, 1.f, 2.f, 1.1f, 8.f, 10.f, 1.f, 10.2f, 0.f, 20.f},
                                             element::i32,
                                             Shape{4},
                                             std::vector<int32_t>{1, 4, 10, 20},
                                             true,
                                             element::i32,
                                             std::vector<int32_t>{2, 0, 1, 1, 2, 2, 0, 3, 0, 3}),
                             // fp32, int32, with_right_bound
                             BucketizeParams(element::i32,
                                             Shape{1, 1, 10},
                                             std::vector<int32_t>{8, 1, 2, 1, 8, 5, 1, 5, 0, 20},
                                             element::i32,
                                             Shape{4},
                                             std::vector<int32_t>{1, 4, 10, 20},
                                             false,
                                             element::i32,
                                             std::vector<int32_t>{2, 1, 1, 1, 2, 2, 1, 2, 0, 4})),
                         ReferenceBucketizeLayerTest::getTestCaseName);
