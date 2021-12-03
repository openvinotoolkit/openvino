// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <split_shape_inference.hpp>

#include "utils.hpp"

using namespace ov;

static std::shared_ptr<op::v1::Split> build_split(PartialShape data_shape,
                                                  std::initializer_list<int64_t> axis_value,
                                                  size_t num_splits) {
    std::shared_ptr<ov::Node> axis;
    const auto data = std::make_shared<op::v0::Parameter>(element::f32, data_shape);
    if (axis_value.size())
        axis = op::v0::Constant::create(element::i64, ov::Shape{}, {*axis_value.begin()});
    else
        axis = std::make_shared<op::v0::Parameter>(element::i64, ov::PartialShape{});

    return std::make_shared<op::v1::Split>(data, axis, num_splits);
}

TEST(StaticShapeInferenceTest, SplitV1) {
    const auto op = build_split(PartialShape{-1, -1, -1}, {}, 3);
    check_static_shape(op.get(), {StaticShape{2, 3, 4}, 1}, {{2, 1, 4}, {2, 1, 4}, {2, 1, 4}});
}

TEST(StaticShapeInferenceTest, SplitV1_Dynamic) {
    check_output_shape(build_split(PartialShape({2, 8, 4}), {}, 4).get(),
                       {ov::PartialShape::dynamic(ov::Rank(3)),
                        ov::PartialShape::dynamic(ov::Rank(3)),
                        ov::PartialShape::dynamic(ov::Rank(3)),
                        ov::PartialShape::dynamic(ov::Rank(3))});
}

TEST(StaticShapeInferenceTest, SplitV1_StaticWithConstMap) {
    check_static_shape(build_split(PartialShape({-1, -1, -1}), {}, 4).get(),
                       {StaticShape{2, 8, 4}, 2},
                       {{2, 8, 1}, {2, 8, 1}, {2, 8, 1}, {2, 8, 1}});
}

template <typename Callable>
static void perf_test(Callable func) {
    static int perf_test_N = std::getenv("PERFN") ? std::atoi(std::getenv("PERFN")) : 1;
    std::chrono::time_point<std::chrono::high_resolution_clock> before;
    std::chrono::time_point<std::chrono::high_resolution_clock> after;
    std::vector<std::chrono::nanoseconds::rep> diffs;

    if (perf_test_N == 1) {
        func();
        return;
    }

    for (size_t i = 0; i < perf_test_N; ++i) {
        before = std::chrono::high_resolution_clock::now();
        func();
        after = std::chrono::high_resolution_clock::now();
        auto diff = std::chrono::duration_cast<std::chrono::nanoseconds>(after - before).count();
        diffs.push_back(diff);
    }

    auto drop_percentage = 20;
    std::sort(diffs.begin(), diffs.end());
    auto skip = diffs.size() * drop_percentage / 2 / 100;
    auto sum = std::accumulate(diffs.begin() + skip, diffs.end() - skip, 0);
    auto avg = sum / (diffs.size() - 2 * skip);

    std::cout << " avg:" << avg << std::endl;
}

#define PERF_TEST(st) \
    perf_test([&]() { \
        st;           \
    })

using VectorDims = std::vector<size_t>;

std::vector<VectorDims> shapeInfer0(const op::v1::Split* op, VectorDims* dims_in, int dims_cnt, int64_t* axes_values) {
    NODE_VALIDATION_CHECK(op, (dims_cnt == 2));

    const auto& data_ps = dims_in[0];
    const auto& axis_ps = dims_in[1];

    NODE_VALIDATION_CHECK(op, axis_ps.size() == 0, "'axis' input must be a scalar. Got: ", axis_ps.size());

    auto each_output_shape = data_ps;
    const auto data_rank = data_ps.size();

    auto num_splits = op->get_num_splits();

    auto axis = ov::normalize_axis(op, axes_values[0], ov::Rank(data_rank));

    const auto dimension_at_axis = data_ps[axis];

    NODE_VALIDATION_CHECK(op,
                          dimension_at_axis % num_splits == 0,
                          "Dimension of data input shape along 'axis': ",
                          dimension_at_axis,
                          " must be evenly divisible by 'num_splits' attribute value: ",
                          num_splits);

    each_output_shape[axis] = dimension_at_axis / num_splits;

    std::vector<VectorDims> ret;
    for (size_t i = 0; i < num_splits; ++i)
        ret.push_back(each_output_shape);

    return ret;
}

std::vector<VectorDims> shapeInfer1(const op::v1::Split* op, VectorDims* dims_in, int dims_cnt, int64_t* axes_values) {
    std::vector<StaticShape> input_shapes = {dims_in[0], dims_in[1]};

    std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>> input_values = {
        {1, std::make_shared<ngraph::runtime::HostTensor>(ngraph::element::Type_t::i32, ov::Shape{}, axes_values)}};

    std::vector<StaticShape> output_shapes = {StaticShape{}};

    shape_infer(op, input_shapes, output_shapes, input_values);

    std::vector<VectorDims> result(output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), result.begin(), [](const ov::StaticShape& s) {
        return s.to_shape();
    });
    return result;
}

template <typename T>
class span {
    T* ptr_;
    std::size_t len_;

public:
    span(T* ptr, std::size_t len) noexcept : ptr_{ptr}, len_{len} {}

    T& operator[](int i) noexcept {
        return *ptr_[i];
    }

    T const& operator[](int i) const noexcept {
        return *ptr_[i];
    }

    std::size_t size() const noexcept {
        return len_;
    }

    T* begin() noexcept {
        return ptr_;
    }

    T* end() noexcept {
        return ptr_ + len_;
    }
};

TEST(StaticShapeInferenceTest, SplitV1_S2) {
    std::vector<VectorDims> dims_in = {{2, 8, 4}, {}};
    std::vector<VectorDims> ret;
    int64_t axes_values[2] = {2, 2};
    auto op = build_split(PartialShape({2, 8, 4}), {}, 4);

    PERF_TEST(ret = shapeInfer0(op.get(), dims_in.data(), dims_in.size(), axes_values));
    PERF_TEST(ret = shapeInfer1(op.get(), dims_in.data(), dims_in.size(), axes_values));
}
