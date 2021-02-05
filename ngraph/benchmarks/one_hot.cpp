#include <numeric>
#include <vector>
#include <algorithm>
#include <random>
#include <vector>

#include <benchmark/benchmark.h>
#include "ngraph/op/one_hot.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/pass/constant_folding.hpp"
#include "ngraph/pass/manager.hpp"

using namespace ngraph;
using namespace std;

static void one_hot_bench_4(benchmark::State& state)
{
    std::vector <int64_t> indices(4);
    std::random_device rnd;
    std::mt19937 gen(1);
    std::uniform_int_distribution<> dist(0.,10.);
    std::generate(indices.begin(), indices.end(), [&gen, &dist]() { return dist(gen); });
    auto on_value = 1;
    auto off_value = 0;

    const auto indices_const = op::Constant::create(element::i64, Shape{2, 2}, indices);
    const auto depth_const = op::Constant::create(element::i64, Shape{}, {2});
    const auto on_const = op::Constant::create(element::i64, Shape{}, {on_value});
    const auto off_const = op::Constant::create(element::i64, Shape{}, {off_value});
    int64_t axis = -1;

    auto ind_p = std::make_shared<HostTensor>(indices_const);
    auto depth_p = std::make_shared<HostTensor>(depth_const);
    auto on_p = std::make_shared<HostTensor>(on_const);
    auto off_p = std::make_shared<HostTensor>(off_const);
    HostTensorVector input = {ind_p, depth_p, on_p, off_p};

    auto oup = std::make_shared<HostTensor>(std::make_shared<op::v0::Constant>(element::i64, Shape{4, 4, 4}));
    HostTensorVector output = {oup};

    for (auto _ : state)
    {
        benchmark::DoNotOptimize(output[0]->get_data_ptr());
        auto one_hot_v =
                std::make_shared<op::v1::OneHot>(indices_const, depth_const, on_const, off_const, axis);
        //one_hot_v->validate_and_infer_types();
        one_hot_v->evaluate(output, input);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(one_hot_bench_4)->Unit(benchmark::kMicrosecond);

static void one_hot_bench_16(benchmark::State& state)
{
    std::vector <int64_t> indices(16);
    std::random_device rnd;
    std::mt19937 gen(1);
    std::uniform_int_distribution<> dist(0.,10.);
    std::generate(indices.begin(), indices.end(), [&gen, &dist]() { return dist(gen); });
    auto on_value = 1;
    auto off_value = 0;

    const auto indices_const = op::Constant::create(element::i64, Shape{4, 4}, indices);
    const auto depth_const = op::Constant::create(element::i64, Shape{}, {4});
    const auto on_const = op::Constant::create(element::i64, Shape{}, {on_value});
    const auto off_const = op::Constant::create(element::i64, Shape{}, {off_value});
    int64_t axis = -1;

    auto ind_p = std::make_shared<HostTensor>(indices_const);
    auto depth_p = std::make_shared<HostTensor>(depth_const);
    auto on_p = std::make_shared<HostTensor>(on_const);
    auto off_p = std::make_shared<HostTensor>(off_const);
    HostTensorVector input = {ind_p, depth_p, on_p, off_p};

    auto oup = std::make_shared<HostTensor>(std::make_shared<op::v0::Constant>(element::i64, Shape{4, 4, 4}));
    HostTensorVector output = {oup};

    for (auto _ : state)
    {
        benchmark::DoNotOptimize(output[0]->get_data_ptr());
        auto one_hot_v =
                std::make_shared<op::v1::OneHot>(indices_const, depth_const, on_const, off_const, axis);
        //one_hot_v->validate_and_infer_types();
        one_hot_v->evaluate(output, input);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(one_hot_bench_16)->Unit(benchmark::kMicrosecond);

static void one_hot_bench_64(benchmark::State& state)
{
    std::vector <int64_t> indices(64);
    std::random_device rnd;
    std::mt19937 gen(1);
    std::uniform_int_distribution<> dist(0.,10.);
    std::generate(indices.begin(), indices.end(), [&gen, &dist]() { return dist(gen); });
    auto on_value = 1;
    auto off_value = 0;

    const auto indices_const = op::Constant::create(element::i64, Shape{8, 8}, indices);
    const auto depth_const = op::Constant::create(element::i64, Shape{}, {4});
    const auto on_const = op::Constant::create(element::i64, Shape{}, {on_value});
    const auto off_const = op::Constant::create(element::i64, Shape{}, {off_value});
    int64_t axis = -1;

    auto ind_p = std::make_shared<HostTensor>(indices_const);
    auto depth_p = std::make_shared<HostTensor>(depth_const);
    auto on_p = std::make_shared<HostTensor>(on_const);
    auto off_p = std::make_shared<HostTensor>(off_const);
    HostTensorVector input = {ind_p, depth_p, on_p, off_p};

    //std::vector <int64_t> out_v(48);
    auto oup = std::make_shared<HostTensor>(std::make_shared<op::v0::Constant>(element::i64, Shape{8, 8, 4}));
    HostTensorVector output = {oup};

    for (auto _ : state)
    {
        benchmark::DoNotOptimize(output[0]->get_data_ptr());
        auto one_hot_v =
                std::make_shared<op::v1::OneHot>(indices_const, depth_const, on_const, off_const, axis);
        //one_hot_v->validate_and_infer_types();
        one_hot_v->evaluate(output, input);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(one_hot_bench_64)->Unit(benchmark::kMicrosecond);
