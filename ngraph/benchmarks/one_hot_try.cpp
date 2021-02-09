#include <numeric>
#include <vector>
#include <algorithm>
#include <random>
#include <vector>

//#include <benchmark/benchmark.h>
#include "ngraph/op/one_hot.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/pass/constant_folding.hpp"
#include "ngraph/pass/manager.hpp"

using namespace ngraph;
using namespace std;

int main()
{
    std::vector <int64_t> indices = {} ;
    const int depth = 4;
    std::random_device rnd;
    std::mt19937 gen(1);
    std::uniform_int_distribution<> dist(0.,10.);
    //std::generate(indices.begin(), indices.end(), [&gen, &dist]() { return dist(gen); });
    auto on_value = 1;
    auto off_value = 0;

    const auto indices_const = op::Constant::create(element::i64, Shape{}, indices);
    const auto depth_const = op::Constant::create(element::i64, Shape{}, {depth});
    const auto on_const = op::Constant::create(element::i64, Shape{}, {on_value});
    const auto off_const = op::Constant::create(element::i64, Shape{}, {off_value});
    int64_t axis = -1;

    auto ind_p = std::make_shared<HostTensor>(indices_const);
    auto depth_p = std::make_shared<HostTensor>(depth_const);
    auto on_p = std::make_shared<HostTensor>(on_const);
    auto off_p = std::make_shared<HostTensor>(off_const);
    HostTensorVector input = {ind_p, depth_p, on_p, off_p};

    auto oup = std::make_shared<HostTensor>(std::make_shared<op::v0::Constant>(element::i64, Shape{4, depth}));
    HostTensorVector output = {oup};


    auto one_hot_v =
            std::make_shared<op::v1::OneHot>(indices_const, depth_const, on_const, off_const, axis);
    one_hot_v->validate_and_infer_types();
    one_hot_v->evaluate(output, input);


    std::cout << "Input values:\n";
    for(int i=0; i<indices.size(); ++i) {
        std::cout << indices[i] << " ";
        if ((i+1) % 1 == 0)
            std::cout << "\n";
    }
    std::cout << "Output values:\n";
    auto out_print = (int64_t*)(output[0]->get_data_ptr());
    for(int i=0; i<indices.size()*depth; ++i) {
        std::cout << out_print[i] << " ";
        if ((i+1) % depth == 0)
            std::cout << "\n";
    }
    std::cout << sizeof (long long int) << "\n";
    long long int val=-(1<<28);
    size_t one_hot_pos = static_cast<unsigned int>(val);
    std::cout << "Val:"  << val << " | " << "Cast: " << one_hot_pos << "\n";
    return 0;
}