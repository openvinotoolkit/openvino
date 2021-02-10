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

#define mode64
#ifdef mode64
#define out_type int64_t
#define out_elem i64
#endif

int main()
{
    /*
    std::vector <int64_t> indices = {0, 2, 1, 3} ;
    int depth = 4;
    std::random_device rnd;
    std::mt19937 gen(1);
    std::uniform_int_distribution<> dist(0.,10.);
    //std::generate(indices.begin(), indices.end(), [&gen, &dist]() { return dist(gen); });
    out_type on_value = 1;
    out_type off_value = 0;

    const auto indices_const = op::Constant::create(element::i64, Shape{4}, indices);
    const auto depth_const = op::Constant::create(element::i32, Shape{}, {depth});
    const auto on_const = op::Constant::create(element::out_elem, Shape{}, {on_value});
    const auto off_const = op::Constant::create(element::out_elem, Shape{}, {off_value});
    int64_t axis = -1;

    auto ind_p = std::make_shared<HostTensor>(indices_const);
    auto depth_p = std::make_shared<HostTensor>(depth_const);
    auto on_p = std::make_shared<HostTensor>(on_const);
    auto off_p = std::make_shared<HostTensor>(off_const);
    HostTensorVector input = {ind_p, depth_p, on_p, off_p};

    auto oup = std::make_shared<HostTensor>(std::make_shared<op::v0::Constant>(element::out_elem, Shape{4, 4}));
    HostTensorVector output = {oup};

    std::cout << "Input values:\n";
    for(int i=0; i<indices.size(); ++i) {
        std::cout << indices[i] << " ";
        if ((i+1) % 1 == 0)
            std::cout << "\n";
    }

    {
        auto one_hot_v =
                std::make_shared<op::v1::OneHot>(indices_const, depth_const, on_const, off_const, axis);
        one_hot_v->validate_and_infer_types();
        one_hot_v->evaluate_old(output, input);

        std::cout << "OLD output values:\n";
        auto out_print = static_cast<out_type *> (output[0]->get_data_ptr());
        for (int i = 0; i < indices.size() * depth; ++i) {
            std::cout << out_print[i] << " ";
            if ((i + 1) % depth == 0)
                std::cout << "\n";
        }
    }
    {
        auto one_hot_v =
                std::make_shared<op::v1::OneHot>(indices_const, depth_const, on_const, off_const, axis);
        one_hot_v->validate_and_infer_types();
        one_hot_v->evaluate(output, input);

        std::cout << "\nNEW output values:\n";
        auto out_print = static_cast<out_type *> (output[0]->get_data_ptr());
        for (int i = 0; i < indices.size() * depth; ++i) {
            std::cout << out_print[i] << " ";
            if ((i + 1) % depth == 0)
                std::cout << "\n";
        }
    }

    */
    std::cout << sizeof (long long int) << "\n";
    long long int val=-(1<<28);
    size_t one_hot_pos = static_cast<unsigned int>(val);
    std::cout << "Val:"  << val << " | " << "Cast: " << one_hot_pos << "\n";
    std::cout << "sizeof(bool): " << sizeof (bool ) << "\n";
    std::cout << "sizeof(char): " << sizeof (char ) << "\n";
    return 0;
}