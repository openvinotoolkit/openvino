//
// Created by Nico Galoppo on 7/20/21.
//

#include "test_utils.h"

#include <cldnn/runtime/event.hpp>
#include <cldnn/primitives/generic_primitive.hpp>

using namespace cldnn;
using namespace ::tests;

TEST(generic_primitive_f32, add_basic_in2x2x2x2) {
    //  Input2   : 2x2x2
    //  Input  : 2x2x2x2
    //  Output : 2x2x2x2

    //  Input:
    //  f0: b0:  1    2  b1:   0    0
    //  f0: b0:  3    4  b1:   0.5 -0.5
    //  f1: b0:  5    6  b1:   1.5  5.2
    //  f1: b0:  7    8  b1:   12   8
    //
    //  Input2
    //  f0: b0: 0.5  5   b1: 2.5  7
    //  f0: b0: 15  -2   b1: 17   6.5
    //  f1: b0: 0.5  2   b1: 2.5  4
    //  f1: b0: 8   -0.5 b1: 10   -2.5
    //
    //  Output:
    //  f0: b0:   1.5  7    b1:  2.5   7
    //  f0: b0:   18   2    b1:  17.5  6
    //  f1: b0:   5.5  8    b1:   4    9.2
    //  f1: b0:   15  16.5  b1:  22    16.5
    //

    auto &engine = get_test_engine();

    auto input =
        engine.allocate_memory({data_types::f32, format::yxfb, {2, 2, 2, 2}});
    auto input2 =
        engine.allocate_memory({data_types::f32, format::yxfb, {2, 2, 2, 2}});

    generic_primitive::execute_function f = [](const std::vector<event::ptr>& dependent_events,
                                               const std::vector<memory::ptr>& inputs,
                                               const std::vector<memory::ptr>& outputs) {
      stream& stream = inputs[0]->get_engine()->get_program_stream();

      for (auto& ev : dependent_events) {
        ev->wait();
      }

      cldnn::event::ptr ev = stream.create_user_event(false);

      mem_lock<float> input1(inputs[0], stream);
      mem_lock<float> input2(inputs[1], stream);
      mem_lock<float> output(outputs[0], stream);

      float* p_in1 = input1.data();
      float* p_in2 = input2.data();
      float* out   = output.data();

      for (size_t i = 0; i < input1.size(); ++i)
      {
        out[i] = p_in1[i] + p_in2[i];
      }

      ev->set();
      return ev;
    };

    layout output_layout = {data_types::f32, format::yxfb, {2, 2, 2, 2}};

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(generic_primitive("generic_function",
                                   {"input", "input2"},
                                   f, output_layout));

    set_values(input, {1.f, 0.f, 5.f, 1.5f, 2.f, 0.f, 6.f, 5.2f, 3.f, 0.5f, 7.f,
                       12.f, 4.f, -0.5f, 8.f, 8.f});

    set_values(input2, {0.5f, 2.5f, 0.5f, 2.5f, 5.f, 7.f, 2.f, 4.f, 15.f, 17.f,
                        8.f, 10.f, -2.f, 6.5f, -0.5f, -2.5f});

    network network(engine, topology);

    network.set_input_data("input", input);
    network.set_input_data("input2", input2);
    auto outputs = network.execute();

    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "generic_function");

    auto output = outputs.at("generic_function").get_memory();

    float answers[16] = {1.5f, 2.5f,  5.5f, 4.f,  7.f, 7.f, 8.f,  9.2f,
                         18.f, 17.5f, 15.f, 22.f, 2.f, 6.f, 7.5f, 5.5f};

    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (int i = 0; i < 16; i++) {
      EXPECT_TRUE(are_equal(answers[i], output_ptr[i]));
    }
}
