// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <test_utils.h>

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/range.hpp>

namespace cldnn {
namespace {

struct RangeArg {
    primitive_id name;
    memory::ptr p;
    RangeArg(data_types dt, const char name[]) :
        name { name }, p { tests::get_test_engine().allocate_memory( { dt, format::bfyx, tensor { spatial() } }) } {
    }
    void addTo(topology &t) const {
        t.add(input_layout { name, p->get_layout() });
    }
    void setData(network &n) const {
        n.set_input_data(name, p);
    }
};

struct RangeArgs {
    data_types dt;
    RangeArg start { dt, "start" };
    RangeArg stop { dt, "stop" };
    RangeArg step { dt, "step" };
    explicit RangeArgs(data_types dt) :
        dt { dt } {
    }
    memory::ptr run(int outLen) const {
        topology topology;
        start.addTo(topology);
        stop.addTo(topology);
        step.addTo(topology);
        topology.add(range { "range", { start.name, stop.name, step.name }, { dt, format::bfyx,
            tensor { spatial(outLen) } } });

        network network { tests::get_test_engine(), topology };

        start.setData(network);
        stop.setData(network);
        step.setData(network);

        auto outputs = network.execute();
        return outputs.at("range").get_memory();
    }
};

template<typename T> void doSmokeRange(T start, T stop, T step) {
    RangeArgs args { type_to_data_type<T>::value };

    tests::set_values(args.start.p, { start });
    tests::set_values(args.stop.p, { stop });
    tests::set_values(args.step.p, { step });

    T outLen = (stop - start) / step + 1;

    auto output = args.run(outLen);
    mem_lock<T> output_ptr { output, tests::get_test_stream() };

    for (std::size_t i = 0; i < outLen; ++i)
        EXPECT_EQ(start + i * step, output_ptr[i]);
}

void doSmokeRangeAllTypes(int start, int stop, int step) {
    doSmokeRange<std::int8_t>(start, stop, step);
    doSmokeRange<std::uint8_t>(start, stop, step);
    doSmokeRange<int>(start, stop, step);
    doSmokeRange<float>(start, stop, step);
    doSmokeRange<std::int64_t>(start, stop, step);
}

TEST(smoke, Range) {
    doSmokeRangeAllTypes(1, 21, 2);
    doSmokeRangeAllTypes(4, 0, -1);
}
}  // namespace
}  // namespace cldnn
