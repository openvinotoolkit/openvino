#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/convolution.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/reorder.hpp>

using namespace cldnn;
using namespace ::tests;

TEST(onednn_tests, profiling) {
    auto& engine = get_test_engine();

    if (!engine.get_device_info().supports_immad)
        return;

    auto input = engine.allocate_memory({ data_types::f16, format::bfyx, { 1, 32, 64, 64 } });
    auto weights = engine.allocate_memory({ data_types::f16, format::bfyx, { 32, 32, 3, 3 } });

    topology topology;
    topology.add(data("weights", weights));
    topology.add(input_layout("input", input->get_layout()));
    topology.add(reorder("reorder", input_info("input"), format::b_fs_yx_fsv16, data_types::f16));
    topology.add(convolution("conv", input_info("reorder"), "weights", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));

    ExecutionConfig config = get_test_default_config(engine);
    ov::intel_gpu::ImplementationDesc impl = { format::b_fs_yx_fsv16, std::string(""), impl_types::onednn };
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"conv", impl} }));
    config.set_property(ov::enable_profiling(true));

    network network(engine, topology, config);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    outputs.at("conv").get_memory();

    auto executed_primitives = network.get_executed_primitives();
    ASSERT_NE(executed_primitives.find("conv"), executed_primitives.end());

    auto conv_ev = executed_primitives["conv"];
    ASSERT_NE(conv_ev, nullptr);

    auto intervals = conv_ev->get_profiling_info();
    ASSERT_EQ(intervals.size(), 1);
    ASSERT_EQ(intervals[0].stage, instrumentation::profiling_stage::executing);
    ASSERT_NE(intervals[0].value, nullptr);
}
