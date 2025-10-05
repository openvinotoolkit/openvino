// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/ov_test_utils.hpp"

#include "snippets/runtime_configurator.hpp"
#include "snippets/utils/utils.hpp"

namespace ov {
namespace test {
namespace snippets {

namespace {

class TestRuntimeConfigurator : public ov::snippets::RuntimeConfigurator {
public:
    TestRuntimeConfigurator() : RuntimeConfigurator(std::make_shared<ov::snippets::RuntimeConfig>()) {}

    void set_state(size_t in_num,
                   const std::vector<ov::snippets::VectorDims>& shapes,
                   const std::vector<ov::snippets::VectorDims>& latest_shapes,
                   const std::vector<std::vector<size_t>>& layouts,
                   const std::vector<size_t>& data_sizes,
                   size_t tensor_rank,
                   const std::vector<ov::snippets::VectorDims>& offsets) {
        m_in_num = in_num;
        m_io_num = shapes.size();
        m_io_data_sizes = data_sizes;
        m_config->tensor_rank = tensor_rank;
        m_config->io_shapes = shapes;
        m_config->latest_shapes = latest_shapes;
        m_config->io_layouts = layouts;
        m_config->io_data_offsets = offsets;
    }

    void run_update_data_offsets() {
        update_data_offsets();
    }
};

}  // namespace

TEST(RuntimeConfiguratorOffsets, KeepsPreviousDynamicPortsAndUpdatesLaterPorts) {
    const auto dynamic = ov::snippets::utils::get_dynamic_value<size_t>();

    TestRuntimeConfigurator configurator;
    configurator.set_state(1,
                           {ov::snippets::VectorDims{dynamic, 5}, ov::snippets::VectorDims{2, 3}},
                           {ov::snippets::VectorDims{1, 5}, ov::snippets::VectorDims{1, 3}},
                           {std::vector<size_t>{}, std::vector<size_t>{}},
                           {1, 1},
                           2,
                           {ov::snippets::VectorDims{42, 24}, ov::snippets::VectorDims{10, 10}});

    configurator.run_update_data_offsets();

    const auto config = configurator.get_config();
    ASSERT_EQ(config->io_data_offsets.size(), 2);
    EXPECT_EQ(config->io_data_offsets[0], (ov::snippets::VectorDims{42, 24}));
    EXPECT_EQ(config->io_data_offsets[1], (ov::snippets::VectorDims{3, 1}));
}

}  // namespace snippets
}  // namespace test
}  // namespace ov
