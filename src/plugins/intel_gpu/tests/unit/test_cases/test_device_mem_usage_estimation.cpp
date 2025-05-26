// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstddef>

#include "test_utils.h"
#include <intel_gpu/primitives/permute.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include <openvino/runtime/core.hpp>
#include "openvino/op/matmul.hpp"
#include "openvino/runtime/intel_gpu/remote_properties.hpp"

using namespace cldnn;
using namespace tests;

class test_device_mem_usage_estimation: public ::testing::Test {
public:
    void test_basic(bool is_caching_test) {
        ExecutionConfig cfg = get_test_default_config(get_test_engine());
        cfg.set_property(ov::intel_gpu::queue_type(QueueTypes::out_of_order));

        std::shared_ptr<cldnn::engine> engine1 = create_test_engine();
        if (engine1->get_device_info().supports_immad) {
            // Enable this test for out_of_order queue-type if Onednn supports out_of_order
            return;
        }

        auto input1 = engine1->allocate_memory({ data_types::f16, format::bfyx,{ 2, 2, 256, 256} });
        auto input2 = engine1->allocate_memory({ data_types::f16, format::bfyx,{ 2, 2, 256, 256} });
        topology topology(
            input_layout("input1", input1->get_layout()),
            input_layout("input2", input2->get_layout()),
            permute("permute1", input_info("input1"), { 0, 3, 1, 2 }),
            permute("permute2", input_info("input2"), { 0, 2, 1, 3 }),
            eltwise("eltw", { input_info("permute1"), input_info("permute2") }, eltwise_mode::sum, data_types::f16),
            reorder("output", input_info("eltw"), format::bfyx, data_types::f32)
        );

        auto prog = program::build_program(*engine1, topology, cfg);
        std::pair<int64_t, int64_t> estimated_mem_usage = prog->get_estimated_device_mem_usage();

        std::shared_ptr<cldnn::engine> engine2 = create_test_engine();
        auto input3 = engine2->allocate_memory({ data_types::f16, format::bfyx,{ 2, 2, 256, 256} });
        auto input4 = engine2->allocate_memory({ data_types::f16, format::bfyx,{ 2, 2, 256, 256} });

        cldnn::network::ptr network = get_network(*engine2, topology, cfg, get_test_stream_ptr(), is_caching_test);
        network->set_input_data("input1", input3);
        network->set_input_data("input2", input4);
        ASSERT_EQ(estimated_mem_usage.first + estimated_mem_usage.second, engine2->get_used_device_memory(allocation_type::usm_device));
    }

    std::shared_ptr<ov::Model> make_custom_net(ov::PartialShape input_shape, ov::element::Type type) {
        auto parameter = std::make_shared<ov::op::v0::Parameter>(type, input_shape);
        auto mul_const = ov::op::v0::Constant::create(type, {1, 224, 1}, {0});
        auto mul = std::make_shared<ov::op::v0::MatMul>(parameter, mul_const);
        auto res = std::make_shared<ov::op::v0::Result>(mul);
        auto model = std::make_shared<ov::Model>(ov::ResultVector({res}), ov::ParameterVector{parameter});
        return model;
    }

    void get_max_batch_size() {
        ov::Core ie;
        auto& engine = get_test_engine();
        uint32_t batch_size = 0, batch_size_native = 0;
        uint32_t n_streams = 1;
        std::string target_device = "GPU";
        auto simpleNetwork = make_custom_net(ov::PartialShape({16,1,1,224}), ov::element::Type("i8"));
        auto exec_net = ie.compile_model(simpleNetwork, target_device);
        auto max_global_mem_size = engine.get_device_info().max_global_mem_size;
        unsigned int alloc_size = 1024*100;
        auto ctx = exec_net.get_context();
        std::vector<ov::RemoteTensor> v;
        v.push_back(ctx.create_tensor(ov::element::Type("i8"), ov::Shape({1,1,1, alloc_size}), {ov::intel_gpu::shared_mem_type(ov::intel_gpu::SharedMemType::USM_DEVICE_BUFFER)}));
        ov::AnyMap _options_native = {ov::hint::model(simpleNetwork),
                                      ov::num_streams(n_streams)};

        OV_ASSERT_NO_THROW(batch_size_native = ie.get_property(target_device, ov::max_batch_size.name(), _options_native).as<unsigned int>());
        
        auto statistic_result = ie.get_property(target_device, ov::intel_gpu::memory_statistics.name()).as<std::map<std::string, uint64_t>>();
        std::ostringstream usm_device_oss;
        usm_device_oss << cldnn::allocation_type::usm_device;
        auto occupied_usm_dev = statistic_result.find(usm_device_oss.str());
        auto occupied_device_mem = occupied_usm_dev->second;

        auto available_device_memory = max_global_mem_size - occupied_device_mem;
        
        ov::AnyMap _options = {ov::hint::model(simpleNetwork),
                               ov::num_streams(n_streams),
                               ov::intel_gpu::hint::available_device_mem(available_device_memory)};

        OV_ASSERT_NO_THROW(batch_size = ie.get_property(target_device, ov::max_batch_size.name(), _options).as<unsigned int>());
        GTEST_ASSERT_EQ(batch_size_native, batch_size);
        GTEST_ASSERT_NE(batch_size_native, 1);
    }
};

TEST_F(test_device_mem_usage_estimation, max_batch_size) {
    this->get_max_batch_size();
}

TEST_F(test_device_mem_usage_estimation, basic) {
    this->test_basic(false);
}

TEST_F(test_device_mem_usage_estimation, basic_cached) {
    this->test_basic(true);
}
