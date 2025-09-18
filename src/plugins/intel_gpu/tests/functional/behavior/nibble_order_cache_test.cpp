// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Test for CVS-161778: Nibble order issue in 4-bit types with caching

#include <gtest/gtest.h>
#include <fstream>
#include <memory>
#include <vector>

#include "openvino/openvino.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/pass/serialize.hpp"
#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/test_common.hpp"

namespace ov {
namespace test {
namespace behavior {

class NibbleOrderCacheTest : public testing::Test {
public:
    void SetUp() override {
        // Create temporary directory for cache
        cache_dir = ov::test::utils::generateTestFilePrefix() + "_nibble_cache";
        ov::test::utils::createDirectory(cache_dir);
        
        // Create paths for serialized model
        xml_path = cache_dir + "/model.xml";
        bin_path = cache_dir + "/model.bin";
    }

    void TearDown() override {
        // Clean up
        std::remove(xml_path.c_str());
        std::remove(bin_path.c_str());
        ov::test::utils::removeFilesWithExt(cache_dir, "blob");
        ov::test::utils::removeFilesWithExt(cache_dir, "cl_cache");
        ov::test::utils::removeDir(cache_dir);
    }

protected:
    std::string cache_dir;
    std::string xml_path;
    std::string bin_path;

    std::shared_ptr<ov::Model> create_u4_test_model() {
        // Create test pattern where nibble order matters
        // Pattern: 0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF
        // This packs to: [1,0,3,2,5,4,7,6,9,8,B,A,D,C,F,E] in low-high nibble order
        std::vector<uint8_t> packed_data = {0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF};
        
        // Unpack to u4 values (16 values from 8 bytes)
        std::vector<uint8_t> u4_values;
        u4_values.reserve(16);
        for (auto byte : packed_data) {
            u4_values.push_back(byte & 0x0F);        // Low nibble
            u4_values.push_back((byte >> 4) & 0x0F); // High nibble  
        }
        
        // Create u4 constant
        auto const_u4 = std::make_shared<ov::op::v0::Constant>(
            ov::element::u4, 
            ov::Shape{16}, 
            u4_values
        );
        
        // Convert to u8 to make values observable
        auto convert = std::make_shared<ov::op::v0::Convert>(const_u4, ov::element::u8);
        
        // Create result
        auto result = std::make_shared<ov::op::v0::Result>(convert);
        
        // Create model
        return std::make_shared<ov::Model>(
            ov::ResultVector{result},
            ov::ParameterVector{},
            "nibble_order_test_model"
        );
    }

    std::vector<uint8_t> run_inference(ov::CompiledModel& compiled_model) {
        auto infer_request = compiled_model.create_infer_request();
        infer_request.infer();
        
        auto output_tensor = infer_request.get_output_tensor(0);
        auto data_ptr = output_tensor.data<uint8_t>();
        auto size = output_tensor.get_size();
        
        return std::vector<uint8_t>(data_ptr, data_ptr + size);
    }
};

TEST_F(NibbleOrderCacheTest, U4ConversionWithCache) {
    // This test demonstrates the nibble order issue with u4 types and caching.
    
    auto model = create_u4_test_model();
    
    // Serialize model
    ov::pass::Serialize(xml_path, bin_path).run_on_model(model);
    
    // Create Core and set cache configuration
    ov::Core core;
    ov::AnyMap config = {
        ov::cache_dir(cache_dir),
        ov::cache_mode(ov::CacheMode::OPTIMIZE_SIZE)
    };
    
    // First compilation - creates cache
    auto compiled_model_1 = core.compile_model(xml_path, "GPU", config);
    auto result_1 = run_inference(compiled_model_1);
    
    // Expected pattern after correct unpacking: [1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14]
    // This is because bytes are: 0x01 -> [1,0], 0x23 -> [3,2], etc.
    std::vector<uint8_t> expected = {1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14};
    
    std::cout << "First compilation result: ";
    for (auto val : result_1) {
        std::cout << (int)val << " ";
    }
    std::cout << std::endl;
    
    // Clear compiled model to ensure cache is used
    compiled_model_1 = {};
    
    // Second compilation - loads from cache
    auto compiled_model_2 = core.compile_model(xml_path, "GPU", config);
    auto result_2 = run_inference(compiled_model_2);
    
    std::cout << "Second compilation result: ";
    for (auto val : result_2) {
        std::cout << (int)val << " ";
    }
    std::cout << std::endl;
    
    // Check if results match
    bool results_match = (result_1 == result_2);
    
    if (!results_match) {
        std::cout << "NIBBLE ORDER ISSUE DETECTED!" << std::endl;
        std::cout << "Results differ between cache creation and loading:" << std::endl;
        
        std::cout << "Differences at indices: ";
        for (size_t i = 0; i < result_1.size(); ++i) {
            if (result_1[i] != result_2[i]) {
                std::cout << i << " ";
            }
        }
        std::cout << std::endl;
        
        // Check if it's the specific nibble swap pattern
        bool is_nibble_swap = true;
        for (size_t i = 0; i < result_1.size(); i += 2) {
            if (i + 1 < result_1.size()) {
                // Check if pairs are swapped
                if (result_1[i] != result_2[i+1] || result_1[i+1] != result_2[i]) {
                    is_nibble_swap = false;
                    break;
                }
            }
        }
        
        if (is_nibble_swap) {
            std::cout << "Pattern matches nibble pair swapping (ABCDEF -> BADCFE)" << std::endl;
        }
    }
    
    // Test should fail if nibble order issue exists
    EXPECT_EQ(result_1, result_2) 
        << "Nibble order mismatch: Results differ between cache creation and loading";
}

TEST_F(NibbleOrderCacheTest, U4ConversionNoCacheCPUvsGPU) {
    // Compare CPU and GPU results without caching
    auto model = create_u4_test_model();
    
    ov::Core core;
    
    // Check if GPU is available
    auto devices = core.get_available_devices();
    bool has_gpu = std::find(devices.begin(), devices.end(), "GPU") != devices.end();
    
    if (!has_gpu) {
        GTEST_SKIP() << "GPU device not available";
    }
    
    // Compile for CPU
    auto compiled_cpu = core.compile_model(model, "CPU");
    auto result_cpu = run_inference(compiled_cpu);
    
    // Compile for GPU (no cache)
    auto compiled_gpu = core.compile_model(model, "GPU");
    auto result_gpu = run_inference(compiled_gpu);
    
    std::cout << "CPU result: ";
    for (auto val : result_cpu) {
        std::cout << (int)val << " ";
    }
    std::cout << std::endl;
    
    std::cout << "GPU result: ";
    for (auto val : result_gpu) {
        std::cout << (int)val << " ";
    }
    std::cout << std::endl;
    
    // Without caching, CPU and GPU should match
    EXPECT_EQ(result_cpu, result_gpu) 
        << "CPU and GPU results should match without caching";
}

TEST_F(NibbleOrderCacheTest, I4ConversionWithCache) {
    // Test with i4 type as well
    
    // Create i4 constant with pattern that shows sign extension
    std::vector<int8_t> i4_values = {0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7, -7, -8};
    
    auto const_i4 = std::make_shared<ov::op::v0::Constant>(
        ov::element::i4,
        ov::Shape{16},
        i4_values
    );
    
    auto convert = std::make_shared<ov::op::v0::Convert>(const_i4, ov::element::i8);
    auto result = std::make_shared<ov::op::v0::Result>(convert);
    auto model = std::make_shared<ov::Model>(
        ov::ResultVector{result},
        ov::ParameterVector{},
        "i4_nibble_test"
    );
    
    // Serialize model
    ov::pass::Serialize(xml_path, bin_path).run_on_model(model);
    
    ov::Core core;
    ov::AnyMap config = {
        ov::cache_dir(cache_dir),
        ov::cache_mode(ov::CacheMode::OPTIMIZE_SIZE)
    };
    
    // First compilation
    auto compiled_1 = core.compile_model(xml_path, "GPU", config);
    auto infer_1 = compiled_1.create_infer_request();
    infer_1.infer();
    auto output_1 = infer_1.get_output_tensor(0);
    std::vector<int8_t> result_1(output_1.data<int8_t>(), 
                                  output_1.data<int8_t>() + output_1.get_size());
    
    compiled_1 = {};
    
    // Second compilation (from cache)
    auto compiled_2 = core.compile_model(xml_path, "GPU", config);
    auto infer_2 = compiled_2.create_infer_request();
    infer_2.infer();
    auto output_2 = infer_2.get_output_tensor(0);
    std::vector<int8_t> result_2(output_2.data<int8_t>(), 
                                  output_2.data<int8_t>() + output_2.get_size());
    
    // Check for nibble order issue with signed values
    EXPECT_EQ(result_1, result_2) 
        << "i4 nibble order mismatch between cache creation and loading";
}

} // namespace behavior
} // namespace test  
} // namespace ov