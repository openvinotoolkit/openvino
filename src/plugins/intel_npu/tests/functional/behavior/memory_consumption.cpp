// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <chrono>
#include <common_test_utils/common_utils.hpp>
#include <common_test_utils/subgraph_builders/llm_builders.hpp>
#include <common_test_utils/test_constants.hpp>
#include <fstream>
#include <map>
#include <openvino/runtime/core.hpp>
#include <openvino/runtime/intel_npu/properties.hpp>
#include <openvino/util/file_path.hpp>
#include <thread>
#include <unordered_set>

enum class MemoryCaptureEvent {
    COMPILE,
    COMPILE_CACHED,
    IMPORT_MODEL_ISTREAM,
    IMPORT_MODEL_TENSOR,
    CREATE_INFER_REQUEST,
    INFER,
    IDLE
};

void print_memory_stats_for_event(const std::string& event,
                                  double peakMemoryMB,
                                  double currentMemoryMB,
                                  double deltaMemoryMB);

void print_memory_stats_for_event(const std::string& event,
                                  double peakMemoryMB,
                                  double currentMemoryMB,
                                  double deltaMemoryMB) {
    std::cout << "Peak Working Set size during " << event << " = " << peakMemoryMB << " MB" << std::endl;
    std::cout << "Working Set Size after " << event << " = " << currentMemoryMB << " MB" << std::endl;
    std::cout << "Delta Working Set Size after " << event << " = " << currentMemoryMB - deltaMemoryMB << " MB"
              << std::endl;
    std::cout << std::endl;
}

namespace ov {
namespace test {
namespace behavior {

class OVMemoryConsumptionNPU : public ::testing::Test {};

using compatibility_OVMemoryConsumptionNPU3720 = OVMemoryConsumptionNPU;

TEST_F(compatibility_OVMemoryConsumptionNPU3720, MemoryConsistentForBigLLMModel) {
    ov::Core core;
    core.set_property({intel_npu::bypass_umd_caching(true), intel_npu::defer_weights_load(true)});
    auto model = ov::test::utils::make_llm_kv_cache_sdpa_pattern(/* batch = */ 1,
                                                                 /* n_heads = */ 256,
                                                                 /* k_features = */ 512,
                                                                 /* v_features = */ 512,
                                                                 /* element_type = */ ov::element::f16,
                                                                 /* qkv_order = */ {0, 1, 2, 3},
                                                                 /* causal = */ true,
                                                                 /* with_mask =  */ true,
                                                                 /* with_scale = */ true,
                                                                 /* stateful = */ false,
                                                                 /* fuse_cache_reorder = */ false,
                                                                 /* num_groups = */ 1);
    constexpr size_t sequenceLength = 512;
    double paramsTotalSizeMB = 0.0;
    double resultsTotalSizeMB = 0.0;

    std::map<std::string, ov::PartialShape> reshapeMap;
    for (const auto& param : model->get_parameters()) {
        param->get_output_tensor(0).set_names(std::unordered_set<std::string>{param->get_friendly_name()});
        auto shape = param->get_partial_shape();
        shape[2] = sequenceLength;
        reshapeMap.insert({param->get_friendly_name(), shape});
        paramsTotalSizeMB += ov::shape_size(shape.get_shape()) * param->get_element_type().size();
    }
    paramsTotalSizeMB /= (1024.0 * 1024.0);

    model->reshape(reshapeMap);

    for (const auto& result : model->get_results()) {
        resultsTotalSizeMB += ov::shape_size(result->get_shape()) * result->get_element_type().size();
    }
    resultsTotalSizeMB /= (1024.0 * 1024.0);

    std::cout << "Model params total size = " << paramsTotalSizeMB << " MB" << std::endl;
    std::cout << "Model results total size = " << resultsTotalSizeMB << " MB" << std::endl;

    double blobSizeMB = 0.0;
    {
        // first compile not measurable, just take blob size
        std::stringstream ss;
        auto compiledModel = core.compile_model(model, ov::test::utils::DEVICE_NPU);
        compiledModel.export_model(ss);
        ss.seekg(0, std::ios::end);
        blobSizeMB = ss.tellg() / 1024.0 / 1024.0;
        // by the time of writing the test, blobSize ~= 23.6 MB
        std::cout << "blobSize = " << blobSizeMB << " MB" << std::endl;
    }
    core.set_property({ov::cache_dir("cache_dir")});

    double compileModelPeakWorkingSetSizeMB = 0.0;
    double compileModelCachedPeakWorkingSetSizeMB = 0.0;
    double importModelIStreamPeakWorkingSetSizeMB = 0.0;
    double importModelTensorPeakWorkingSetSizeMB = 0.0;
    double createInferRequestPeakWorkingSetSizeMB = 0.0;
    double inferPeakWorkingSetSizeMB = 0.0;

    bool captureMemory = true;
    MemoryCaptureEvent memoryEvent = MemoryCaptureEvent::IDLE;
    std::thread memoryCaptureT([&]() {
        double currentMemoryMB = 0.0;
        while (captureMemory) {
            currentMemoryMB =
                ov::test::utils::getVmRSSInKB() / 1024.0;  // get peak memory from windows API might be useful here
            switch (memoryEvent) {
            case MemoryCaptureEvent::COMPILE:
                compileModelPeakWorkingSetSizeMB = currentMemoryMB > compileModelPeakWorkingSetSizeMB
                                                       ? currentMemoryMB
                                                       : compileModelPeakWorkingSetSizeMB;
                break;
            case MemoryCaptureEvent::COMPILE_CACHED:
                compileModelCachedPeakWorkingSetSizeMB = currentMemoryMB > compileModelCachedPeakWorkingSetSizeMB
                                                             ? currentMemoryMB
                                                             : compileModelCachedPeakWorkingSetSizeMB;
                break;
            case MemoryCaptureEvent::IMPORT_MODEL_ISTREAM:
                importModelIStreamPeakWorkingSetSizeMB = currentMemoryMB > importModelIStreamPeakWorkingSetSizeMB
                                                             ? currentMemoryMB
                                                             : importModelIStreamPeakWorkingSetSizeMB;
                break;
            case MemoryCaptureEvent::IMPORT_MODEL_TENSOR:
                importModelTensorPeakWorkingSetSizeMB = currentMemoryMB > importModelTensorPeakWorkingSetSizeMB
                                                            ? currentMemoryMB
                                                            : importModelTensorPeakWorkingSetSizeMB;
                break;
            case MemoryCaptureEvent::CREATE_INFER_REQUEST:
                createInferRequestPeakWorkingSetSizeMB = currentMemoryMB > createInferRequestPeakWorkingSetSizeMB
                                                             ? currentMemoryMB
                                                             : createInferRequestPeakWorkingSetSizeMB;
                break;
            case MemoryCaptureEvent::INFER:
                inferPeakWorkingSetSizeMB =
                    currentMemoryMB > inferPeakWorkingSetSizeMB ? currentMemoryMB : inferPeakWorkingSetSizeMB;
                break;
            default: /* IDLE EVENT */
                break;
            }
            std::this_thread::sleep_for(std::chrono::duration<uint16_t, std::milli>(1000));  // sleep for 1 second
        }
    });

    double currentMemoryMB = 0.0;
    { /* compile_model case */
        double deltaWorkingSetSizeMB = ov::test::utils::getVmRSSInKB() / 1024.0;
        memoryEvent = MemoryCaptureEvent::COMPILE;
        auto compiledModel =
            core.compile_model(model, ov::test::utils::DEVICE_NPU);  // blob size 44,658,688 bytes / 29,200,384 bytes
        memoryEvent = MemoryCaptureEvent::IDLE;
        currentMemoryMB = ov::test::utils::getVmRSSInKB() / 1024.0;
        print_memory_stats_for_event("compile_model",
                                     compileModelPeakWorkingSetSizeMB,
                                     currentMemoryMB,
                                     deltaWorkingSetSizeMB);
        EXPECT_LE(currentMemoryMB - deltaWorkingSetSizeMB, 2 * blobSizeMB)
            << "[ COMPILE_MODEL CASE ] Current measured blob size exceeds twice the initially measured blob size!";
        deltaWorkingSetSizeMB = currentMemoryMB;

        memoryEvent = MemoryCaptureEvent::CREATE_INFER_REQUEST;
        auto inferRequest = compiledModel.create_infer_request();
        memoryEvent = MemoryCaptureEvent::IDLE;
        currentMemoryMB = ov::test::utils::getVmRSSInKB() / 1024.0;
        print_memory_stats_for_event("create_infer_request",
                                     createInferRequestPeakWorkingSetSizeMB,
                                     currentMemoryMB,
                                     deltaWorkingSetSizeMB);
        EXPECT_LE(currentMemoryMB - deltaWorkingSetSizeMB, blobSizeMB)
            << "[ COMPILE_MODEL CASE ] Memory used after creation of infer request exceeds 1x blob size!";
        deltaWorkingSetSizeMB = currentMemoryMB;

        memoryEvent = MemoryCaptureEvent::INFER;
        inferRequest.infer();
        memoryEvent = MemoryCaptureEvent::IDLE;
        currentMemoryMB = ov::test::utils::getVmRSSInKB() / 1024.0;
        print_memory_stats_for_event("infer", inferPeakWorkingSetSizeMB, currentMemoryMB, deltaWorkingSetSizeMB);
        EXPECT_LE(currentMemoryMB - deltaWorkingSetSizeMB, 2 * (paramsTotalSizeMB + resultsTotalSizeMB))
            << "[ COMPILE_MODEL CASE ] Scratch buffer exceeds 2x size of params + results!";

        {
            std::ofstream blobWriter(model->get_friendly_name() + ".blob", std::ios::out | std::ios::binary);
            compiledModel.export_model(blobWriter);
        }
        // reset peaks for compile_model(cached)
        createInferRequestPeakWorkingSetSizeMB = 0.0;
        inferPeakWorkingSetSizeMB = 0.0;
    } /* end of compile_model case */

    { /* compile_model(cached) case */
        double deltaWorkingSetSizeMB = ov::test::utils::getVmRSSInKB() / 1024.0;
        memoryEvent = MemoryCaptureEvent::COMPILE_CACHED;
        auto compiledModel = core.compile_model(model, ov::test::utils::DEVICE_NPU);
        memoryEvent = MemoryCaptureEvent::IDLE;
        currentMemoryMB = ov::test::utils::getVmRSSInKB() / 1024.0;
        print_memory_stats_for_event("compile_model(cached)",
                                     compileModelCachedPeakWorkingSetSizeMB,
                                     currentMemoryMB,
                                     deltaWorkingSetSizeMB);
        if (compileModelCachedPeakWorkingSetSizeMB > 0 && compileModelPeakWorkingSetSizeMB > 0) {
            EXPECT_LE(compileModelCachedPeakWorkingSetSizeMB, compileModelPeakWorkingSetSizeMB)
                << "[ COMPILE_MODEL_CACHED CASE ] Compile model with cache peak is expected to be less than actual "
                   "compilation peak!";
        }
        EXPECT_LE(currentMemoryMB - deltaWorkingSetSizeMB, 2 * blobSizeMB)
            << "[ COMPILE_MODEL_CACHED CASE ] Current measured blob size exceeds twice the initially measured blob "
               "size!";
        deltaWorkingSetSizeMB = currentMemoryMB;

        memoryEvent = MemoryCaptureEvent::CREATE_INFER_REQUEST;
        auto inferRequest = compiledModel.create_infer_request();
        memoryEvent = MemoryCaptureEvent::IDLE;
        currentMemoryMB = ov::test::utils::getVmRSSInKB() / 1024.0;
        print_memory_stats_for_event("create_infer_request",
                                     createInferRequestPeakWorkingSetSizeMB,
                                     currentMemoryMB,
                                     deltaWorkingSetSizeMB);
        EXPECT_LE(currentMemoryMB - deltaWorkingSetSizeMB, blobSizeMB)
            << "[ COMPILE_MODEL_CACHED CASE ] Memory used after creation of infer request exceeds 1x blob size!";
        deltaWorkingSetSizeMB = currentMemoryMB;

        memoryEvent = MemoryCaptureEvent::INFER;
        inferRequest.infer();
        memoryEvent = MemoryCaptureEvent::IDLE;
        currentMemoryMB = ov::test::utils::getVmRSSInKB() / 1024.0;
        print_memory_stats_for_event("infer", inferPeakWorkingSetSizeMB, currentMemoryMB, deltaWorkingSetSizeMB);
        EXPECT_LE(currentMemoryMB - deltaWorkingSetSizeMB, 2 * (paramsTotalSizeMB + resultsTotalSizeMB))
            << "[ COMPILE_MODEL_CACHED CASE ] Scratch buffer exceeds 2x size of params + results!";

        {
            std::ofstream blobWriter(model->get_friendly_name() + ".blob", std::ios::out | std::ios::binary);
            compiledModel.export_model(blobWriter);
        }

        // reset peaks for import_model(istream)
        createInferRequestPeakWorkingSetSizeMB = 0.0;
        inferPeakWorkingSetSizeMB = 0.0;
    } /* end of compile_model(cached) case */

    { /* import_model(istream) case */
        std::ifstream blobReader(model->get_friendly_name() + ".blob", std::ios::in | std::ios::binary);

        double deltaWorkingSetSizeMB = ov::test::utils::getVmRSSInKB() / 1024.0;

        memoryEvent = MemoryCaptureEvent::IMPORT_MODEL_ISTREAM;
        auto compiledModel = core.import_model(blobReader, ov::test::utils::DEVICE_NPU);
        memoryEvent = MemoryCaptureEvent::IDLE;
        currentMemoryMB = ov::test::utils::getVmRSSInKB() / 1024.0;
        print_memory_stats_for_event("import_model(istream)",
                                     importModelIStreamPeakWorkingSetSizeMB,
                                     currentMemoryMB,
                                     deltaWorkingSetSizeMB);
        EXPECT_LE(currentMemoryMB - deltaWorkingSetSizeMB, 2 * blobSizeMB)
            << "[ IMPORT_MODEL_ISTREAM CASE ] Current measured blob size exceeds twice the initially measured blob "
               "size!";
        deltaWorkingSetSizeMB = currentMemoryMB;

        memoryEvent = MemoryCaptureEvent::CREATE_INFER_REQUEST;
        auto inferRequest = compiledModel.create_infer_request();
        memoryEvent = MemoryCaptureEvent::IDLE;
        currentMemoryMB = ov::test::utils::getVmRSSInKB() / 1024.0;
        print_memory_stats_for_event("create_infer_request",
                                     createInferRequestPeakWorkingSetSizeMB,
                                     currentMemoryMB,
                                     deltaWorkingSetSizeMB);
        EXPECT_LE(currentMemoryMB - deltaWorkingSetSizeMB, blobSizeMB)
            << "[ IMPORT_MODEL_ISTREAM CASE ] Memory used after creation of infer request exceeds 1x blob size!";
        deltaWorkingSetSizeMB = currentMemoryMB;

        memoryEvent = MemoryCaptureEvent::INFER;
        inferRequest.infer();
        memoryEvent = MemoryCaptureEvent::IDLE;
        currentMemoryMB = ov::test::utils::getVmRSSInKB() / 1024.0;
        print_memory_stats_for_event("infer", inferPeakWorkingSetSizeMB, currentMemoryMB, deltaWorkingSetSizeMB);
        EXPECT_LE(currentMemoryMB - deltaWorkingSetSizeMB, 2 * (paramsTotalSizeMB + resultsTotalSizeMB))
            << "[ IMPORT_MODEL_ISTREAM CASE ] Scratch buffer exceeds twiche the size of params + results!";

        // reset peaks for import_model(tensor)
        createInferRequestPeakWorkingSetSizeMB = 0.0;
        inferPeakWorkingSetSizeMB = 0.0;
    } /* end of import_model(istream) case */

    { /* import_model(tensor) case */
        auto tensor = ov::read_tensor_data(model->get_friendly_name() + ".blob");

        double deltaWorkingSetSizeMB = ov::test::utils::getVmRSSInKB() / 1024.0;

        memoryEvent = MemoryCaptureEvent::IMPORT_MODEL_TENSOR;
        auto compiledModel = core.import_model(tensor, ov::test::utils::DEVICE_NPU);
        memoryEvent = MemoryCaptureEvent::IDLE;
        currentMemoryMB = ov::test::utils::getVmRSSInKB() / 1024.0;
        print_memory_stats_for_event("import_model(tensor)",
                                     importModelTensorPeakWorkingSetSizeMB,
                                     currentMemoryMB,
                                     deltaWorkingSetSizeMB);
        EXPECT_LE(currentMemoryMB - deltaWorkingSetSizeMB, 2 * blobSizeMB)
            << "[ IMPORT_MODEL_TENSOR CASE ] Current measured blob size exceeds twice the initially measured blob "
               "size!";
        deltaWorkingSetSizeMB = currentMemoryMB;

        memoryEvent = MemoryCaptureEvent::CREATE_INFER_REQUEST;
        auto inferRequest = compiledModel.create_infer_request();
        memoryEvent = MemoryCaptureEvent::IDLE;
        currentMemoryMB = ov::test::utils::getVmRSSInKB() / 1024.0;
        print_memory_stats_for_event("create_infer_request",
                                     createInferRequestPeakWorkingSetSizeMB,
                                     currentMemoryMB,
                                     deltaWorkingSetSizeMB);
        EXPECT_LE(currentMemoryMB - deltaWorkingSetSizeMB, blobSizeMB)
            << "[ IMPORT_MODEL_TENSOR CASE ] Memory used after creation of infer request exceeds 1x blob size!";
        deltaWorkingSetSizeMB = currentMemoryMB;

        memoryEvent = MemoryCaptureEvent::INFER;
        inferRequest.infer();
        memoryEvent = MemoryCaptureEvent::IDLE;
        currentMemoryMB = ov::test::utils::getVmRSSInKB() / 1024.0;
        print_memory_stats_for_event("infer", inferPeakWorkingSetSizeMB, currentMemoryMB, deltaWorkingSetSizeMB);
        EXPECT_LE(currentMemoryMB - deltaWorkingSetSizeMB, 2 * (paramsTotalSizeMB + resultsTotalSizeMB))
            << "[ IMPORT_MODEL_TENSOR CASE ] Scratch buffer exceeds twiche the size of params + results!";
    } /* end of import_model(tensor) case */

    captureMemory = false;
    memoryCaptureT.join();  // ensure thread finished
    std::filesystem::remove(model->get_friendly_name() + ".blob");
    std::filesystem::remove_all("cache_dir");
}

}  // namespace behavior

}  // namespace test

}  // namespace ov
