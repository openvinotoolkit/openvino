#include "intel_gpu/runtime/layout.hpp"
#include "intel_gpu/runtime/execution_config.hpp"
#include "vulkan/vk_engine.hpp"
#include "vulkan/vk_kernel.hpp"
#include "vulkan/vk_memory.hpp"
#include "vulkan/vk_stream.hpp"
#include <cstdio>
#include <vector>
int main() {
    cldnn::layout l({1, 1, 1, 1024}, cldnn::data_types::f32, cldnn::format::bfyx);
    std::printf("layout bytes=%zu count=%zu\n", l.bytes_count(), l.count());
    cldnn::vk::vk_engine eng;
    auto mem = eng.allocate_memory(l, cldnn::allocation_type::usm_device, false);
    std::printf("allocated %zu bytes\n", mem->size());
    auto stream = eng.create_stream(cldnn::ExecutionConfig());
    std::vector<uint32_t> spirv;
    (void)spirv;
    std::printf("ENGINE+LAYOUT OK, runtime=%d\n", (int)eng.runtime_type());
    return 0;
}
