#include <openvino/runtime/core.hpp>

void part0() {
    std::string modelPath = "/tmp/myModel.xml";
    std::string device = "GNA";
    ov::AnyMap config;
//! [ov:caching:part0]
ov::Core core;                                              // Step 1: create ov::Core object
core.set_property(ov::cache_dir("/path/to/cache/dir"));     // Step 1b: Enable caching
auto model = core.read_model(modelPath);                    // Step 2: Read Model
//...                                                       // Step 3: Prepare inputs/outputs
//...                                                       // Step 4: Set device configuration
auto compiled = core.compile_model(model, device, config);  // Step 5: LoadNetwork
//! [ov:caching:part0]
    if (!compiled) {
        throw std::runtime_error("error");
    }
}

void part1() {
    std::string modelPath = "/tmp/myModel.xml";
    std::string device = "GNA";
    ov::AnyMap config;
//! [ov:caching:part1]
ov::Core core;                                                  // Step 1: create ov::Core object
auto compiled = core.compile_model(modelPath, device, config);  // Step 2: Compile model by file path
//! [ov:caching:part1]
    if (!compiled) {
        throw std::runtime_error("error");
    }
}

void part2() {
    std::string modelPath = "/tmp/myModel.xml";
    std::string device = "GNA";
    ov::AnyMap config;
//! [ov:caching:part2]
ov::Core core;                                                  // Step 1: create ov::Core object
core.set_property(ov::cache_dir("/path/to/cache/dir"));         // Step 1b: Enable caching
auto compiled = core.compile_model(modelPath, device, config);  // Step 2: Compile model by file path
//! [ov:caching:part2]
    if (!compiled) {
        throw std::runtime_error("error");
    }
}

void part3() {
    std::string deviceName = "GNA";
    ov::AnyMap config;
    ov::Core core;
//! [ov:caching:part3]
// Get list of supported device capabilities
std::vector<std::string> caps = core.get_property(deviceName, ov::device::capabilities);

// Find 'EXPORT_IMPORT' capability in supported capabilities
bool cachingSupported = std::find(caps.begin(), caps.end(), ov::device::capability::EXPORT_IMPORT) != caps.end();
//! [ov:caching:part3]
    if (!cachingSupported) {
        throw std::runtime_error("GNA should support model caching");
    }
}

int main() {
    try {
        part0();
        part1();
        part2();
        part3();
    } catch (...) {
    }
    return 0;
}