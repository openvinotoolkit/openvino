#include <openvino/runtime/core.hpp>

//! [ov:caching:part0]
void part0() {
    std::string modelPath = "/tmp/myModel.xml";
    std::string device = "GPU";                             // For example: "CPU", "GPU", "NPU".
    ov::AnyMap config;
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
    std::string device = "GPU";
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
    std::string device = "GPU";
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
    std::string deviceName = "GPU";
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

void part4() {
    std::string modelPath = "/tmp/myModel.xml";
    std::string device = "CPU";
    ov::Core core;                                           // Step 1: create ov::Core object
    core.set_property(ov::cache_dir("/path/to/cache/dir"));  // Step 1b: Enable caching
    auto model = core.read_model(modelPath);                 // Step 2: Read Model
//! [ov:caching:part4]
ov::AnyMap config;
ov::EncryptionCallbacks encryption_callbacks;
static const char codec_key[] = {0x30, 0x60, 0x70, 0x02, 0x04, 0x08, 0x3F, 0x6F, 0x72, 0x74, 0x78, 0x7F};
auto codec_xor = [&](const std::string& source_str) {
    auto key_size = sizeof(codec_key);
    int key_idx = 0;
    std::string dst_str = source_str;
    for (char& c : dst_str) {
        c ^= codec_key[key_idx % key_size];
        key_idx++;
    }
    return dst_str;
};
encryption_callbacks.encrypt = codec_xor;
encryption_callbacks.decrypt = codec_xor;
config.insert(ov::cache_encryption_callbacks(encryption_callbacks));  // Step 4: Set device configuration
auto compiled = core.compile_model(model, device, config);            // Step 5: LoadNetwork
//! [ov:caching:part4]
    if (!compiled) {
        throw std::runtime_error("error");
    }
}

int main() {
    try {
        part0();
        part1();
        part2();
        part3();
        part4();
    } catch (...) {
    }
    return 0;
}