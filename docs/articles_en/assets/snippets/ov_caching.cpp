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
    std::string device = "GPU";
    ov::Core core;                                           // Step 1: create ov::Core object
    bool hasGPU = false;                                     // Step 1a: Check if GPU is available
    auto devices = core.get_available_devices();
    for (auto&& supported : devices) {
        hasGPU |= supported.find(device) != std::string::npos;
    }
    if(!hasGPU) {
        return;
    }
    core.set_property(ov::cache_dir("/path/to/cache/dir"));  // Step 1b: Enable caching
//! [ov:caching:part4]
// Note: model path needs to point to the *.xml file, not *.bin when using the IR model format.
auto compiled = core.compile_model(modelPath,
                                   device,
                                   ov::cache_mode(ov::CacheMode::OPTIMIZE_SIZE));
//! [ov:caching:part4]
    if (!compiled) {
        throw std::runtime_error("error");
    }
}

void part5() {
    std::string modelPath = "/tmp/myModel.xml";
    std::string device = "CPU";
    ov::Core core;                                           // Step 1: create ov::Core object
    core.set_property(ov::cache_dir("/path/to/cache/dir"));  // Step 1b: Enable caching
    auto model = core.read_model(modelPath);                 // Step 2: Read Model
//! [ov:caching:part5]
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
//! [ov:caching:part5]
    if (!compiled) {
        throw std::runtime_error("error");
    }
}

void part6() {
    std::string modelPath = "/tmp/myModel.xml";
    std::string device = "GPU";
    ov::Core core;                                           // Step 1: create ov::Core object
    bool hasGPU = false;                                     // Step 1a: Check if GPU is available
    auto devices = core.get_available_devices();
    for (auto&& supported : devices) {
        hasGPU |= supported.find(device) != std::string::npos;
    }
    if(!hasGPU) {
        return;
    }
    core.set_property(ov::cache_dir("/path/to/cache/dir"));  // Step 1b: Enable caching
//! [ov:caching:part6]
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
auto compiled = core.compile_model(modelPath,
                                   device,
                                   ov::cache_encryption_callbacks(ov::EncryptionCallbacks{codec_xor, codec_xor}),
                                   ov::cache_mode(ov::CacheMode::OPTIMIZE_SIZE));  // Step 5: Compile model
//! [ov:caching:part6]
    if (!compiled) {
        throw std::runtime_error("error");
    }
}

void part7() {
    std::string modelPath = "/tmp/myModel.xml";
    std::string device = "CPU";
//! [ov:caching:part7]
ov::Core core;
auto model = core.read_model(modelPath);
auto compiled = core.compile_model(model, device);

// Export the compiled model to a stream
std::stringstream model_stream;
compiled.export_model(model_stream);

// Get the exported data as a string, then wrap it in a Tensor
std::string model_data = model_stream.str();
ov::Tensor compiled_blob(ov::element::u8, {model_data.size()}, model_data.data());

// Import the compiled model back from the Tensor
auto imported = core.import_model(compiled_blob, device);
//! [ov:caching:part7]
    if (!imported) {
        throw std::runtime_error("error");
    }
}

void part8() {
    std::string modelPath = "/tmp/myModel.xml";
    std::string device = "CPU";
//! [ov:caching:part8]
ov::Core core;

// ov::enable_weightless takes priority over CacheMode when both are set.
auto compiled = core.compile_model(modelPath, device, ov::enable_weightless(true));
//! [ov:caching:part8]
    if (!compiled) {
        throw std::runtime_error("error");
    }
}

void part9() {
    std::string modelPath = "/tmp/myModel.xml";
    std::string device = "CPU";
//! [ov:caching:part9]
ov::Core core;
auto model = core.read_model(modelPath);
auto compiled = core.compile_model(model, device, ov::enable_weightless(true));

// Export a weightless blob: it does not contain the model weights.
std::stringstream model_stream;
compiled.export_model(model_stream);
std::string model_data = model_stream.str();
ov::Tensor weightless_blob(ov::element::u8, {model_data.size()}, model_data.data());

// The weights must be supplied separately when importing a weightless blob:
// Option 1: a path to the original weights file.
auto imported_from_path = core.import_model(weightless_blob, device, ov::weights_path("/tmp/myModel.bin"));

// Option 2: the original ov::Model object.
auto imported_from_model = core.import_model(weightless_blob, device, ov::hint::model(model));
//! [ov:caching:part9]
    if (!imported_from_path || !imported_from_model) {
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
        part5();
        part6();
        part7();
        part8();
        part9();
    } catch (...) {
    }
    return 0;
}
