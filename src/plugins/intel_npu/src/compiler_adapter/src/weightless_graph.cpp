// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "weightless_graph.hpp"

#include <algorithm>
#include <cinttypes>
#include <condition_variable>
#include <cstring>
#include <iterator>
#include <mutex>
#include <optional>
#include <queue>
#include <unordered_map>
#include <unordered_set>

#include "intel_npu/config/options.hpp"
#include "intel_npu/utils/utils.hpp"
#include "intel_npu/utils/zero/zero_cmd_queue_pool.hpp"
#include "intel_npu/utils/zero/zero_utils.hpp"
#include "openvino/core/memory_util.hpp"
#include "openvino/core/rt_info/weightless_caching_attributes.hpp"
#include "openvino/core/weight_sharing_util.hpp"
#include "openvino/runtime/icore.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/shared_buffer.hpp"
#include "openvino/util/common_util.hpp"
#include "openvino/util/mmap_object.hpp"

#define USE_SINGLE_THREADED_RUN_INIT 1

namespace intel_npu {

namespace {

/**
 * @brief Process-wide record of the weights produced by the init schedules of weightless graphs, used when
 * NPU_SHARE_WEIGHTLESS_INIT_OUTPUTS is enabled.
 * @details Applications often load several compiled models over the same weights (an LLM compiled for several token
 * block sizes or context lengths, prefill and decode graphs). Each weightless graph runs its init schedules into a
 * device buffer of its own, so every such model holds another full copy of the weights. After an init schedule ran,
 * every output is looked up here (size + a sampled hash, then compared byte by byte) among the outputs of graphs that
 * are still alive:
 * - an output found whole, or as an aligned part of a larger earlier output, is bound to that memory;
 * - when every output was found, the new buffer is released;
 * - when only some were, the others are copied into a compact buffer of their own and the new buffer is released too.
 * The buffers bound this way are kept alive by the graph that uses them (shared ownership of the ZeroTensor).
 */
class InitOutputsRegistry {
public:
    static InitOutputsRegistry& get() {
        static InitOutputsRegistry registry;
        return registry;
    }

    /**
     * @brief Looks up the outputs of an init schedule that ran into "allocatedTensor".
     * @param tensorsMap The views of "allocatedTensor" by output name; entries found elsewhere are replaced by views
     * of the existing memory.
     * @returns The buffers the outputs now point to, when "allocatedTensor" is no longer needed; std::nullopt when
     * nothing was found (the outputs are then recorded for later graphs).
     */
    std::optional<std::vector<std::shared_ptr<ZeroTensor>>> share(
        const std::shared_ptr<ZeroInitStructsHolder>& initStructs,
        const std::shared_ptr<ZeroTensor>& allocatedTensor,
        std::unordered_map<std::string, std::shared_ptr<ov::ITensor>>& tensorsMap,
        const Logger& logger) {
        std::lock_guard<std::mutex> lock(_mutex);

        std::unordered_map<std::string, std::pair<std::shared_ptr<ZeroTensor>, const unsigned char*>> found;
        size_t foundBytes = 0;
        for (const auto& [name, tensor] : tensorsMap) {
            const auto* data = static_cast<const unsigned char*>(tensor->data());
            const size_t size = tensor->get_byte_size();
            if (auto match = find(data, size, allocatedTensor)) {
                found.emplace(name, *match);
                foundBytes += size;
            }
        }

        if (found.empty()) {
            record(allocatedTensor, tensorsMap, nullptr);
            logger.debug("Init outputs not found elsewhere: %zu outputs, %zu bytes",
                         tensorsMap.size(),
                         allocatedTensor->get_byte_size());
            return std::nullopt;
        }

        std::vector<std::shared_ptr<ZeroTensor>> owners;
        for (const auto& [name, match] : found) {
            const auto& view = tensorsMap.at(name);
            tensorsMap[name] =
                ov::make_tensor(view->get_element_type(), view->get_shape(), const_cast<unsigned char*>(match.second));
            owners.push_back(match.first);
        }

        if (found.size() != tensorsMap.size()) {
            // The outputs not found elsewhere are copied into a compact buffer, each at an aligned offset
            const auto align = [](size_t value) {
                return (value + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
            };
            size_t compactSize = 0;
            for (const auto& [name, tensor] : tensorsMap) {
                if (found.count(name) == 0) {
                    compactSize += align(tensor->get_byte_size());
                }
            }
            const auto compact =
                std::make_shared<ZeroTensor>(initStructs, ov::element::Type_t::u8, ov::Shape({compactSize}), false);
            auto* base = static_cast<unsigned char*>(const_cast<void*>(compact->data(ov::element::Type_t::u8)));
            size_t offset = 0;
            std::unordered_set<std::string> moved;
            for (auto& [name, tensor] : tensorsMap) {
                if (found.count(name) != 0) {
                    continue;
                }
                const size_t size = tensor->get_byte_size();
                std::memcpy(base + offset, tensor->data(), size);
                tensor = ov::make_tensor(tensor->get_element_type(), tensor->get_shape(), base + offset);
                moved.insert(name);
                offset += align(size);
            }
            record(compact, tensorsMap, &moved);
            owners.push_back(compact);
            logger.info("Init outputs partly shared: %zu of %zu outputs (%zu bytes) found, %zu bytes kept of %zu",
                        found.size(),
                        tensorsMap.size(),
                        foundBytes,
                        compactSize,
                        allocatedTensor->get_byte_size());
            return owners;
        }

        logger.info("Init outputs shared: all %zu outputs found, %zu bytes released",
                    found.size(),
                    allocatedTensor->get_byte_size());
        return owners;
    }

private:
    struct Entry {
        std::weak_ptr<ZeroTensor> owner;
        const unsigned char* data;
        size_t size;
    };

    static constexpr size_t ALIGNMENT = 64;
    // Outputs from this size on are also looked up as a part of a larger earlier output: e.g. a group-quantized weight
    // kept per group by one graph and as chunks of several groups by another one
    static constexpr size_t MIN_PART_SIZE = size_t(1) << 15;
    static constexpr size_t MIN_CONTAINER_SIZE = size_t(1) << 16;

    static uint64_t sampled_hash(const unsigned char* data, size_t size) {
        uint64_t hash = 1469598103934665603ULL ^ size;  // FNV-1a over 64 bytes of every 4 KB, and the last 64 bytes
        const auto mix = [&](const unsigned char* begin, size_t length) {
            for (size_t i = 0; i < length; ++i) {
                hash = (hash ^ begin[i]) * 1099511628211ULL;
            }
        };
        for (size_t offset = 0; offset < size; offset += 4096) {
            mix(data + offset, std::min<size_t>(64, size - offset));
        }
        if (size > 64) {
            mix(data + size - 64, 64);
        }
        return hash;
    }

    std::optional<std::pair<std::shared_ptr<ZeroTensor>, const unsigned char*>>
    find(const unsigned char* data, size_t size, const std::shared_ptr<ZeroTensor>& self) const {
        const auto range = _byHash.equal_range(sampled_hash(data, size));
        for (auto it = range.first; it != range.second; ++it) {
            auto owner = it->second.owner.lock();
            if (owner == nullptr || owner == self || it->second.size != size) {
                continue;
            }
            if (std::memcmp(it->second.data, data, size) == 0) {
                return std::make_pair(owner, it->second.data);
            }
        }
        if (size < MIN_PART_SIZE) {
            return std::nullopt;
        }
        for (const auto& container : _containers) {
            if (container.size <= size || container.size % size != 0) {
                continue;
            }
            auto owner = container.owner.lock();
            if (owner == nullptr || owner == self) {
                continue;
            }
            for (size_t offset = 0; offset + size <= container.size; offset += size) {
                const unsigned char* part = container.data + offset;
                if (std::memcmp(part, data, 64) == 0 && std::memcmp(part, data, size) == 0) {
                    return std::make_pair(owner, part);
                }
            }
        }
        return std::nullopt;
    }

    void record(const std::shared_ptr<ZeroTensor>& owner,
                const std::unordered_map<std::string, std::shared_ptr<ov::ITensor>>& tensorsMap,
                const std::unordered_set<std::string>* only) {
        // Entries of buffers released in the meantime go first
        for (auto it = _byHash.begin(); it != _byHash.end();) {
            it = it->second.owner.expired() ? _byHash.erase(it) : std::next(it);
        }
        _containers.erase(std::remove_if(_containers.begin(),
                                         _containers.end(),
                                         [](const Entry& entry) {
                                             return entry.owner.expired();
                                         }),
                          _containers.end());

        for (const auto& [name, tensor] : tensorsMap) {
            if (only != nullptr && only->count(name) == 0) {
                continue;
            }
            const auto* data = static_cast<const unsigned char*>(tensor->data());
            const size_t size = tensor->get_byte_size();
            _byHash.emplace(sampled_hash(data, size), Entry{owner, data, size});
            if (size >= MIN_CONTAINER_SIZE) {
                _containers.push_back(Entry{owner, data, size});
            }
        }
    }

    std::mutex _mutex;
    std::unordered_multimap<uint64_t, Entry> _byHash;
    std::vector<Entry> _containers;
};

constexpr uint8_t MAIN_SCHEDULE_INDEX = 0;
constexpr std::string_view WEIGHTS_IR_EXTENSION = ".bin";
constexpr std::string_view WEIGHTS_ONNX_EXTENSION = ".data";
constexpr std::string_view WEIGHTS_ONNX_PROXY_EXTENSION = ".data_proxy";
constexpr std::string_view ONNX_EXTENSION = ".onnx";

constexpr std::string_view CONSTANT_OVERFLOW_MESSAGE = "Overflow while computing byte size for constant: ";

std::unordered_map<size_t, std::shared_ptr<ov::op::v0::Constant>> get_all_constants_in_topological_order(
    const std::shared_ptr<const ov::Model>& model) {
    std::unordered_map<size_t, std::shared_ptr<ov::op::v0::Constant>> constants;

    OPENVINO_ASSERT(model != nullptr, "Model is required to extract constants in topological order.");
    bool at_least_one_wca_found = false;

    // Match the inputs of the "init" model with the Constant nodes of the original model
    for (auto&& node : model->get_ops()) {
        if (!ov::is_type<ov::op::v0::Constant>(node)) {
            continue;
        }

        auto constantNode = std::static_pointer_cast<ov::op::v0::Constant>(node);
        ov::RTMap& runtimeInfoMap = constantNode->get_rt_info();
        const auto& weightlessCacheAttrIt = runtimeInfoMap.find(ov::WeightlessCacheAttribute::get_type_info_static());
        if (weightlessCacheAttrIt != runtimeInfoMap.end()) {
            at_least_one_wca_found = true;
            auto& weightlessCacheAttr = weightlessCacheAttrIt->second.as<ov::WeightlessCacheAttribute>();

            auto& constant = constants[weightlessCacheAttr.bin_offset];
            if (constant != nullptr) {
                // if multiple constants point to the same buffer, ensure that
                // their binary sizes are the same
                OPENVINO_ASSERT(constant->get_byte_size() == constantNode->get_byte_size(),
                                "Found ov::Constant that points to the common buffer but has mismatching byte size. "
                                "This may indicate a bug in OV model compression.");
                continue;
            }
            constant = std::move(constantNode);
        }
    }

    OPENVINO_ASSERT(at_least_one_wca_found,
                    "No \"WeightlessCacheAttribute\" has been found in any of the model's Constant nodes. This "
                    "attribute is required for running the \"weights separation\" flow.");

    return constants;
}

std::unordered_map<size_t, std::shared_ptr<ov::op::v0::Constant>> get_all_constants_memory_mapped(
    std::string_view weightsPath,
    const std::vector<NetworkMetadata>& initNetworkMetadata) {
    std::unordered_map<size_t, std::shared_ptr<ov::op::v0::Constant>> constants;

    auto mapped_memory = ov::load_mmap_object(weightsPath);
    for (const auto& initMetadata : initNetworkMetadata) {
        for (const IODescriptor& descriptor : initMetadata.inputs) {
            const auto& opt = ov::util::view_to_number<size_t>(descriptor.nameFromCompiler);
            OPENVINO_ASSERT(opt.has_value(), "Failed to parse id for constant: ", descriptor.nameFromCompiler);

            const size_t id = opt.value();
            const auto byte_size_opt =
                ov::util::get_memory_size_safe(descriptor.precision, descriptor.shapeFromCompiler.to_shape());
            OPENVINO_ASSERT(byte_size_opt.has_value(), CONSTANT_OVERFLOW_MESSAGE, descriptor.nameFromCompiler);
            const size_t byte_size = byte_size_opt.value();
            OPENVINO_ASSERT(id <= mapped_memory->size() && byte_size <= mapped_memory->size() - id,
                            "Constant offset/size is out of bounds for mapped weights file: offset=",
                            id,
                            ", size=",
                            byte_size,
                            ", file_size=",
                            mapped_memory->size(),
                            ", name=",
                            descriptor.nameFromCompiler);
            auto weight_buffer =
                std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::MappedMemory>>>(mapped_memory->data() + id,
                                                                                      byte_size,
                                                                                      mapped_memory);

            auto [it, inserted] =
                constants.emplace(id,
                                  std::make_shared<ov::op::v0::Constant>(descriptor.precision,
                                                                         descriptor.shapeFromCompiler.to_shape(),
                                                                         weight_buffer));
            if (!inserted) {
                OPENVINO_ASSERT(it->second->get_byte_size() == byte_size,
                                "Duplicate constant offset found with mismatching byte size: offset=",
                                id);
            }
        }
    }

    return constants;
}

std::unordered_map<size_t, std::shared_ptr<ov::op::v0::Constant>> extract_constants_map(
    std::variant<std::monostate, std::shared_ptr<const ov::Model>, std::pair<std::string, std::shared_ptr<ov::ICore>>>&&
        weightsSource,
    const std::vector<NetworkMetadata>& initNetworkMetadata) {
    if (const std::shared_ptr<const ov::Model>* model = std::get_if<std::shared_ptr<const ov::Model>>(&weightsSource)) {
        return get_all_constants_in_topological_order(*model);
    } else if (const std::pair<std::string, std::shared_ptr<ov::ICore>>* weightsPathAndCore =
                   std::get_if<std::pair<std::string, std::shared_ptr<ov::ICore>>>(&weightsSource)) {
        auto [weightsPath, core] = *weightsPathAndCore;

        auto ext = ov::util::path_to_string(ov::util::make_path(weightsPath).extension());
        if (ext == ONNX_EXTENSION) {
            const auto model = core->read_model(weightsPath, weightsPath, {});
            return get_all_constants_in_topological_order(model);
        } else if (ext == WEIGHTS_IR_EXTENSION || ext == WEIGHTS_ONNX_EXTENSION ||
                   ext == WEIGHTS_ONNX_PROXY_EXTENSION) {
            return get_all_constants_memory_mapped(weightsPath, initNetworkMetadata);
        } else {
            OPENVINO_THROW("Invalid path to the weights: ",
                           weightsPath,
                           ". A \".bin\", \".data\", \".data_proxy\" or \".onnx\" extension was expected.");
        }
    }

    OPENVINO_THROW("No source of weights has been provided to the \"WeightlessGraph\" ctor");
}

struct QueueData {
    int64_t initIndex = -1;
    WeightlessGraph::InputData inputs;
    WeightlessGraph::OutputData outputs;

    bool isTerminator() const {
        return initIndex == -1;
    }
};

// very simple multi-threaded executor for 2 kinds of tasks where task 1 "calls"
// task 2
template <typename Task1Callable, typename Task2Callable>
class Parallelizer {
    std::unordered_map<size_t, std::shared_ptr<ov::op::v0::Constant>> _modelConstants;

    std::mutex _mutex;
    std::queue<QueueData> _payloads;
    std::atomic_bool _progressTask1 = true;

    Task1Callable _task1;
    Task2Callable _task2;

public:
    Parallelizer(std::unordered_map<size_t, std::shared_ptr<ov::op::v0::Constant>>&& _constants,
                 Task1Callable&& task1,
                 Task2Callable&& task2,
                 const Logger& logger)
        : _modelConstants(std::move(_constants)),
          _task1(std::forward<Task1Callable>(task1)),
          _task2(std::forward<Task2Callable>(task2)) {}

    void callForAllAndWait(const int64_t numberOfInits) {
        std::condition_variable task1SyncPoint;
        std::condition_variable task2SyncPoint;

        std::thread task2Thread([&]() {
            while (true) {
                std::unique_lock lock(_mutex);
                task2SyncPoint.wait(lock, [&]() {
                    return !_payloads.empty();
                });

                auto payload = std::move(_payloads.front());
                _payloads.pop();
                if (payload.isTerminator()) {
                    return;  // Note: exit condition
                }
                lock.unlock();

                _task2(std::move(payload), task1SyncPoint,
                       _progressTask1);  // TODO: putting sync point inside is meh
            }
        });

        std::mutex task1Mutex;
        for (int64_t i = 0; i < numberOfInits; ++i) {
            {
                // TODO: task1Mutex is *only* used here. this is a poor man's
                // spinlock on an atomic boolean (without busy wait)
                std::unique_lock lock(task1Mutex);
                task1SyncPoint.wait(lock, [&]() {
                    return _progressTask1.load();
                });
            }

            // TODO: should we run task1 under mutex?
            auto payload = _task1(_modelConstants, i);
            _progressTask1.store(false);
            {
                std::lock_guard guard(_mutex);
                _payloads.push(std::move(payload));
            }
            task2SyncPoint.notify_one();
        }
        {
            std::lock_guard guard(_mutex);
            _payloads.push({});  // isTerminator()
        }
        task2SyncPoint.notify_one();

        task2Thread.join();
        _modelConstants = {};
    }
};

// c++17 deduction guide
template <typename Task1Callable, typename Task2Callable>
Parallelizer(std::unordered_map<size_t, std::shared_ptr<ov::op::v0::Constant>>&&,
             Task1Callable&&,
             Task2Callable&&,
             const Logger&) -> Parallelizer<Task1Callable, Task2Callable>;

void merge_two_maps(std::unordered_map<std::string, std::shared_ptr<ov::ITensor>>& dst,
                    std::unordered_map<std::string, std::shared_ptr<ov::ITensor>>& src) {
    dst.merge(src);
    OPENVINO_ASSERT(src.empty(), "Found weights inputs collision between different inits");
}

}  // namespace

WeightlessGraph::WeightlessGraph(
    const std::shared_ptr<ZeGraphExtWrappers>& zeGraphExt,
    const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStruct,
    const GraphDescriptor& mainGraphDesc,
    NetworkMetadata mainMetadata,
    std::optional<ov::Tensor> mainBlob,
    const std::vector<GraphDescriptor>& initGraphDesc,
    std::vector<NetworkMetadata> initMetadata,
    std::optional<std::vector<ov::Tensor>> initBlobs,
    std::variant<std::monostate, std::shared_ptr<const ov::Model>, std::pair<std::string, std::shared_ptr<ov::ICore>>>&&
        weightsSource,
    const bool blobIsPersistent,
    const std::optional<std::string>& compatibilityDescriptor,
    const std::shared_ptr<IProfilingDecoder>& profilingDecoder)
    : Graph(zeGraphExt,
            zeroInitStruct,
            mainGraphDesc,
            std::move(mainMetadata),
            std::move(mainBlob),
            compatibilityDescriptor,
            blobIsPersistent,
            profilingDecoder),
      _initsGraphDesc(initGraphDesc),
      _initBlobs(std::move(initBlobs)),
      _initsMetadata(std::move(initMetadata)),
      _constants(extract_constants_map(std::move(weightsSource), _initsMetadata)),
      _wgLogger("WeightlessGraph", Logger::global().level()) {}

std::pair<uint64_t, std::optional<std::vector<uint64_t>>> WeightlessGraph::export_blob(std::ostream& stream) const {
    if (_blobIsReleased) {
        OPENVINO_THROW("Model was optimized away. Try importing it using `ov::hint::compiled_blob` property to extend "
                       "its lifetime.");
    }

    size_t blobIndex = 0;
    std::uint32_t totalResult = 1171117u;
    totalResult = ((totalResult << 7) + totalResult);

    const auto writeToStream = [&](GraphDescriptor _graphDesc,
                                   const std::optional<ov::Tensor>& blobTensor) -> uint64_t {
        uint64_t blobSize;
        const uint8_t* blobRawPtr = nullptr;
        std::vector<uint8_t> blob;

        if (blobTensor == std::nullopt) {
            OPENVINO_ASSERT(_zeGraphExt != nullptr, "Zero compiler adapter wasn't initialized");
            // when compiling the model using Compiler in Driver, the blob is handled by the driver
            _zeGraphExt->getGraphBinary(_graphDesc, blob, blobRawPtr, blobSize);
        } else {
            // in all other cases, the blob is handled by the plugin
            blobRawPtr = static_cast<const uint8_t*>(blobTensor->data());
            blobSize = blobTensor->get_byte_size();
        }

        if (blobSize > static_cast<decltype(blobSize)>(std::numeric_limits<std::streamsize>::max())) {
            OPENVINO_THROW("Blob size is too large to be represented on a std::streamsize!");
        }
        stream.write(reinterpret_cast<const char*>(blobRawPtr), static_cast<std::streamsize>(blobSize));

        if (!stream) {
            _wgLogger.error("Write blob to stream failed. Blob is broken!");
            return 0;
        }

        if (_wgLogger.level() >= ov::log::Level::INFO) {
            std::uint32_t result = 1171117u;
            for (const uint8_t* it = blobRawPtr; it != blobRawPtr + blobSize; ++it) {
                result = ((result << 7) + result) + static_cast<uint32_t>(*it);
            }

            totalResult += result;

            if (blobIndex == MAIN_SCHEDULE_INDEX) {
                _wgLogger.info("Main blob size: %" PRIu64 ", hash: %x", blobSize, result);
            } else {
                _wgLogger.info("Init part %zu blob size %" PRIu64 ", hash: %x", blobIndex, blobSize, result);
            }
        }

        size_t size = utils::align_size_to_standard_page_size(blobSize);
        size_t paddingSize = size - blobSize;
        if (paddingSize > 0) {
            std::fill_n(std::ostream_iterator<char>(stream), paddingSize, 0);

            if (!stream) {
                _wgLogger.error("Write padding to stream failed. Blob is broken!");
                return 0;
            }

            _wgLogger.info("Blob size with padding: %zu", size);
        }

        return size;
    };

    // By convention, first write the main part
    uint64_t mainBlobSize = writeToStream(_graphDesc, _blob);
    uint64_t totalBlobSize = mainBlobSize;
    ++blobIndex;

    // Then the init schedules
    std::vector<uint64_t> initSizes;
    for (size_t initIndex = 0; initIndex < _initsGraphDesc.size(); ++initIndex) {
        uint64_t initBlobSize = writeToStream(_initsGraphDesc.at(initIndex)._handle,
                                              _initBlobs.has_value() && _initBlobs->at(initIndex)
                                                  ? std::make_optional(_initBlobs->at(initIndex))
                                                  : std::nullopt);
        totalBlobSize += initBlobSize;
        initSizes.push_back(initBlobSize);
        ++blobIndex;
    }

    _wgLogger.info("Blob size: %" PRIu64 ", hash: %x", totalBlobSize, totalResult);

    _wgLogger.info("Write blob to stream successfully.");
    return std::make_pair(totalBlobSize, initSizes);
}

void WeightlessGraph::initialize_impl(const Config& config) {
    if (_zeGraphExt == nullptr || _graphDesc._handle == nullptr || _zeroInitStruct == nullptr) {
        // To ensure that no issues are thrown during subsequent calls.
        return;
    }

    // Simplified version for init schedules
    const size_t numberOfInits = _initsGraphDesc.size();
    _initsCommandLists.resize(numberOfInits);
    _initsFences.resize(numberOfInits);

    for (size_t initIndex = 0; initIndex < numberOfInits; ++initIndex) {
        _wgLogger.debug("WeightlessGraph initialize start, init schedule ", initIndex);
        _zeGraphExt->initializeGraph(_initsGraphDesc.at(initIndex));
        _wgLogger.debug("WeightlessGraph initialize finish, init schedule ", initIndex);

        //  We are allowed to release the original blob because weights were loaded in NPU memory during
        //  _zeGraphExt->initializeGraph(). The driver will not access the original blob from this moment on, so we are
        //  releasing it here to avoid unnecessary memory usage.
        release_init_blob(initIndex);
    }

    // Create a single command queue for all weights initialization schedules
    uint32_t commandQueueOptions = 0;
    if (config.has<TURBO>() && config.get<TURBO>()) {
        if (_zeroInitStruct->getCommandQueueDdiTable().version() >= ZE_MAKE_VERSION(1, 0)) {
            _wgLogger.debug("Set ZE_NPU_COMMAND_QUEUE_OPTION_TURBO in init command queue options");
            commandQueueOptions = commandQueueOptions | ZE_NPU_COMMAND_QUEUE_OPTION_TURBO;
        }
    }
    CommandQueueDesc commandQueueDesc{
        zeroUtils::toZeQueuePriority(config.get<MODEL_PRIORITY>()),
        config.has<WORKLOAD_TYPE>() ? zeroUtils::toZeQueueWorkloadType(config.get<WORKLOAD_TYPE>()) : std::nullopt,
        commandQueueOptions,
        this,
        config.get<SHARED_COMMON_QUEUE>()};
    _initsCommandQueue = ZeroCmdQueuePool::getInstance().getCommandQueue(_zeroInitStruct, commandQueueDesc);

    _shareInitOutputs = config.get<SHARE_WEIGHTLESS_INIT_OUTPUTS>();

#if USE_SINGLE_THREADED_RUN_INIT
    run_init_single_threaded();
#else
    run_init_multi_threaded();
#endif

    if (_initBlobs != std::nullopt) {  // Do not release the graph when compiling a model on the CiD path, and we don't
                                       // have a blob. We may need it to export later.
        release_graphs();
    }

    _initsCommandLists.clear();
    _initsFences.clear();
    _initsMetadata.clear();
    _initsCommandQueue.reset();
    _sharedInitOutputsToRelease.clear();

    // The main schedule is initialized after the weights initialization ones in order to save some memory
    Graph::initialize_impl(config);

    set_weights_inputs();
}

WeightlessGraph::InputData WeightlessGraph::allocate_inputs(
    const size_t initIndex,
    std::unordered_map<size_t, std::shared_ptr<ov::op::v0::Constant>>& constants) {
    std::vector<std::shared_ptr<ov::ITensor>> initInputsViewTensors;
    size_t initInputsByteSize = 0;

    for (const IODescriptor& descriptor : _initsMetadata.at(initIndex).inputs) {
        const std::optional<size_t> size =
            ov::util::get_memory_size_safe(descriptor.precision, shape_size(descriptor.shapeFromCompiler.to_shape()));
        OPENVINO_ASSERT(size.has_value(), CONSTANT_OVERFLOW_MESSAGE, descriptor.nameFromCompiler);
        OPENVINO_ASSERT(initInputsByteSize + size.value() >= initInputsByteSize,
                        "Buffer size overflow while summing all input sizes for the init schedule index ",
                        initIndex);
        initInputsByteSize += size.value();
    }

    // Due to the large number of init inputs, allocating a single buffer for all of them is more efficient. "View
    // tensors" are used for separating them.
    const std::shared_ptr<ZeroTensor> initInputsAllocatedTensor =
        std::make_shared<ZeroTensor>(_zeroInitStruct, ov::element::Type_t::u8, ov::Shape({initInputsByteSize}), true);

    std::vector<size_t> noLongerRequiredIds;
    noLongerRequiredIds.reserve(_initsMetadata.at(initIndex).inputs.size());

    size_t offset = 0;
    for (const IODescriptor& descriptor : _initsMetadata.at(initIndex).inputs) {
        auto currentInputBufferLocation =
            static_cast<unsigned char*>(const_cast<void*>(initInputsAllocatedTensor->data(ov::element::Type_t::u8))) +
            offset;
        const auto tensorShapeFromCompiler = descriptor.shapeFromCompiler.to_shape();
        const size_t currentInputSize =
            ov::util::get_memory_size(descriptor.precision, shape_size(tensorShapeFromCompiler));

        const auto& opt = ov::util::view_to_number<size_t>(descriptor.nameFromCompiler);
        OPENVINO_ASSERT(opt.has_value(), "Failed to parse id for constant: ", descriptor.nameFromCompiler);

        const size_t id = opt.value();
        auto constantIt = constants.find(id);
        OPENVINO_ASSERT(constantIt != constants.end(),
                        "Weights ID ",
                        id,
                        " not found in the model constants. This may indicate a mismatch between the model and the "
                        "metadata of the compiled model.");

        const auto constant = constantIt->second;
        OPENVINO_ASSERT(constant->get_byte_size() == currentInputSize,
                        "Binary size mismatch found for weights ID ",
                        id,
                        " between the model and compiled metadata.");
        std::memcpy(currentInputBufferLocation, constant->get_data_ptr(), currentInputSize);

        // Note: Use compiler-provided precision and shape, because duplicates -
        // constants that point to the same binary data - can in theory have
        // different shape or even type (OV model compression only guarantees
        // that the data is the same). In order to avoid any potential issues
        // due to shape/type mismatches, init tensors should align with
        // compiler's expectations.
        initInputsViewTensors.push_back(
            ov::make_tensor(descriptor.precision, tensorShapeFromCompiler, currentInputBufferLocation));
        offset += currentInputSize;

        // Note: One cannot immediately delete the constant, because there might
        // be a duplicate. The deletion should happen once the input data for
        // init schedule is fully set.
        noLongerRequiredIds.push_back(id);
    }

    for (size_t id : noLongerRequiredIds) {
        // Note: By construction of the weight schedule, every constant from OV
        // model appears in exactly one schedule. Thus, one can delete
        // the handle to the constant memory early.
        if (auto it = constants.find(id); it != constants.end()) {
            ov::wsh::Extension::hint_evict(*it->second);
            constants.erase(it);
        }
    }

    return {std::move(initInputsViewTensors), initInputsAllocatedTensor};
}

WeightlessGraph::OutputData WeightlessGraph::allocate_outputs(const size_t initIndex) {
    std::vector<std::shared_ptr<ov::ITensor>> initOutputsViewTensorsVector;
    std::unordered_map<std::string, std::shared_ptr<ov::ITensor>> initOutputsViewTensorsMap;
    size_t initOutputsByteSize = 0;

    for (const IODescriptor& descriptor : _initsMetadata.at(initIndex).outputs) {
        const std::optional<size_t> size =
            ov::util::get_memory_size_safe(descriptor.precision, shape_size(descriptor.shapeFromCompiler.to_shape()));
        OPENVINO_ASSERT(size.has_value(), CONSTANT_OVERFLOW_MESSAGE, descriptor.nameFromCompiler);
        OPENVINO_ASSERT(initOutputsByteSize + size.value() >= initOutputsByteSize,
                        "Buffer size overflow while summing all output sizes for the init schedule index ",
                        initIndex);
        initOutputsByteSize += size.value();
    }

    const std::shared_ptr<ZeroTensor> initOutputsAllocatedTensor =
        std::make_shared<ZeroTensor>(_zeroInitStruct, ov::element::Type_t::u8, ov::Shape({initOutputsByteSize}), false);

    size_t offset = 0;
    for (const IODescriptor& descriptor : _initsMetadata.at(initIndex).outputs) {
        const auto currentOutputBufferLocation =
            static_cast<unsigned char*>(const_cast<void*>(initOutputsAllocatedTensor->data(ov::element::Type_t::u8))) +
            offset;

        const std::shared_ptr<ov::ITensor> hostTensor =
            ov::make_tensor(descriptor.precision, descriptor.shapeFromCompiler.to_shape(), currentOutputBufferLocation);

        initOutputsViewTensorsVector.push_back(hostTensor);
        initOutputsViewTensorsMap.emplace(descriptor.nameFromCompiler, hostTensor);
        offset += ov::util::get_memory_size(descriptor.precision, shape_size(descriptor.shapeFromCompiler.to_shape()));
    }

    return {std::move(initOutputsViewTensorsVector), initOutputsAllocatedTensor, std::move(initOutputsViewTensorsMap)};
}

void WeightlessGraph::run_init_single_threaded() {
    const size_t numberOfInits = _initsGraphDesc.size();

    for (size_t initIndex = 0; initIndex < numberOfInits; ++initIndex) {
        auto [initInputsViewTensors, initInputsAllocatedTensor] = allocate_inputs(initIndex, _constants);
        auto [initOutputsViewTensors, initOutputsAllocatedTensor, initOutputsViewTensorsMap] =
            allocate_outputs(initIndex);

        // Create zero-pipeline and run it (infer init schedule)
        create_pipeline(initIndex, initInputsViewTensors, initOutputsViewTensors);
        run_pipeline(initIndex);

        if (_shareInitOutputs) {
            if (auto owners = InitOutputsRegistry::get().share(_zeroInitStruct,
                                                               initOutputsAllocatedTensor,
                                                               initOutputsViewTensorsMap,
                                                               _wgLogger)) {
                merge_two_maps(_mainInputsViewTensors, initOutputsViewTensorsMap);
                _mainInputsAllocatedTensors.insert(_mainInputsAllocatedTensors.end(), owners->begin(), owners->end());
                _sharedInitOutputsToRelease.push_back(std::move(initOutputsAllocatedTensor));
                continue;
            }
        }
        merge_two_maps(_mainInputsViewTensors, initOutputsViewTensorsMap);
        _mainInputsAllocatedTensors.push_back(std::move(initOutputsAllocatedTensor));
    }
    // Ensure constants are no longer kept in memory after all init schedules have been run
    _constants.clear();
}

void WeightlessGraph::run_init_multi_threaded() {
    if (_initsGraphDesc.size() == 1) {
        _wgLogger.info("::run_init_multi_threaded() for single init - fallback to ::runInit()");
        run_init_single_threaded();
        return;
    }

    // the pipeline:
    // allocate I/O -> create Pipeline -> run Pipeline
    //                                    allocate I/O -> create Pipeline -> run Pipeline
    Parallelizer multiThreadedRunner(
        std::move(_constants),
        [&](std::unordered_map<size_t, std::shared_ptr<ov::op::v0::Constant>>& constants,
            int64_t initIndex) -> QueueData {
            QueueData data{};
            data.initIndex = initIndex;
            data.inputs = allocate_inputs(initIndex, constants);
            data.outputs = allocate_outputs(initIndex);
            return data;
        },
        [&](QueueData&& data, std::condition_variable& cv, std::atomic_bool& flag) {
            // Create zero-pipeline and run it (infer init schedule)
            create_pipeline(data.initIndex, data.inputs.tensors, data.outputs.tensors);

            // progress task 1:
            flag.store(true);
            cv.notify_one();

            run_pipeline(data.initIndex);

            // TODO: pre-allocate those well in advance? (outside of this loop)
            if (_shareInitOutputs) {
                if (auto owners = InitOutputsRegistry::get().share(_zeroInitStruct,
                                                                   data.outputs.hostTensor,
                                                                   data.outputs.tensorsMap,
                                                                   _wgLogger)) {
                    merge_two_maps(_mainInputsViewTensors, data.outputs.tensorsMap);
                    _mainInputsAllocatedTensors.insert(_mainInputsAllocatedTensors.end(),
                                                       owners->begin(),
                                                       owners->end());
                    _sharedInitOutputsToRelease.push_back(data.outputs.hostTensor);
                    return;
                }
            }
            merge_two_maps(_mainInputsViewTensors, data.outputs.tensorsMap);
            _mainInputsAllocatedTensors.push_back(data.outputs.hostTensor);
        },
        _wgLogger);

    multiThreadedRunner.callForAllAndWait(_initsGraphDesc.size());
}

void WeightlessGraph::create_pipeline(const size_t initIndex,
                                      const std::vector<std::shared_ptr<ov::ITensor>>& inputTensors,
                                      const std::vector<std::shared_ptr<ov::ITensor>>& outputTensors) {
    _wgLogger.debug("Init Pipeline - initialize started");

    if (_zeGraphExt == nullptr) {
        OPENVINO_THROW("Zero compiler adapter wasn't initialized");
    }
    if (_zeroInitStruct == nullptr) {
        OPENVINO_THROW("Zero init structure wasn't initialized");
    }

    _initsCommandLists.at(initIndex) = std::make_unique<CommandList>(_zeroInitStruct);
    _initsFences.at(initIndex) = std::make_unique<Fence>(_initsCommandQueue);

    size_t io_index = 0;
    for (const auto& desc : _initsMetadata.at(initIndex).inputs) {
        const void* data = inputTensors.at(io_index++)->data();
        _zeGraphExt->setGraphArgumentValue(_initsGraphDesc.at(initIndex),
                                           desc.indexUsedByDriver,
                                           static_cast<const unsigned char*>(data));
    }

    io_index = 0;
    for (const auto& desc : _initsMetadata.at(initIndex).outputs) {
        const void* data = outputTensors.at(io_index++)->data();
        _zeGraphExt->setGraphArgumentValue(_initsGraphDesc.at(initIndex),
                                           desc.indexUsedByDriver,
                                           static_cast<const unsigned char*>(data));
    }

    _initsCommandLists.at(initIndex)->appendGraphExecute(
        static_cast<ze_graph_handle_t>(_initsGraphDesc.at(initIndex)._handle),
        nullptr);

    _wgLogger.debug("Init Pipeline - initialize completed");
}

void WeightlessGraph::run_pipeline(const size_t initIndex) {
    _wgLogger.debug("Init Pipeline - push() started");
    _initsCommandLists.at(initIndex)->close();

    _initsCommandQueue->executeCommandList(*_initsCommandLists.at(initIndex), *_initsFences.at(initIndex));
    _wgLogger.debug("Init Pipeline - push() completed");

    _wgLogger.debug("Init Pipeline - pull() started");
    _initsFences.at(initIndex)->hostSynchronize();
    _wgLogger.debug("Init Pipeline - pull() completed");
}

void WeightlessGraph::set_weights_inputs() {
    for (const auto& desc : _metadata.inputs) {
        if (!desc.isMainInputWeights) {
            continue;
        }

        OPENVINO_ASSERT(_mainInputsViewTensors.count(desc.nameFromCompiler),
                        "Mismatch between main inputs and init outputs. The input of the main schedule \"",
                        desc.nameFromCompiler,
                        "\" has no correspondent within the init outputs.");
        std::shared_ptr<ov::ITensor> weightsTensor = _mainInputsViewTensors.at(desc.nameFromCompiler);
        set_argument_value(desc.indexUsedByDriver, static_cast<unsigned char*>(weightsTensor->data()));
    }
}

void WeightlessGraph::release_init_blob(const size_t initIndex) {
    if (_blobIsPersistent || _initBlobs == std::nullopt || _zeGraphExt == nullptr ||
        _zeGraphExt->isBlobDataImported(_graphDesc) ||
        !_zeGraphExt->isInitStageRequired(_initsGraphDesc.at(initIndex))) {
        return;
    }

    _initBlobs->at(initIndex) = ov::Tensor();
    _wgLogger.debug("Init blob is released");
}

void WeightlessGraph::release_graphs() {
    if (_zeGraphExt != nullptr) {
        for (auto& initGraphDesc : _initsGraphDesc) {
            if (initGraphDesc._handle) {
                _zeGraphExt->destroyGraph(initGraphDesc);
            }
        }

        _wgLogger.debug("Init graphs are destroyed");
    }

    if (!_blobIsPersistent && _initBlobs != std::nullopt) {
        for (size_t initIndex = 0; initIndex < _initBlobs->size(); ++initIndex) {
            if (_initBlobs->at(initIndex)) {
                _initBlobs->at(initIndex) = ov::Tensor();
            }
        }

        _wgLogger.debug("Init blobs are released");
    }
}

WeightlessGraph::~WeightlessGraph() {
    // make sure all the context-dependent components are destroyed before the zero context is destroyed
    release_graphs();
}

}  // namespace intel_npu
