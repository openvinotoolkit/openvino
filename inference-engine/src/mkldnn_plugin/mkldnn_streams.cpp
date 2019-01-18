// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <map>
#include <vector>
#include <limits>
#include <chrono>
#include <climits>
#include <memory>

#include "mkldnn_graph.h"
#include "ie_parallel.hpp"
#include "mkldnn_streams.h"

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

namespace MKLDNNPlugin {

thread_local MultiWorkerTaskContext MultiWorkerTaskExecutor::ptrContext;

bool check_env_variables() {
#if IE_THREAD == IE_THREAD_OMP
    return MKLDNNPlugin::cpu::checkOpenMpEnvVars(false);
#else
    return false;
#endif
}

#if !(defined(__APPLE__) || defined(_WIN32))
/* Get the cores affinity mask for the current process */
bool get_process_mask(int& ncpus, cpu_set_t*& mask) {
    for (ncpus = sizeof(cpu_set_t) / CHAR_BIT; ncpus < 1024 /* reasonable limit of #cores*/; ncpus <<= 1) {
        mask = CPU_ALLOC(ncpus);
        if (!mask) return false;

        const size_t size = CPU_ALLOC_SIZE(ncpus);
        CPU_ZERO_S(size, mask);
        const int err = sched_getaffinity(getpid(), size, mask);
        // the result fits the mask
        if (!err) break;
        // mask size is not enough
        CPU_FREE(mask);
        mask = NULL;
        // other error
        if (errno != EINVAL) break;
    }
    if (!mask) {
        return false;
    }
    return true;
}
/* Pin current thread to a set of cores determined by the mask. */
bool pin_current_thread_by_mask(int ncores, const cpu_set_t* proc_mask) {
    return 0 == sched_setaffinity(0, ncores, proc_mask);
}
/* Pin thread to a spare core in the round-robin scheme, while respecting the given process mask.
 * The function can also handle the hyper-threading (by populating the physical cores first) */
bool pin_thread_to_vacant_core(int thr_idx, int hyperthreads, int ncores, const cpu_set_t* proc_mask) {
    const size_t size = CPU_ALLOC_SIZE(ncores);
    const int num_cpus = CPU_COUNT_S(size, proc_mask);
    thr_idx %= num_cpus;  // To limit unique number in [; num_cpus-1] range

    // Place threads with specified step
    int cpu_idx = 0;
    for (int i = 0, offset = 0; i < thr_idx; ++i) {
        cpu_idx += hyperthreads;
        if (cpu_idx >= num_cpus)
            cpu_idx = ++offset;
    }

    // Find index of 'cpu_idx'-th bit that equals to 1
    int mapped_idx = -1;
    while (cpu_idx >= 0) {
        if (CPU_ISSET_S(++mapped_idx, size, proc_mask))
            --cpu_idx;
    }

    cpu_set_t *target_mask = CPU_ALLOC(ncores);
    CPU_ZERO_S(size, target_mask);
    CPU_SET_S(mapped_idx, size, target_mask);
    bool res = pin_current_thread_by_mask(size, target_mask);
    CPU_FREE(target_mask);
    return res;
}
#else   // no threads pinning/binding on Win/MacOS
bool get_process_mask(int& ncpus, cpu_set_t*& mask) {
    ncpus = 0;
    mask =  nullptr;
    return false;
}
bool pin_thread_to_vacant_core(int thr_idx, int hyperthreads, int ncores, const cpu_set_t* proc_mask) {
    return false;
}
bool pin_current_thread_by_mask(int ncores, const cpu_set_t* proc_mask) {
    return false;
}
#endif  // !(defined(__APPLE__) || defined(_WIN32))

MultiWorkerTaskExecutor::MultiWorkerTaskExecutor(const std::vector<Task::Ptr>& init_tasks, std::string name) :
        _isStopped(false), _name(name), _initCount(0) {
    for (auto t : init_tasks) {
        _threads.push_back(std::thread([&, t] {
            // initialization (no contention, every worker thread is doing it's own task)
            t->runNoThrowNoBusyCheck();
            _initCount++;

            while (!_isStopped) {
                bool isQueueEmpty;
                Task::Ptr currentTask = nullptr;
                {  // waiting for the new task or for stop signal
                    std::unique_lock<std::mutex> lock(_queueMutex);
                    _queueCondVar.wait(lock, [&]() { return !_taskQueue.empty() || _isStopped; });
                    isQueueEmpty = _taskQueue.empty();
                    if (!isQueueEmpty) {
                        currentTask = _taskQueue.front();
                        _taskQueue.pop();
                        isQueueEmpty = _taskQueue.empty();
                    }
                }
                if (currentTask)
                    currentTask->runNoThrowNoBusyCheck();
                if (_isStopped)
                    break;
                if (isQueueEmpty)  // notify dtor, that all tasks were completed
                    _queueCondVar.notify_all();
            }
        }));
    }
    while (_initCount != init_tasks.size()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

MultiWorkerTaskExecutor::~MultiWorkerTaskExecutor() {
    {
        std::unique_lock<std::mutex> lock(_queueMutex);
        if (!_taskQueue.empty()) {
            _queueCondVar.wait(lock, [this]() { return _taskQueue.empty(); });
        }
        _isStopped = true;
        _queueCondVar.notify_all();
    }
    for (auto& thread : _threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

bool MultiWorkerTaskExecutor::startTask(Task::Ptr task) {
    if (!task->occupy()) return false;
    std::unique_lock<std::mutex> lock(_queueMutex);
    _taskQueue.push(task);
    _queueCondVar.notify_one();
    return true;
}

MKLDNNPlugin::MKLDNNGraphlessInferRequest::MKLDNNGraphlessInferRequest(InferenceEngine::InputsDataMap networkInputs,
                                                                       InferenceEngine::OutputsDataMap networkOutputs)
        : InferRequestInternal(networkInputs, networkOutputs), m_curBatch(-1) {
    // Allocate all input blobs
    for (const auto& it : networkInputs) {
        InferenceEngine::Blob::Ptr blob;
        GetBlob(it.first.c_str(), blob);
    }
    // Allocate all output blobs
    for (const auto& it : networkOutputs) {
        InferenceEngine::Blob::Ptr blob;
        GetBlob(it.first.c_str(), blob);
    }
}


void MKLDNNPlugin::MKLDNNGraphlessInferRequest::InferImpl() {
    IE_PROFILING_AUTO_SCOPE(MKLDNN_INFER)

    auto infer = [this] {
        IE_ASSERT(MKLDNNPlugin::MultiWorkerTaskExecutor::ptrContext.ptrGraph != nullptr);
        MKLDNNGraph::Ptr graph = MKLDNNPlugin::MultiWorkerTaskExecutor::ptrContext.ptrGraph;
        if (!graph->IsReady())
            THROW_IE_EXCEPTION << "Network not loaded.";
        if (m_curBatch > 0 && !graph->getProperty().enableDynamicBatch)
            THROW_IE_EXCEPTION << "Dynamic batch is not enabled.";

        if (m_curBatch > graph->getProperty().batchLimit)
            THROW_IE_EXCEPTION << "Invalid dynamic batch size " << m_curBatch <<
                               " for this request.";

        // execute input pre-processing.
        execDataPreprocessing(_inputs);

        // need to retain converted blobs until infer finish
        std::vector<InferenceEngine::Blob::Ptr> convertedInputs;
        for (auto input : _inputs) {
            if (!_networkInputs[input.first]) {
                THROW_IE_EXCEPTION <<
                                   "input blobs map contains not registered during IInferencePlugin::LoadNetwork blob with name "
                                   << input.first;
            }
            InferenceEngine::Blob::Ptr iconv;
            InferenceEngine::TBlob<float> *in_f = nullptr;
            switch (input.second->precision()) {
                case InferenceEngine::Precision::FP32:
                    graph->PushInputData(input.first, input.second);
                    break;
                case InferenceEngine::Precision::U16:
                    // U16 is unsupported by mkldnn, so here we convert the blob and send FP32
                    iconv = InferenceEngine::make_shared_blob<float, const InferenceEngine::SizeVector>(
                            InferenceEngine::Precision::FP32,
                            input.second->getTensorDesc().getLayout(), input.second->dims());
                    convertedInputs.push_back(iconv);
                    iconv->allocate();
                    in_f = dynamic_cast<InferenceEngine::TBlob<float> *>(iconv.get());
                    InferenceEngine::copyToFloat<uint16_t>(in_f->data(), input.second.get());
                    graph->PushInputData(input.first, iconv);
                    break;
                case InferenceEngine::Precision::I16:
                    if (graph->hasMeanImageFor(input.first)) {
                        // If a mean image exists, we convert the blob and send FP32
                        iconv = InferenceEngine::make_shared_blob<float, const InferenceEngine::SizeVector>(
                                InferenceEngine::Precision::FP32,
                                input.second->getTensorDesc().getLayout(), input.second->dims());
                        convertedInputs.push_back(iconv);
                        iconv->allocate();
                        in_f = dynamic_cast<InferenceEngine::TBlob<float> *>(iconv.get());
                        InferenceEngine::copyToFloat<int16_t>(in_f->data(), input.second.get());
                        graph->PushInputData(input.first, iconv);
                    } else {
                        // Instead we can send I16 directly
                        graph->PushInputData(input.first, input.second);
                    }
                    break;
                case InferenceEngine::Precision::U8:
                    if (graph->hasMeanImageFor(input.first)) {
                        // If a mean image exists, we convert the blob and send FP32
                        iconv = InferenceEngine::make_shared_blob<float, const InferenceEngine::SizeVector>(
                                InferenceEngine::Precision::FP32,
                                input.second->getTensorDesc().getLayout(), input.second->dims());
                        convertedInputs.push_back(iconv);
                        iconv->allocate();
                        in_f = dynamic_cast<InferenceEngine::TBlob<float> *>(iconv.get());
                        InferenceEngine::copyToFloat<uint8_t>(in_f->data(), input.second.get());
                        graph->PushInputData(input.first, iconv);
                    } else {
                        // Instead we can send I8 directly
                        graph->PushInputData(input.first, input.second);
                    }
                    break;
                default:
                    THROW_IE_EXCEPTION << "Unsupported input precision " << input.second->precision();
            }
        }
        graph->Infer(m_curBatch);
        graph->PullOutputData(_outputs);
        if (graph->getProperty().collectPerfCounters) {
            m_perfMap.clear();
            graph->GetPerfData(m_perfMap);
        }
    };
#if IE_THREAD == IE_THREAD_TBB
    auto_scope_observing observer(MKLDNNPlugin::MultiWorkerTaskExecutor::ptrContext.ptrGraph->ptrObserver);
    // a TBB arena is made "this" for Infer call via executing lambda for the arena
    MKLDNNPlugin::MultiWorkerTaskExecutor::ptrContext.ptrGraph->ptrArena->execute([&] { infer(); });
#else
    infer();
#endif
}

void MKLDNNPlugin::MKLDNNGraphlessInferRequest::GetPerformanceCounts(
        std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> &perfMap) const {
    perfMap = m_perfMap;
}

void MKLDNNPlugin::MKLDNNGraphlessInferRequest::GetBlob(const char *name, InferenceEngine::Blob::Ptr &data) {
    // ROI blob is returned only if it was set previously.
    auto it = _preProcData.find(name);
    if (it != _preProcData.end()) {
        data = it->second.getRoiBlob();
        return;
    }

    if (_inputs.find(name) != _inputs.end()) {
        data = _inputs[name];
        checkBlob(data, name, true);
        return;
    } else if (_networkInputs.find(name) != _networkInputs.end()) {
        InferenceEngine::Layout l = _networkInputs[name]->getLayout();
        InferenceEngine::Precision p = _networkInputs[name]->getPrecision();
        InferenceEngine::SizeVector dims = _networkInputs[name]->getTensorDesc().getDims();

        InferenceEngine::TensorDesc desc = InferenceEngine::TensorDesc(p, dims, l);
        _inputs[name] = data = make_blob_with_precision(desc);
        _inputs[name]->allocate();
        checkBlob(data, name, true);
        return;
    }

    if (_outputs.find(name) != _outputs.end()) {
        data = _outputs[name];
        checkBlob(data, name, false);
        return;
    } else if (_networkOutputs.find(name) != _networkOutputs.end()) {
        InferenceEngine::Layout l = _networkOutputs[name]->getLayout();
        InferenceEngine::Precision p = _networkOutputs[name]->getPrecision();
        InferenceEngine::SizeVector dims = _networkOutputs[name]->getTensorDesc().getDims();

        InferenceEngine::TensorDesc desc = InferenceEngine::TensorDesc(p, dims, l);
        _outputs[name] = data = make_blob_with_precision(desc);
        _outputs[name]->allocate();
        checkBlob(data, name, false);
        return;
    }

    THROW_IE_EXCEPTION << "Cannot find blob with name: " << name;
}

void MKLDNNPlugin::MKLDNNGraphlessInferRequest::SetBlob(const char *name, const InferenceEngine::Blob::Ptr &data) {
    if (!data)
        THROW_IE_EXCEPTION << NOT_ALLOCATED_str << "Failed to set empty blob with name: \'" << name << "\'";
    if (data->buffer() == nullptr)
        THROW_IE_EXCEPTION << "Input data was not allocated. Input name: \'" << name << "\'";
    if (name == nullptr) {
        THROW_IE_EXCEPTION << NOT_FOUND_str + "Failed to set blob with empty name";
    }
    InferenceEngine::InputInfo::Ptr foundInput;
    InferenceEngine::DataPtr foundOutput;
    size_t dataSize = data->size();
    if (findInputAndOutputBlobByName(name, foundInput, foundOutput)) {
        if (foundInput->getInputPrecision() != data->precision()) {
            THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str << "Failed to set Blob with precision "
                               << data->precision();
        }

        if (foundInput->getPreProcess().getResizeAlgorithm() != InferenceEngine::ResizeAlgorithm::NO_RESIZE) {
            // Stores the given blob as ROI blob. It will be used to fill in network input during pre-processing.
            _preProcData[name].setRoiBlob(data);
        } else {
            size_t inputSize = InferenceEngine::details::product(foundInput->getDims());
            if (dataSize != inputSize) {
                THROW_IE_EXCEPTION << "Input blob size is not equal network input size ("
                                   << dataSize << "!=" << inputSize << ").";
            }
            _inputs[name] = data;
        }
    } else {
        size_t outputSize = InferenceEngine::details::product(foundOutput->getDims());
        if (dataSize != outputSize) {
            THROW_IE_EXCEPTION << "Output blob size is not equal network output size ("
                               << dataSize << "!=" << outputSize << ").";
        }
        if (foundOutput->getPrecision() != data->precision()) {
            THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str
                               << "Failed to set Blob with precision not corresponding to user output precision";
        }
        _outputs[name] = data;
    }
}

void MKLDNNPlugin::MKLDNNGraphlessInferRequest::SetBatch(int new_batch) {
    if (new_batch < 1) {
        THROW_IE_EXCEPTION << "Invalid dynamic batch size " << new_batch <<
                           " for this request.";
    }
    m_curBatch = new_batch;
}

}  // namespace MKLDNNPlugin
