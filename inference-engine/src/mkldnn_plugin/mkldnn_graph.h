// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ie_parallel.hpp"
#include "config.h"
#include "mkldnn_memory.h"
#include "mean_image.h"
#include "mkldnn_node.h"
#include "mkldnn_edge.h"
#include "mkldnn_streams.h"

#include <map>
#include <string>
#include <vector>
#include <memory>

namespace MKLDNNPlugin {

class MKLDNNGraph {
public:
    typedef std::shared_ptr<MKLDNNGraph> Ptr;
    int socket;

    enum Status {
        NotReady = 0,
        Ready = 1,
    };

    MKLDNNGraph(): status(NotReady), eng(mkldnn::engine(mkldnn::engine::kind::cpu, 0)), socket(0) {}

    Status GetStatus() {
        return status;
    }

    bool IsReady() {
        return (GetStatus() == Ready);
    }

    void setConfig(const Config &cfg);
    void setProperty(const std::map<std::string, std::string> &properties);
    Config getProperty();

    void getInputBlobs(InferenceEngine::BlobMap &in_map);
    void getOutputBlobs(InferenceEngine::BlobMap &out_map);

    template<typename NET>
    void CreateGraph(const NET &network,
                     const MKLDNNExtensionManager::Ptr& extMgr,
                     int socket = 0);

    bool hasMeanImageFor(const std::string& name) {
        return _meanImages.find(name) != _meanImages.end();
    }

    void PushInputData(const std::string& name, const InferenceEngine::Blob::Ptr &in);
    void PullOutputData(InferenceEngine::BlobMap &out);

    void Infer(int batch = -1);

    std::vector<MKLDNNNodePtr>& GetNodes() {
        return graphNodes;
    }

    std::vector<MKLDNNEdgePtr>& GetEdges() {
        return graphEdges;
    }

    std::vector<MKLDNNNodePtr>& GetOutputNodes() {
        return outputNodes;
    }

    mkldnn::engine getEngine() const {
        return eng;
    }

    void GetPerfData(std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> &perfMap) const;

    void RemoveDroppedNodes();
    void RemoveDroppedEdges();
    void DropNode(const MKLDNNNodePtr& node);

    void CreateArena(int threads_per_stream) {
        #if IE_THREAD == IE_THREAD_OMP
        omp_set_num_threads(threads_per_stream);
        #elif(IE_THREAD == IE_THREAD_TBB || IE_THREAD == IE_THREAD_TBB_AUTO)
        ptrArena = std::unique_ptr<tbb::task_arena>(new tbb::task_arena(threads_per_stream));
        #endif
    }

    void CreateObserver(int _stream_id, int _threads_per_stream, int _pinning_step = 1) {
        #if (IE_THREAD == IE_THREAD_TBB || IE_THREAD == IE_THREAD_TBB_AUTO)
        ptrObserver
                = std::unique_ptr<tbb::task_scheduler_observer>(
                new pinning_observer(*ptrArena.get(), _stream_id, _threads_per_stream, _pinning_step));
        #else
        cpu_set_t *process_mask = nullptr;
        int ncpus = 0;
        get_process_mask(ncpus, process_mask);
            #if IE_THREAD == IE_THREAD_OMP
            #pragma omp parallel for
                    for (int thread_index = 0; thread_index < _threads_per_stream; thread_index++) {
                        pin_thread_to_vacant_core(_stream_id * _threads_per_stream + thread_index, 1, ncpus, process_mask);
                    }
            #elif IE_THREAD == IE_THREAD_SEQ
            pin_thread_to_vacant_core(_stream_id * _threads_per_stream, 1, ncpus, process_mask);
            #endif
        CPU_FREE(process_mask);
        #endif
    }

    InferenceEngine::ICNNNetwork::Ptr dump() const;

    template<typename NET>
    static void ApplyUnrollPasses(NET &net);

    void ResetInferCount() { infer_count = 0; }

protected:
    void VisitNode(MKLDNNNodePtr node, std::vector<MKLDNNNodePtr>& sortedNodes);
    void SortTopologically();

    void ForgetGraphData() {
        status = NotReady;
        eng = mkldnn::engine(mkldnn::engine::kind::cpu, 0);

        inputNodes.clear();
        outputNodes.clear();
        graphNodes.clear();
        graphEdges.clear();
        _meanImages.clear();
    }
    Status status;
    Config config;

    // For dumping purposes. -1 - no counting, all other positive
    // values mean increment it within each Infer() call
    int infer_count = -1;

    bool reuse_io_tensors = true;

    MKLDNNMemoryPtr memWorkspace;

    std::map<std::string, MKLDNNNodePtr> inputNodes;
    std::vector<MKLDNNNodePtr> outputNodes;
    std::vector<MKLDNNNodePtr> graphNodes;
    std::vector<MKLDNNEdgePtr> graphEdges;

    std::map<std::string, MeanImage> _meanImages;
    std::string _name;

    #if (IE_THREAD == IE_THREAD_TBB || IE_THREAD == IE_THREAD_TBB_AUTO)
    std::unique_ptr<tbb::task_arena> ptrArena;
    std::unique_ptr<tbb::task_scheduler_observer> ptrObserver;
    #endif
    mkldnn::engine eng;

    void Replicate(const ICNNNetwork &network, const MKLDNNExtensionManager::Ptr& extMgr);
    void Replicate(const TensorIterator::Body &subgraph, const MKLDNNExtensionManager::Ptr& extMgr);
    void InitGraph();
    void InitNodes();
    void InitEdges();
    void Allocate();
    void AllocateWithReuse();
    void CreatePrimitives();

    void do_before(const std::string &dir, const MKLDNNNodePtr &node);
    void do_after(const std::string &dir, const MKLDNNNodePtr &node);

    friend class MKLDNNInferRequest;
    friend class MKLDNNGraphlessInferRequest;
    friend std::shared_ptr<InferenceEngine::ICNNNetwork> dump_graph_as_ie_net(const MKLDNNGraph &graph);

private:
    void dumpToDotFile(std::string file) const;
    struct ParsedLayer {
        MKLDNNNodePtr parent;
        InferenceEngine::CNNLayerPtr cnnLayer;
        size_t outIdx;
    };
};

}  // namespace MKLDNNPlugin
