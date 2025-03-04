// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/detection_output.hpp"

#include "detection_output.h"
#include "onednn/dnnl.h"
#include "openvino/core/parallel.hpp"

using namespace dnnl;

namespace ov::intel_cpu::node {
namespace {

template <typename T>
bool SortScorePairDescend(const std::pair<float, T>& pair1, const std::pair<float, T>& pair2) {
    return (pair1.first > pair2.first) || (pair1.first == pair2.first && pair1.second < pair2.second);
}

template <>
bool SortScorePairDescend<std::pair<int, int>>(const std::pair<float, std::pair<int, int>>& pair1,
                                               const std::pair<float, std::pair<int, int>>& pair2) {
    return (pair1.first > pair2.first) || (pair1.first == pair2.first && pair1.second.second < pair2.second.second);
}

}  // namespace

bool DetectionOutput::isSupportedOperation(const std::shared_ptr<const ov::Node>& op,
                                           std::string& errorMessage) noexcept {
    try {
        const auto doOp = ov::as_type_ptr<const ov::op::v8::DetectionOutput>(op);
        if (!doOp) {
            errorMessage = "Node is not an instance of the DetectionOutput from the operations set v8.";
            return false;
        }
        if (!ov::intel_cpu::CaselessEq<std::string>()(doOp->get_attrs().code_type,
                                                      "caffe.PriorBoxParameter.CENTER_SIZE") &&
            !ov::intel_cpu::CaselessEq<std::string>()(doOp->get_attrs().code_type, "caffe.PriorBoxParameter.CORNER")) {
            errorMessage = "Unsupported code_type attribute: " + doOp->get_attrs().code_type;
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

DetectionOutput::DetectionOutput(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }

    if (getOriginalInputsNumber() != 3 && getOriginalInputsNumber() != 5) {
        THROW_CPU_NODE_ERR("has incorrect number of input edges.");
    }

    if (getOriginalOutputsNumber() != 1) {
        THROW_CPU_NODE_ERR("has incorrect number of output edges.");
    }

    auto doOp = ov::as_type_ptr<const ov::op::v8::DetectionOutput>(op);
    auto attributes = doOp->get_attrs();

    backgroundClassId = attributes.background_label_id;
    topK = attributes.top_k;
    varianceEncodedInTarget = attributes.variance_encoded_in_target;
    keepTopK = attributes.keep_top_k[0];
    NMSThreshold = attributes.nms_threshold;
    confidenceThreshold = attributes.confidence_threshold;
    isShareLoc = attributes.share_location;
    clipBeforeNMS = attributes.clip_before_nms;
    clipAfterNMS = attributes.clip_after_nms;
    decreaseClassId = attributes.decrease_label_id;
    normalized = attributes.normalized;
    imgHeight = attributes.input_height;
    imgWidth = attributes.input_width;
    priorSize = normalized ? 4 : 5;
    coordOffset = normalized ? 0 : 1;
    cacheSizeL3 = utils::get_cache_size(3, true);

    withAddBoxPred = getOriginalInputsNumber() == 5;
    objScore = attributes.objectness_score;

    codeType = (ov::intel_cpu::CaselessEq<std::string>()(attributes.code_type, "caffe.PriorBoxParameter.CENTER_SIZE")
                    ? CodeType::CENTER_SIZE
                    : CodeType::CORNER);
}

void DetectionOutput::prepareParams() {
    const auto& idPriorDims = getParentEdgeAt(ID_PRIOR)->getMemory().getShape().getStaticDims();
    const auto& idConfDims = getParentEdgeAt(ID_CONF)->getMemory().getShape().getStaticDims();
    priorsNum = static_cast<int>(idPriorDims.back() / priorSize);
    isPriorsPerImg = idPriorDims.front() != 1;
    classesNum = static_cast<int>(idConfDims.back() / priorsNum);
    locNumForClasses = isShareLoc ? 1 : classesNum;

    const auto& idLocDims = getParentEdgeAt(ID_LOC)->getMemory().getShape().getStaticDims();
    if (priorsNum * locNumForClasses * 4 != static_cast<int>(idLocDims[1])) {
        THROW_CPU_NODE_ERR("has incorrect number of priors, which must match number of location predictions (",
                           priorsNum * locNumForClasses * 4,
                           " vs ",
                           idLocDims[1],
                           ")");
    }

    if (priorsNum * classesNum != static_cast<int>(idConfDims.back())) {
        THROW_CPU_NODE_ERR("has incorrect number of priors, which must match number of confidence predictions.");
    }

    if (decreaseClassId && backgroundClassId != 0) {
        THROW_CPU_NODE_ERR("cannot use decrease_label_id and background_label_id parameter simultaneously.");
    }

    imgNum = static_cast<int>(idConfDims[0]);

    decodedBboxes.resize(imgNum * classesNum * priorsNum * 4);
    bboxSizes.resize(imgNum * classesNum * priorsNum);
    indicesBuffer.resize(imgNum * classesNum * priorsNum);
    indices.resize(imgNum * classesNum * priorsNum);
    // prior info for shared_location
    if (isShareLoc) {
        confInfoForPrior.resize(imgNum * priorsNum);
    }

    // confs...count...indices for caffe style and sparsity case.
    // caffe: filter(conf_info for sparsity or indices for dense) --> topk(buffer) --> nms(indices)
    //        --> g_topk(vector<>(all detections) --> indices per class))
    // MXNet: max conf for prior within img, filter(indices) --> topk_img(buffer) --> nms_cls(indices)
    //        --> g_topk(vector<>(all detections) --> indices per class))
    isSparsityWorthwhile = (confidenceThreshold > sparsityThreshold) &&
                           ((classesNum * priorsNum * sizeof(float) * 2) > static_cast<size_t>(cacheSizeL3));
    confInfoLen = (!decreaseClassId && isSparsityWorthwhile) ? (2 * priorsNum + 1) : priorsNum;
    reorderedConf.resize(imgNum * classesNum * confInfoLen);

    detectionsCount.resize(imgNum * classesNum);
    numPriorsActual.resize(imgNum);
}

void DetectionOutput::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }

    std::vector<PortConfigurator> inDataConf;
    inDataConf.reserve(inputShapes.size());
    for (size_t i = 0; i < inputShapes.size(); ++i) {
        inDataConf.emplace_back(LayoutType::ncsp, ov::element::f32);
    }

    addSupportedPrimDesc(inDataConf, {{LayoutType::ncsp, ov::element::f32}}, impl_desc_type::ref_any);
}

struct ConfidenceComparatorDO {
    explicit ConfidenceComparatorDO(const float* confDataIn) : confData(confDataIn) {}

    bool operator()(int idx1, int idx2) {
        if (confData[idx1] > confData[idx2]) {
            return true;
        }
        if (confData[idx1] < confData[idx2]) {
            return false;
        }
        return idx1 < idx2;
    }

    const float* confData;
};

void DetectionOutput::executeDynamicImpl(const dnnl::stream& strm) {
    execute(strm);
}

void DetectionOutput::execute(const dnnl::stream& strm) {
    auto* dstData = getDstDataAtPortAs<float>(0);

    const auto* locData = getSrcDataAtPortAs<const float>(ID_LOC);
    const auto* confData = getSrcDataAtPortAs<const float>(ID_CONF);
    const auto* priorData = getSrcDataAtPortAs<const float>(ID_PRIOR);
    const float* ARMConfData = inputShapes.size() > 3 ? getSrcDataAtPortAs<const float>(ID_ARM_CONF) : nullptr;
    const float* ARMLocData = inputShapes.size() > 4 ? getSrcDataAtPortAs<const float>(ID_ARM_LOC) : nullptr;

    float* reorderedConfData = reorderedConf.data();
    auto* reorderedConfDataIndices = reinterpret_cast<int*>(reorderedConf.data());

    float* decodedBboxesData = decodedBboxes.data();
    float* bboxSizesData = bboxSizes.data();
    int* indicesData = indices.data();
    int* indicesBufData = indicesBuffer.data();
    int* detectionsData = detectionsCount.data();

    memset(detectionsData, 0, imgNum * classesNum * sizeof(int));

    int priorsBatch = isPriorsPerImg ? imgNum : 1;
    int* numPriorsActualdata = numPriorsActual.data();
    for (int n = 0; n < priorsBatch; ++n) {
        const float* ppriors = priorData;
        ppriors += varianceEncodedInTarget ? (n * priorsNum * priorSize) : (2 * n * priorsNum * priorSize);
        getActualPriorNum(ppriors, numPriorsActualdata, n);
    }
    if (!isPriorsPerImg && imgNum > 1) {
        std::fill_n(numPriorsActualdata + 1, imgNum - 1, numPriorsActualdata[0]);
    }

    if (!isSparsityWorthwhile) {
        confReorderDense(confData, ARMConfData, reorderedConfData);
    } else {  // sparsity
        if (!decreaseClassId) {
            confReorderAndFilterSparsityCF(confData,
                                           ARMConfData,
                                           reorderedConfData,
                                           indicesData,
                                           indicesBufData,
                                           detectionsData);
        } else {
            confReorderAndFilterSparsityMX(confData,
                                           ARMConfData,
                                           reorderedConfData,
                                           indicesData,
                                           indicesBufData,
                                           detectionsData);
        }
    }

    int* confInfoV = confInfoForPrior.data();

    for (int n = 0; n < imgNum; ++n) {
        const float* ppriors = priorData;
        const float* priorVariances = priorData + priorsNum * priorSize;
        if (isPriorsPerImg) {
            int priorSizePerImg =
                varianceEncodedInTarget ? (n * priorsNum * priorSize) : (2 * n * priorsNum * priorSize);
            ppriors += priorSizePerImg;
            priorVariances += varianceEncodedInTarget ? 0 : priorSizePerImg;
        }

        if (isShareLoc) {
            int locShift = n * priorsNum;
            int coordShift = locShift * 4;
            const float* ploc = locData + coordShift;
            float* pboxes = decodedBboxesData + coordShift;
            float* psizes = bboxSizesData + locShift;
            int* confInfoVB = confInfoV + locShift;

            if (withAddBoxPred) {
                const float* pARMLoc = ARMLocData + coordShift;
                decodeBBoxes(ppriors,
                             pARMLoc,
                             priorVariances,
                             pboxes,
                             psizes,
                             numPriorsActualdata,
                             n,
                             coordOffset,
                             priorSize,
                             true,
                             nullptr,
                             confInfoVB);
                decodeBBoxes(pboxes,
                             ploc,
                             priorVariances,
                             pboxes,
                             psizes,
                             numPriorsActualdata,
                             n,
                             0,
                             4,
                             false,
                             nullptr,
                             confInfoVB);
            } else {
                decodeBBoxes(ppriors,
                             ploc,
                             priorVariances,
                             pboxes,
                             psizes,
                             numPriorsActualdata,
                             n,
                             coordOffset,
                             priorSize,
                             true,
                             nullptr,
                             confInfoVB);
            }
        } else {
            for (int c = 0; c < locNumForClasses; ++c) {
                if (c == backgroundClassId) {
                    continue;
                }
                int locShift = n * priorsNum * locNumForClasses;
                int coordShift = locShift * 4;
                const float* ploc = locData + coordShift + c * 4;
                float* pboxes = decodedBboxesData + coordShift + c * 4 * priorsNum;
                float* psizes = bboxSizesData + locShift + c * priorsNum;
                int* confInfoHBC = reorderedConfDataIndices + n * confInfoLen * classesNum + c * confInfoLen;
                if (withAddBoxPred) {
                    const float* pARMLoc = ARMLocData + n * 4 * locNumForClasses * priorsNum + c * 4;
                    decodeBBoxes(ppriors,
                                 pARMLoc,
                                 priorVariances,
                                 pboxes,
                                 psizes,
                                 numPriorsActualdata,
                                 n,
                                 coordOffset,
                                 priorSize,
                                 true,
                                 confInfoHBC);
                    decodeBBoxes(pboxes,
                                 ploc,
                                 priorVariances,
                                 pboxes,
                                 psizes,
                                 numPriorsActualdata,
                                 n,
                                 0,
                                 4,
                                 false,
                                 confInfoHBC);
                } else {
                    decodeBBoxes(ppriors,
                                 ploc,
                                 priorVariances,
                                 pboxes,
                                 psizes,
                                 numPriorsActualdata,
                                 n,
                                 coordOffset,
                                 priorSize,
                                 true,
                                 confInfoHBC);
                }
            }
        }
    }

    // NMS
    for (int n = 0; n < imgNum; ++n) {
        if (!decreaseClassId) {
            // Caffe style
            parallel_for(classesNum, [&](int c) {
                if (c != backgroundClassId) {  // Ignore background class
                    const int off = n * priorsNum * classesNum + c * priorsNum;
                    const float* pconfReorder = reorderedConfData + off;
                    int* pindices = indicesData + off;
                    int* pbuffer = indicesBufData + off;
                    int* pdetections = detectionsData + n * classesNum + c;

                    if (!isSparsityWorthwhile) {
                        confFilterCF(pconfReorder, pindices, pbuffer, pdetections, n);
                    }

                    const float* pboxes;
                    const float* psizes;
                    if (isShareLoc) {
                        pboxes = decodedBboxesData + n * 4 * priorsNum;
                        psizes = bboxSizesData + n * priorsNum;
                    } else {
                        pboxes = decodedBboxesData + n * 4 * classesNum * priorsNum + c * 4 * priorsNum;
                        psizes = bboxSizesData + n * classesNum * priorsNum + c * priorsNum;
                    }

                    NMSCF(pbuffer, *pdetections, pindices, pboxes, psizes);
                }
            });
        } else {
            // MXNet style
            const int offImg = n * priorsNum * classesNum;
            const float* pconf = confData + offImg;
            float* pconfReorder = reorderedConfData + offImg;
            int* pbuffer = indicesBufData + offImg;
            int* pindices = indicesData + offImg;
            int* pdetections = detectionsData + n * classesNum;

            if (!isSparsityWorthwhile) {
                confFilterMX(pconf, ARMConfData, pconfReorder, pindices, pbuffer, pdetections, n);
            }

            const float* pboxes = decodedBboxesData + n * 4 * locNumForClasses * priorsNum;
            const float* psizes = bboxSizesData + n * locNumForClasses * priorsNum;

            NMSMX(pbuffer, pdetections, pindices, pboxes, psizes);
        }

        int detectionsTotal = 0;
        detectionsTotal = parallel_sum(classesNum, detectionsTotal, [&](size_t c) -> int {
            return detectionsData[n * classesNum + c];
        });

        // combine detections of all class for this image and filter with global(image) topk(keep_topk)
        if (keepTopK > -1 && detectionsTotal > keepTopK) {
            std::vector<std::pair<float, std::pair<int, int>>> confIndicesClassMap;

            std::mutex mtx;
            parallel_for(classesNum, [&](int c) {
                const int detections = detectionsData[n * classesNum + c];
                int* pindices = indicesData + n * classesNum * priorsNum + c * priorsNum;

                float* pconf = reorderedConfData + n * classesNum * confInfoLen + c * confInfoLen;

                for (int i = 0; i < detections; ++i) {
                    int pr = pindices[i];
                    mtx.lock();
                    confIndicesClassMap.emplace_back(pconf[pr], std::make_pair(c, pr));
                    mtx.unlock();
                }
            });

            std::sort(confIndicesClassMap.begin(),
                      confIndicesClassMap.end(),
                      SortScorePairDescend<std::pair<int, int>>);
            confIndicesClassMap.resize(keepTopK);

            // Store the new indices. Assign to class back
            memset(detectionsData + n * classesNum, 0, classesNum * sizeof(int));

            for (auto& j : confIndicesClassMap) {
                const int cls = j.second.first;
                const int pr = j.second.second;
                int* pindices = indicesData + n * classesNum * priorsNum + cls * priorsNum;
                pindices[detectionsData[n * classesNum + cls]] = pr;
                detectionsData[n * classesNum + cls]++;
            }
        }
    }

    // get final output
    generateOutput(reorderedConfData, indicesData, detectionsData, decodedBboxesData, dstData);
}

inline void DetectionOutput::confFilterCF(const float* pconf,
                                          int* pindices,
                                          int* pbuffer,
                                          int* detectionsData,
                                          const int& n) {
    // in:  reorderedConf
    // out: pindices count
    int count = 0;
    for (int i = 0; i < numPriorsActual[n]; ++i) {
        if (pconf[i] > confidenceThreshold) {
            pindices[count] = i;
            count++;
        }
    }

    // in:  pindices count
    // out: buffer detectionCount
    int k = (topK == -1 ? count : (std::min)(topK, count));
    topk(pindices, pbuffer, pconf, count, k);
    detectionsData[0] = k;
}

// MX filter is per image filter, max output is prior num(select max for all class within this prior)
// NMS is per class, keep topk is per image, final output is per class
inline void DetectionOutput::confFilterMX(const float* confData,
                                          const float* ARMConfData,
                                          float* reorderedConfData,
                                          int* indicesData,
                                          int* indicesBufData,
                                          int* detectionsData,
                                          const int& n) {
    std::mutex mtx;
    parallel_for(numPriorsActual[n], [&](size_t p) {
        // in:  origin conf
        // out: pindices, detectionCount
        // intentionally code branch from higher level
        if (withAddBoxPred) {
            const bool isARMPrior = ARMConfData[n * priorsNum * 2 + p * 2 + 1] < objScore;
            float maxConf = -1;
            int maxCIdx = 0;
            for (int c = 1; c < classesNum; ++c) {
                float conf = confData[p * classesNum + c];
                if (isARMPrior) {
                    conf =
                        (c == backgroundClassId) ? 1.0f : 0.0f;  // still need refresh conf due to read from origin conf
                }
                if (conf >= confidenceThreshold && conf > maxConf) {
                    maxConf = conf;
                    maxCIdx = c;
                }
            }
            if (maxCIdx > 0) {
                // include this prior
                mtx.lock();
                indicesData[detectionsData[0]] = maxCIdx * priorsNum + p;  // de-refer to get prior and class id.
                detectionsData[0]++;
                mtx.unlock();
            }
        } else {
            float maxConf = -1;
            int maxCIdx = 0;
            for (int c = 1; c < classesNum; ++c) {
                float conf = confData[p * classesNum + c];
                if (conf >= confidenceThreshold && conf > maxConf) {
                    maxConf = conf;
                    maxCIdx = c;
                }
            }
            if (maxCIdx > 0) {
                // include this prior and class with max conf
                mtx.lock();
                indicesData[detectionsData[0]] = maxCIdx * priorsNum + p;  // de-refer to get prior and class id.
                detectionsData[0]++;
                mtx.unlock();
            }
        }
    });

    // in:  pindices, detectionCount(filtered num)
    // out: buffer, detectionCount(k)
    int count = detectionsData[0];
    int k = (topK == -1 ? count : (std::min)(topK, count));

    const float* pconf = reorderedConfData;
    // int *indices = indicesData;
    // int *pbuffer = indicesBufData;
    topk(indicesData, indicesBufData, pconf, count, k);
    detectionsData[0] = k;
}

inline void DetectionOutput::getActualPriorNum(const float* priorData, int* numPriorsActual, int n) {
    numPriorsActual[n] = priorsNum;
    if (!normalized) {
        int num = 0;
        for (; num < priorsNum; ++num) {
            float imgId = priorData[num * priorSize];
            if (imgId == -1.f) {
                numPriorsActual[n] = num;
                break;
            }
        }
    }
}

inline void DetectionOutput::confReorderDense(const float* confData,
                                              const float* ARMConfData,
                                              float* reorderedConfData) {
    if (withAddBoxPred) {
        parallel_for2d(imgNum, priorsNum, [&](size_t n, size_t p) {
            if (ARMConfData[n * priorsNum * 2 + p * 2 + 1] < objScore) {
                for (int c = 0; c < classesNum; ++c) {
                    reorderedConfData[n * priorsNum * classesNum + c * priorsNum + p] =
                        c == backgroundClassId ? 1.0f : 0.0f;
                }
            } else {
                for (int c = 0; c < classesNum; ++c) {
                    reorderedConfData[n * priorsNum * classesNum + c * priorsNum + p] =
                        confData[n * priorsNum * classesNum + p * classesNum + c];
                }
            }
        });
        return;
    }
    // withAddBoxPred is false
    parallel_for2d(imgNum, classesNum, [&](size_t n, size_t c) {
        const int offset = n * priorsNum * classesNum;
        for (int p = 0; p < priorsNum; ++p) {
            reorderedConfData[offset + c * priorsNum + p] = confData[offset + p * classesNum + c];
        }
    });
}

inline void DetectionOutput::confReorderAndFilterSparsityCF(const float* confData,
                                                            const float* ARMConfData,
                                                            float* reorderedConfData,
                                                            int* indicesData,
                                                            int* indicesBufData,
                                                            int* detectionsData) {
    auto* reorderedConfDataIndices = reinterpret_cast<int*>(reorderedConfData);
    for (int n = 0; n < imgNum; ++n) {
        const int off = n * priorsNum * classesNum;
        const int offV = n * priorsNum;  // vertical info

        const int offH = n * confInfoLen * classesNum;  // horizontal info
        // reset count
        parallel_for(classesNum, [&](size_t c) {
            const int countIdx = offH + c * confInfoLen + priorsNum;
            reorderedConfDataIndices[countIdx] = 0;
        });

        std::mutex mtx;
        parallel_for(numPriorsActual[n], [&](size_t p) {
            // intentionally code branch from higher level
            if (withAddBoxPred) {
                const bool isARMPrior = ARMConfData[n * priorsNum * 2 + p * 2 + 1] < objScore;
                bool priorStatusSet = false;
                if (isShareLoc) {
                    confInfoForPrior[offV + p] = -1;
                }
                int confIdxPrior = off + p * classesNum;
                for (int c = 0; c < classesNum; ++c) {
                    float conf = confData[confIdxPrior + c];
                    if (isARMPrior) {
                        conf = (c == backgroundClassId) ? 1.0f : 0.0f;
                    }
                    if (conf > confidenceThreshold) {
                        const int idx = offH + c * confInfoLen;
                        reorderedConfData[idx + p] = conf;
                        mtx.lock();
                        reorderedConfDataIndices[idx + priorsNum]++;
                        reorderedConfDataIndices[idx + priorsNum + reorderedConfDataIndices[idx + priorsNum]] = p;
                        mtx.unlock();

                        // vertical info for isShareLoc(flag to decode for each prior)
                        if (!priorStatusSet && isShareLoc) {
                            confInfoForPrior[offV + p] = 1;  // 1 for decode
                        }
                    }
                }
            } else {
                bool priorStatusSet = false;
                if (isShareLoc) {
                    confInfoForPrior[offV + p] = -1;
                }
                int confIdxPrior = off + p * classesNum;
                for (int c = 0; c < classesNum; ++c) {
                    float conf = confData[confIdxPrior + c];
                    if (conf > confidenceThreshold) {
                        const int idx = offH + c * confInfoLen;
                        reorderedConfData[idx + p] = conf;
                        mtx.lock();
                        reorderedConfDataIndices[idx + priorsNum]++;
                        reorderedConfDataIndices[idx + priorsNum + reorderedConfDataIndices[idx + priorsNum]] = p;
                        mtx.unlock();

                        if (!priorStatusSet && isShareLoc) {
                            confInfoForPrior[offV + p] = 1;
                        }
                    }
                }
            }
        });
        // topk
        parallel_for(classesNum, [&](size_t c) {
            // in:  conf_h info
            // out: buffer, detectionCount(k)
            if (c == static_cast<size_t>(backgroundClassId)) {  // Ignore background class
                return;
            }
            const int countIdx = offH + c * confInfoLen + priorsNum;
            const int count = reorderedConfDataIndices[countIdx];
            const int k = (topK == -1 ? count : (std::min)(topK, count));

            int* reorderedConfIndices = reorderedConfDataIndices + countIdx + 1;
            int* pbuffer = indicesBufData + off + c * priorsNum;
            const float* pconf = reorderedConfData + offH + c * confInfoLen;

            topk(reorderedConfIndices, pbuffer, pconf, count, k);
            detectionsData[n * classesNum + c] = k;
        });
    }
}

inline void DetectionOutput::confReorderAndFilterSparsityMX(const float* confData,
                                                            const float* ARMConfData,
                                                            float* reorderedConfData,
                                                            int* indicesData,
                                                            int* indicesBufData,
                                                            int* detectionsData) {
    for (int n = 0; n < imgNum; ++n) {
        const int off = n * priorsNum * classesNum;
        const int offV = n * priorsNum;  // vertical info

        std::mutex mtx;
        parallel_for(numPriorsActual[n], [&](size_t p) {
            bool isARMPrior = false;
            if (withAddBoxPred) {
                isARMPrior = ARMConfData[n * priorsNum * 2 + p * 2 + 1] < objScore;
            }
            bool priorStatusSet = false;
            if (isShareLoc) {
                confInfoForPrior[offV + p] = -1;
            }
            float maxConf = -1;
            int maxCIdx = 0;
            int confIdxPrior = off + p * classesNum;
            for (int c = 0; c < classesNum; ++c) {
                float conf = confData[confIdxPrior + c];
                if (withAddBoxPred && isARMPrior) {
                    conf = (c == backgroundClassId) ? 1.0f : 0.0f;
                }
                if (conf >= confidenceThreshold) {
                    int idx = off + c * confInfoLen;
                    reorderedConfData[idx + p] = conf;

                    // vertical info for isShareLoc(flag to decode for each prior)
                    if (!priorStatusSet && isShareLoc) {
                        confInfoForPrior[offV + p] = 1;  // 1 for decode
                    }
                    // vertical info for MXNet style(max conf for each prior)
                    if (c != 0) {
                        if (conf > maxConf) {
                            maxConf = conf;
                            maxCIdx = c;
                        }
                    }
                }
            }
            // MXNet statistic, indices and detectionCount is for each image
            if (maxCIdx > 0) {
                mtx.lock();
                indicesData[off + detectionsData[n * classesNum]] =
                    maxCIdx * priorsNum + p;  // de-refer to get prior and class id.
                detectionsData[n * classesNum]++;
                mtx.unlock();
            }
        });
        // topk
        // in:  indicesData, detection_count(filtered num)
        // out: buffer, detection_count(k)
        const int count = detectionsData[n * classesNum];
        const int k = (topK == -1 ? count : (std::min)(topK, count));

        const float* pconf = reorderedConfData + off;
        int* indices = indicesData + off;
        int* pbuffer = indicesBufData + off;
        topk(indices, pbuffer, pconf, count, k);
        detectionsData[n * classesNum] = k;
    }
}

// apply locData(offset) to priordata, generate decodedBox
inline void DetectionOutput::decodeBBoxes(const float* priorData,
                                          const float* locData,
                                          const float* varianceData,
                                          float* decodedBboxes,
                                          float* decodedBboxSizes,
                                          int* numPriorsActual,
                                          int n,
                                          const int& offs,
                                          const int& priorSize,
                                          bool decodeType,
                                          const int* confInfoH,
                                          const int* confInfoV) {
    int prNum = numPriorsActual[n];
    if (!decodeType) {
        prNum = priorsNum;
    }
    if (isSparsityWorthwhile && !isShareLoc && !decreaseClassId && confInfoH[priorsNum] == 0) {
        return;
    }
    parallel_for(prNum, [&](int p) {
        if (isSparsityWorthwhile && isShareLoc && confInfoV[p] == -1) {
            return;
        }
        float newXMin = 0.0f;
        float newYMin = 0.0f;
        float newXMax = 0.0f;
        float newYMax = 0.0f;

        float priorXMin = priorData[p * priorSize + 0 + offs];
        float priorYMin = priorData[p * priorSize + 1 + offs];
        float priorXMax = priorData[p * priorSize + 2 + offs];
        float priorYMax = priorData[p * priorSize + 3 + offs];

        float locXMin = locData[4 * p * locNumForClasses + 0];
        float locYMin = locData[4 * p * locNumForClasses + 1];
        float locXMax = locData[4 * p * locNumForClasses + 2];
        float locYMax = locData[4 * p * locNumForClasses + 3];

        if (!normalized) {
            priorXMin /= imgWidth;
            priorYMin /= imgHeight;
            priorXMax /= imgWidth;
            priorYMax /= imgHeight;
        }

        if (codeType == CodeType::CORNER) {
            if (varianceEncodedInTarget) {
                // variance is encoded in target, we simply need to add the offset predictions.
                newXMin = priorXMin + locXMin;
                newYMin = priorYMin + locYMin;
                newXMax = priorXMax + locXMax;
                newYMax = priorYMax + locYMax;
            } else {
                newXMin = priorXMin + varianceData[p * 4 + 0] * locXMin;
                newYMin = priorYMin + varianceData[p * 4 + 1] * locYMin;
                newXMax = priorXMax + varianceData[p * 4 + 2] * locXMax;
                newYMax = priorYMax + varianceData[p * 4 + 3] * locYMax;
            }
        } else if (codeType == CodeType::CENTER_SIZE) {
            float priorWidth = priorXMax - priorXMin;
            float priorHeight = priorYMax - priorYMin;
            float priorCenterX = (priorXMin + priorXMax) / 2.0f;
            float priorCenterY = (priorYMin + priorYMax) / 2.0f;

            float decodeBboxCenterX, decodeBboxCenterY;
            float decodeBboxWidth, decodeBboxHeight;

            if (varianceEncodedInTarget) {
                // variance is encoded in target, we simply need to restore the offset predictions.
                decodeBboxCenterX = locXMin * priorWidth + priorCenterX;
                decodeBboxCenterY = locYMin * priorHeight + priorCenterY;
                decodeBboxWidth = std::exp(locXMax) * priorWidth;
                decodeBboxHeight = std::exp(locYMax) * priorHeight;
            } else {
                // variance is encoded in bbox, we need to scale the offset accordingly.
                decodeBboxCenterX = varianceData[p * 4 + 0] * locXMin * priorWidth + priorCenterX;
                decodeBboxCenterY = varianceData[p * 4 + 1] * locYMin * priorHeight + priorCenterY;
                decodeBboxWidth = std::exp(varianceData[p * 4 + 2] * locXMax) * priorWidth;
                decodeBboxHeight = std::exp(varianceData[p * 4 + 3] * locYMax) * priorHeight;
            }

            newXMin = decodeBboxCenterX - decodeBboxWidth / 2.0f;
            newYMin = decodeBboxCenterY - decodeBboxHeight / 2.0f;
            newXMax = decodeBboxCenterX + decodeBboxWidth / 2.0f;
            newYMax = decodeBboxCenterY + decodeBboxHeight / 2.0f;
        }

        if (clipBeforeNMS) {
            newXMin = (std::max)(0.0f, (std::min)(1.0f, newXMin));
            newYMin = (std::max)(0.0f, (std::min)(1.0f, newYMin));
            newXMax = (std::max)(0.0f, (std::min)(1.0f, newXMax));
            newYMax = (std::max)(0.0f, (std::min)(1.0f, newYMax));
        }

        decodedBboxes[p * 4 + 0] = newXMin;
        decodedBboxes[p * 4 + 1] = newYMin;
        decodedBboxes[p * 4 + 2] = newXMax;
        decodedBboxes[p * 4 + 3] = newYMax;

        decodedBboxSizes[p] = (newXMax - newXMin) * (newYMax - newYMin);
    });
}

inline void DetectionOutput::topk(const int* indicesIn, int* indicesOut, const float* conf, int n, int k) {
    std::partial_sort_copy(indicesIn, indicesIn + n, indicesOut, indicesOut + k, ConfidenceComparatorDO(conf));
}

static inline float JaccardOverlap(const float* decodedBbox, const float* bboxSizes, const int idx1, const int idx2) {
    const float xmin1 = decodedBbox[idx1 * 4 + 0];
    const float ymin1 = decodedBbox[idx1 * 4 + 1];
    const float xmax1 = decodedBbox[idx1 * 4 + 2];
    const float ymax1 = decodedBbox[idx1 * 4 + 3];

    const float xmin2 = decodedBbox[idx2 * 4 + 0];
    const float ymin2 = decodedBbox[idx2 * 4 + 1];
    const float xmax2 = decodedBbox[idx2 * 4 + 2];
    const float ymax2 = decodedBbox[idx2 * 4 + 3];

    if (xmin2 > xmax1 || xmax2 < xmin1 || ymin2 > ymax1 || ymax2 < ymin1) {
        return 0.0f;
    }

    float intersectXMin = (std::max)(xmin1, xmin2);
    float intersectYMin = (std::max)(ymin1, ymin2);
    float intersectXMax = (std::min)(xmax1, xmax2);
    float intersectYMax = (std::min)(ymax1, ymax2);

    float intersectWidth = intersectXMax - intersectXMin;
    float intersectHeight = intersectYMax - intersectYMin;

    if (intersectWidth <= 0 || intersectHeight <= 0) {
        return 0.0f;
    }

    float intersectSize = intersectWidth * intersectHeight;
    float bbox1Size = bboxSizes[idx1];
    float bbox2Size = bboxSizes[idx2];

    return intersectSize / (bbox1Size + bbox2Size - intersectSize);
}

inline void DetectionOutput::NMSCF(int* indicesIn,
                                   int& detections,
                                   int* indicesOut,
                                   const float* bboxes,
                                   const float* boxSizes) {
    // nms for this class
    int countIn = detections;
    detections = 0;
    for (int i = 0; i < countIn; ++i) {
        const int prior = indicesIn[i];

        bool keep = true;
        for (int k = 0; k < detections; ++k) {
            const int keptPrior = indicesOut[k];
            float overlap = JaccardOverlap(bboxes, boxSizes, prior, keptPrior);
            if (overlap > NMSThreshold) {
                keep = false;
                break;
            }
        }
        if (keep) {
            indicesOut[detections] = prior;
            detections++;
        }
    }
}

inline void DetectionOutput::NMSMX(int* indicesIn,
                                   int* detections,
                                   int* indicesOut,
                                   const float* bboxes,
                                   const float* sizes) {
    // Input is candidate for image, output is candidate for each class within image
    int countIn = detections[0];
    detections[0] = 0;

    for (int i = 0; i < countIn; ++i) {
        const int idx = indicesIn[i];
        const int cls = idx / priorsNum;
        const int prior = idx % priorsNum;

        // nms within this class
        int& ndetection = detections[cls];
        int* pindices = indicesOut + cls * priorsNum;

        bool keep = true;
        for (int k = 0; k < ndetection; ++k) {
            const int keptPrior = pindices[k];
            float overlap = 0.0f;
            if (isShareLoc) {
                overlap = JaccardOverlap(bboxes, sizes, prior, keptPrior);
            } else {
                overlap = JaccardOverlap(bboxes, sizes, cls * priorsNum + prior, cls * priorsNum + keptPrior);
            }
            if (overlap > NMSThreshold) {
                keep = false;
                break;
            }
        }

        if (keep) {
            pindices[ndetection++] = prior;
        }
    }
}

inline void DetectionOutput::generateOutput(float* reorderedConfData,
                                            int* indicesData,
                                            int* detectionsData,
                                            float* decodedBboxesData,
                                            float* dstData) {
    const auto& outDims = getChildEdgeAt(0)->getMemory().getStaticDims();
    const int numResults = outDims[2];
    const int DETECTION_SIZE = outDims[3];
    if (DETECTION_SIZE != 7) {
        THROW_CPU_NODE_ERR("has unsupported output layout.");
    }

    int dstDataSize = 0;
    if (keepTopK > 0) {
        dstDataSize = imgNum * keepTopK * DETECTION_SIZE * sizeof(float);
    } else if (topK > 0) {
        dstDataSize = imgNum * topK * classesNum * DETECTION_SIZE * sizeof(float);
    } else {
        dstDataSize = imgNum * classesNum * priorsNum * DETECTION_SIZE * sizeof(float);
    }

    if (static_cast<size_t>(dstDataSize) > getChildEdgeAt(0)->getMemory().getSize()) {
        THROW_CPU_NODE_ERR("has insufficient output buffer size.");
    }
    memset(dstData, 0, dstDataSize);

    // set final detection result to output blob
    int count = 0;
    for (int n = 0; n < imgNum; ++n) {
        const float* pconf = reorderedConfData + n * confInfoLen * classesNum;
        const float* pboxes = decodedBboxesData + n * priorsNum * 4 * locNumForClasses;
        const int* pindices = indicesData + n * classesNum * priorsNum;

        for (int c = 0; c < classesNum; ++c) {
            for (int i = 0; i < detectionsData[n * classesNum + c]; ++i) {
                int prIdx = pindices[c * priorsNum + i];

                dstData[count * DETECTION_SIZE + 0] = static_cast<float>(n);
                dstData[count * DETECTION_SIZE + 1] = static_cast<float>(decreaseClassId ? c - 1 : c);
                dstData[count * DETECTION_SIZE + 2] = pconf[c * confInfoLen + prIdx];

                float xmin = isShareLoc ? pboxes[prIdx * 4 + 0] : pboxes[c * 4 * priorsNum + prIdx * 4 + 0];
                float ymin = isShareLoc ? pboxes[prIdx * 4 + 1] : pboxes[c * 4 * priorsNum + prIdx * 4 + 1];
                float xmax = isShareLoc ? pboxes[prIdx * 4 + 2] : pboxes[c * 4 * priorsNum + prIdx * 4 + 2];
                float ymax = isShareLoc ? pboxes[prIdx * 4 + 3] : pboxes[c * 4 * priorsNum + prIdx * 4 + 3];

                if (clipAfterNMS) {
                    xmin = (std::max)(0.0f, (std::min)(1.0f, xmin));
                    ymin = (std::max)(0.0f, (std::min)(1.0f, ymin));
                    xmax = (std::max)(0.0f, (std::min)(1.0f, xmax));
                    ymax = (std::max)(0.0f, (std::min)(1.0f, ymax));
                }

                dstData[count * DETECTION_SIZE + 3] = xmin;
                dstData[count * DETECTION_SIZE + 4] = ymin;
                dstData[count * DETECTION_SIZE + 5] = xmax;
                dstData[count * DETECTION_SIZE + 6] = ymax;

                ++count;
            }
        }
    }

    if (count < numResults) {
        // marker at end of boxes list
        dstData[count * DETECTION_SIZE + 0] = -1;
    }
}

bool DetectionOutput::created() const {
    return getType() == Type::DetectionOutput;
}

}  // namespace ov::intel_cpu::node
