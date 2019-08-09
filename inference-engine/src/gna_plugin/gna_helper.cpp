// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
//  gna_helper.cpp : various GNA-related utility functions
//

#include "lstm.hpp"

#define USING_GCC
#define PROFILE

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>
#include <sstream>
#include <string>
#include "gna-api.h"

#ifndef WIN32
#include <profiler.h>

void clearTimeB(timeb & tb) {
    tb.time = 0;
    tb.dstflag = 0;
    tb.millitm = 0;
    tb.timezone = 0;
}
//  dummy definitions to work around issue with Linux userspace library
void profilerTscStart(intel_gna_profiler_tsc *p) {
    if (nullptr == p) return;
    p->stop = 0;
    p->start = 0;
}
void profilerTscStop(intel_gna_profiler_tsc *p) {
    if (nullptr == p) return;
    p->stop = 0;
    p->start = 0;
}
void profilerTscStartAccumulate(intel_gna_profiler_tsc *p) {
    if (nullptr == p) return;
    p->stop = 0;
    p->start = 0;
}
void profilerTscStopAccumulate(intel_gna_profiler_tsc *p) {
    if (nullptr == p) return;
    p->stop = 0;
}
void profilerRtcClear(intel_gna_profiler_rtc *p) {
    if (nullptr == p) return;
    clearTimeB(p->passed);
    clearTimeB(p->start);
    clearTimeB(p->stop);
}
void profilerRtcStart(intel_gna_profiler_rtc *p) {
    if (nullptr == p) return;
    clearTimeB(p->passed);
    clearTimeB(p->stop);
    ftime(&p->start);
}

void profilerRtcStop(intel_gna_profiler_rtc *p) {
    if (nullptr == p) return;
    ftime(&p->stop);
    /*if ((p->stop.tv_nsec - p->start.tv_nsec)<0) {
        p->passed.tv_sec = p->stop.tv_sec - p->start.tv_sec - 1;
        p->passed.tv_nsec = 1000000000 + p->stop.tv_nsec - p->start.tv_nsec;
    }
    else {
        p->passed.tv_sec = p->stop.tv_sec - p->start.tv_sec;
        p->passed.tv_nsec = p->stop.tv_nsec - p->start.tv_nsec;
    }*/
}
void profilerRtcStartAccumulate(intel_gna_profiler_rtc *p) {
    if (nullptr == p) return;
    clearTimeB(p->stop);
//    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &p->start);
}
void profilerRtcStopAccumulate(intel_gna_profiler_rtc *p) {
    timespec diff;
    if (nullptr == p) return;
//    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &p->stop);
//    if ((p->stop.tv_nsec - p->start.tv_nsec)<0) {
//        diff.tv_sec = p->stop.tv_sec - p->start.tv_sec - 1;
//        diff.tv_nsec = 1000000000 + p->stop.tv_nsec - p->start.tv_nsec;
//    }
//    else {
//        diff.tv_sec = p->stop.tv_sec - p->start.tv_sec;
//        diff.tv_nsec = p->stop.tv_nsec - p->start.tv_nsec;
//    }
//    p->passed.tv_sec += diff.tv_sec;
//    p->passed.tv_nsec += diff.tv_nsec;
//    if (p->passed.tv_nsec > 1000000000) {
//        p->passed.tv_sec++;
//        p->passed.tv_nsec -= 1000000000;
//    }
}

#endif
void PrintMatrixInt16(const char *ptr_name, int16_t *ptr_matrix, int num_rows, int num_cols, int lda, float scale) {
    printf("%s:  %dx%d lda %d\n", ptr_name, num_rows, num_cols, lda);
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            printf("[%d,%d]: %e\n", i, j, *(ptr_matrix + i*lda + j) / scale);
        }
    }
}

void PrintMatrixInt32(char *ptr_name, int32_t *ptr_matrix, int num_rows, int num_cols, int lda, float scale) {
    printf("%s:  %dx%d lda %d\n", ptr_name, num_rows, num_cols, lda);
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            printf("[%d,%d]: %e\n", i, j, *(ptr_matrix + i*lda + j) / scale);
        }
    }
}

void PrintMatrixFloat32(char *ptr_name, float *ptr_matrix, int num_rows, int num_cols, int lda) {
#if (_WIN32 || _WIN64) && (_MSC_VER < 1900)
    _set_output_format(_TWO_DIGIT_EXPONENT);
#endif
    printf("%s:  %dx%d lda %d\n", ptr_name, num_rows, num_cols, lda);
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            printf("[%d,%d]: %e\n", i, j, *(ptr_matrix + i*lda + j));
        }
    }
}

void PrintGnaNetwork(intel_nnet_type_t *ptr_nnet) {
    PrintMatrixInt16("input", reinterpret_cast<int16_t*>(ptr_nnet->pLayers[0].pInputs),
                     ptr_nnet->pLayers[0].nInputRows, ptr_nnet->pLayers[0].nInputColumns, ptr_nnet->pLayers[0].nInputColumns, 1.0);
    for (uint32_t i = 0; i < ptr_nnet->nLayers; i++) {
        char name[256];
        snprintf(name, sizeof(name), "output %d", i);
        if (ptr_nnet->pLayers[i].nBytesPerOutput == 2) {
            PrintMatrixInt16(name, reinterpret_cast<int16_t*>(ptr_nnet->pLayers[i].pOutputs),
                             ptr_nnet->pLayers[i].nOutputRows, ptr_nnet->pLayers[i].nOutputColumns, ptr_nnet->pLayers[i].nOutputColumns, 1.0);
        } else {
            PrintMatrixInt32(name, reinterpret_cast<int32_t*>(ptr_nnet->pLayers[i].pOutputs),
                             ptr_nnet->pLayers[i].nOutputRows, ptr_nnet->pLayers[i].nOutputColumns, ptr_nnet->pLayers[i].nOutputColumns, 1.0);
        }
    }
}

typedef struct {
    std::string sName;
    std::string sType;  //  if wgt/bias/filt/pwl is writeable, then do not write it to file
    void *pAddress;
    uint32_t nBytes;
} intel_memory_region_t;

void AddBufferEntry(std::vector<intel_memory_region_t> &vBuffer,
                    const std::string &sName,
                    const std::string &sType,
                    void *pBuffer,
                    uint32_t nBytes) {
    if (pBuffer != NULL) {
        intel_memory_region_t region;
        region.sName = sName;
        region.sType = sType;
        region.pAddress = pBuffer;
        region.nBytes = nBytes;
        vBuffer.push_back(region);
    }
}

std::string BufferNameFromAddress(std::vector<intel_memory_region_t> &vBuffer, void *pBuffer) {
    std::stringstream ss;
    std::string sAddr, sName;
    void *pParentBuffer = pBuffer;
    bool found = false;
    bool found_persistent = false;
    bool found_output = false;
    for (uint32_t i = 0; i < vBuffer.size(); i++) {
        uint8_t *pBufferStart = reinterpret_cast<uint8_t *>(pBuffer);
        uint8_t *pEntryBufferStart = reinterpret_cast<uint8_t *>(vBuffer.at(i).pAddress);
        uint8_t *pEntryBufferEnd = reinterpret_cast<uint8_t *>(vBuffer.at(i).pAddress) + vBuffer.at(i).nBytes;
        if ((pBufferStart >= pEntryBufferStart) && (pBufferStart < pEntryBufferEnd)) {
            found = true;
            if (pBufferStart > pEntryBufferStart) {
                pParentBuffer = pEntryBufferStart;
            }
            if ((vBuffer.at(i).sType.compare("pOutputs") == 0)
                || (vBuffer.at(i).sType.compare("pOutputsIntermediate") == 0)) {
                found_output = true;
            } else if (vBuffer.at(i).sType.compare("pWeights") == 0) {
                sName = "wgt_";
                found_persistent = true;
            } else if (vBuffer.at(i).sType.compare("pBiases") == 0) {
                sName = "bias_";
                found_persistent = true;
            } else if (vBuffer.at(i).sType.compare("pSegments") == 0) {
                sName = "pwl_";
                found_persistent = true;
            }
        }
    }
    if (found) {
        if ((found_output) || (!found_persistent)) {
            sName = "buf_";
        }
        ss << (int64_t) pParentBuffer;
        sAddr = ss.str();
        sName.append(sAddr);
    } else {
        fprintf(stderr, "Error:  buffer address does not exist in BufferNameFromAddress!\n");
        exit(EXIT_FAILURE);
    }
    return (sName);
}

uint32_t BufferOffsetFromAddress(std::vector<intel_memory_region_t> &vBuffer, void *pBuffer) {
    uint32_t nOffsetBytes = 0;
    for (uint32_t i = 0; i < vBuffer.size(); i++) {
        uint8_t *pBufferStart = reinterpret_cast<uint8_t *>(pBuffer);
        uint8_t *pEntryBufferStart = reinterpret_cast<uint8_t *>(vBuffer.at(i).pAddress);
        uint8_t *pEntryBufferEnd = reinterpret_cast<uint8_t *>(vBuffer.at(i).pAddress) + vBuffer.at(i).nBytes;
        if ((pBufferStart >= pEntryBufferStart) && (pBufferStart < pEntryBufferEnd)) {
            if (pBufferStart > pEntryBufferStart) {
                nOffsetBytes = (uint32_t) (pBufferStart - pEntryBufferStart);
            }
        }
    }
    return (nOffsetBytes);
}

std::string LayerName(intel_nnet_layer_t *pLayer) {
    intel_layer_kind_t nKind = pLayer->nLayerKind;
    std::string sKind;
    if (nKind == INTEL_AFFINE) {
        sKind = "affine";
    } else if (nKind == INTEL_AFFINE_DIAGONAL) {
        sKind = "diagonal";
    } else if (nKind == INTEL_INTERLEAVE) {
        sKind = "interleave";
    } else if (nKind == INTEL_DEINTERLEAVE) {
        sKind = "deinterleave";
    } else {
        fprintf(stderr, "Error:  nLayerKind not supported in LayerName()!\n");
        exit(EXIT_FAILURE);
    }
    return (sKind);
}

uint32_t NumInputs(intel_nnet_layer_t *pLayer) {
    intel_layer_kind_t nKind = pLayer->nLayerKind;
    uint32_t nInputs;
    if ((nKind == INTEL_AFFINE) || (nKind == INTEL_AFFINE_DIAGONAL)) {
        nInputs = pLayer->nInputRows;
    } else if (nKind == INTEL_INTERLEAVE) {
        nInputs = pLayer->nInputColumns;
    } else if (nKind == INTEL_DEINTERLEAVE) {
        nInputs = pLayer->nInputRows;
    } else {
        fprintf(stderr, "Error:  nLayerKind not supported in NumInputs()!\n");
        exit(EXIT_FAILURE);
    }
    return (nInputs);
}

uint32_t NumOutputs(intel_nnet_layer_t *pLayer) {
    intel_layer_kind_t nKind = pLayer->nLayerKind;
    uint32_t nOutputs;
    if ((nKind == INTEL_AFFINE) || (nKind == INTEL_AFFINE_DIAGONAL)) {
        nOutputs = pLayer->nOutputRows;
    } else if (nKind == INTEL_INTERLEAVE) {
        nOutputs = pLayer->nOutputRows;
    } else if (nKind == INTEL_DEINTERLEAVE) {
        nOutputs = pLayer->nOutputColumns;
    } else {
        fprintf(stderr, "Error:  nLayerKind not supported in NumInputs()!\n");
        exit(EXIT_FAILURE);
    }
    return (nOutputs);
}

uint32_t NumGroupSize(intel_nnet_layer_t *pLayer) {
    intel_layer_kind_t nKind = pLayer->nLayerKind;
    uint32_t nGroupSize;
    if ((nKind == INTEL_AFFINE) || (nKind == INTEL_AFFINE_DIAGONAL)) {
        nGroupSize = pLayer->nOutputColumns;
    } else if (nKind == INTEL_INTERLEAVE) {
        nGroupSize = pLayer->nOutputColumns;
    } else if (nKind == INTEL_DEINTERLEAVE) {
        nGroupSize = pLayer->nOutputRows;
    } else {
        fprintf(stderr, "Error:  nLayerKind not supported in NumGroupSize()!\n");
        exit(EXIT_FAILURE);
    }
    return (nGroupSize);
}
