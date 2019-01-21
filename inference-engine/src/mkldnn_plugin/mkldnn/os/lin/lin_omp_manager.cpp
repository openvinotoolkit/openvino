// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lin_omp_manager.h"
#include <fstream>
#include <set>
#include <string>
#include <vector>
#include <iostream>

namespace MKLDNNPlugin {
namespace cpu {

#ifndef __APPLE__

Processor::Processor() {
    processor = 0;
    physicalId = 0;
    cpuCores = 0;
}

CpuInfo::CpuInfo() {
    loadContentFromFile("/proc/cpuinfo");
}

void CpuInfo::loadContentFromFile(const char *fileName) {
    std::ifstream file(fileName);
    std::string content(
            (std::istreambuf_iterator<char>(file)),
            (std::istreambuf_iterator<char>()));

    loadContent(content.c_str());
}

void CpuInfo::loadContent(const char *content) {
    size_t contentLength = strlen(content);
    char *contentCopy = new char[contentLength + 1];
    snprintf(contentCopy, contentLength + 1, "%s", content);

    parseLines(contentCopy);

    fileContentBegin = contentCopy;
    fileContentEnd = &contentCopy[contentLength];
    currentLine = NULL;
}

CpuInfo::~CpuInfo() {
    delete[] fileContentBegin;
}

void CpuInfo::parseLines(char *content) {
    for (; *content; content++) {
        if (*content == '\n') {
            *content = '\0';
        }
    }
}

const char *CpuInfo::getFirstLine() {
    currentLine = fileContentBegin < fileContentEnd ? fileContentBegin : NULL;
    return getNextLine();
}

const char *CpuInfo::getNextLine() {
    if (!currentLine) {
        return NULL;
    }

    const char *savedCurrentLine = currentLine;
    while (*(currentLine++)) {
    }

    if (currentLine >= fileContentEnd) {
        currentLine = NULL;
    }

    return savedCurrentLine;
}

Collection::Collection(CpuInfoInterface *cpuInfo) : cpuInfo(*cpuInfo) {
    totalNumberOfSockets = 0;
    totalNumberOfCpuCores = 0;
    currentProcessor = NULL;

    processors.reserve(96);

    parseCpuInfo();
    collectBasicCpuInformation();
}

unsigned Collection::getTotalNumberOfSockets() {
    return totalNumberOfSockets;
}

unsigned Collection::getTotalNumberOfCpuCores() {
    return totalNumberOfCpuCores;
}

unsigned Collection::getNumberOfProcessors() {
    return processors.size();
}

void Collection::parseCpuInfo() {
    const char *cpuInfoLine = cpuInfo.getFirstLine();
    for (; cpuInfoLine; cpuInfoLine = cpuInfo.getNextLine()) {
        parseCpuInfoLine(cpuInfoLine);
    }
}

void Collection::parseCpuInfoLine(const char *cpuInfoLine) {
    int delimiterPosition = strcspn(cpuInfoLine, ":");

    if (cpuInfoLine[delimiterPosition] == '\0') {
        currentProcessor = NULL;
    } else {
        parseValue(cpuInfoLine, &cpuInfoLine[delimiterPosition + 2]);
    }
}

void Collection::parseValue(const char *fieldName, const char *valueString) {
    if (!currentProcessor) {
        appendNewProcessor();
    }

    if (beginsWith(fieldName, "processor")) {
        currentProcessor->processor = parseInteger(valueString);
    }

    if (beginsWith(fieldName, "physical id")) {
        currentProcessor->physicalId = parseInteger(valueString);
    }

    if (beginsWith(fieldName, "cpu cores")) {
        currentProcessor->cpuCores = parseInteger(valueString);
    }
}

void Collection::appendNewProcessor() {
    processors.push_back(Processor());
    currentProcessor = &processors.back();
}

bool Collection::beginsWith(const char *lineBuffer, const char *text) const {
    while (*text) {
        if (*(lineBuffer++) != *(text++)) {
            return false;
        }
    }

    return true;
}

unsigned Collection::parseInteger(const char *text) const {
    return atol(text);
}

void Collection::collectBasicCpuInformation() {
    std::set<unsigned> uniquePhysicalId;
    std::vector<Processor>::iterator processor = processors.begin();
    for (; processor != processors.end(); processor++) {
        uniquePhysicalId.insert(processor->physicalId);
        updateCpuInformation(*processor, uniquePhysicalId.size());
    }
}

void Collection::updateCpuInformation(const Processor &processor,
                                      unsigned numberOfUniquePhysicalId) {
    if (totalNumberOfSockets == numberOfUniquePhysicalId) {
        return;
    }

    totalNumberOfSockets = numberOfUniquePhysicalId;
    totalNumberOfCpuCores += processor.cpuCores;
}

#include <sched.h>

int getNumberOfCPUSockets() {
    static CpuInfo cpuInfo;
    static Collection collection(&cpuInfo);
    return collection.getTotalNumberOfSockets();
}

int getNumberOfCPUCores() {
    static CpuInfo cpuInfo;
    static Collection collection(&cpuInfo);
    unsigned numberOfProcessors = collection.getNumberOfProcessors();
    unsigned totalNumberOfCpuCores = collection.getTotalNumberOfCpuCores();

    cpu_set_t usedCoreSet, currentCoreSet, currentCpuSet;
    CPU_ZERO(&currentCpuSet);
    CPU_ZERO(&usedCoreSet);
    CPU_ZERO(&currentCoreSet);

    sched_getaffinity(0, sizeof(currentCpuSet), &currentCpuSet);

    for (int processorId = 0; processorId < numberOfProcessors; processorId++) {
        if (CPU_ISSET(processorId, &currentCpuSet)) {
            unsigned coreId = processorId % totalNumberOfCpuCores;
            if (!CPU_ISSET(coreId, &usedCoreSet)) {
                CPU_SET(coreId, &usedCoreSet);
                CPU_SET(processorId, &currentCoreSet);
            }
        }
    }
    return CPU_COUNT(&currentCoreSet);
}

#endif  // #ifndef APPLE
}  // namespace cpu
}  // namespace MKLDNNPlugin
