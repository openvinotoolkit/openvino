// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <sched.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <vector>

namespace MKLDNNPlugin {
namespace cpu {

#ifndef __APPLE__

struct Processor {
    unsigned processor;
    unsigned physicalId;
    unsigned cpuCores;

    Processor();
};

class CpuInfoInterface {
public:
    virtual ~CpuInfoInterface() {}

    virtual const char *getFirstLine() = 0;

    virtual const char *getNextLine() = 0;
};

class CpuInfo : public CpuInfoInterface {
public:
    CpuInfo();

    virtual ~CpuInfo();

    virtual const char *getFirstLine();

    virtual const char *getNextLine();

private:
    const char *fileContentBegin;
    const char *fileContentEnd;
    const char *currentLine;

    void loadContentFromFile(const char *fileName);

    void loadContent(const char *content);

    void parseLines(char *content);
};

class CollectionInterface {
public:
    virtual ~CollectionInterface() {}
    virtual unsigned getTotalNumberOfSockets() = 0;
};

class Collection : public CollectionInterface {
public:
    explicit Collection(CpuInfoInterface *cpuInfo);

    virtual unsigned getTotalNumberOfSockets();
    virtual unsigned getTotalNumberOfCpuCores();
    virtual unsigned getNumberOfProcessors();

private:
    CpuInfoInterface &cpuInfo;
    unsigned totalNumberOfSockets;
    unsigned totalNumberOfCpuCores;
    std::vector<Processor> processors;
    Processor *currentProcessor;

    Collection(const Collection &collection);

    Collection &operator=(const Collection &collection);

    void parseCpuInfo();

    void parseCpuInfoLine(const char *cpuInfoLine);

    void parseValue(const char *fieldName, const char *valueString);

    void appendNewProcessor();

    bool beginsWith(const char *lineBuffer, const char *text) const;

    unsigned parseInteger(const char *text) const;

    void collectBasicCpuInformation();

    void updateCpuInformation(const Processor &processor,
                              unsigned numberOfUniquePhysicalId);
};
#endif  // #ifndef __APPLE__
}  // namespace cpu
}  // namespace MKLDNNPlugin
