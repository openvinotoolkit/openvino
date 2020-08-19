/*
 INTEL CONFIDENTIAL
 Copyright 2017 Intel Corporation.

 The source code contained or described herein and all documents related
 to the source code ("Material") are owned by Intel Corporation or its suppliers
 or licensors. Title to the Material remains with Intel Corporation or its suppliers
 and licensors. The Material may contain trade secrets and proprietary
 and confidential information of Intel Corporation and its suppliers and licensors,
 and is protected by worldwide copyright and trade secret laws and treaty provisions.
 No part of the Material may be used, copied, reproduced, modified, published,
 uploaded, posted, transmitted, distributed, or disclosed in any way without Intel's
 prior express written permission.

 No license under any patent, copyright, trade secret or other intellectual
 property right is granted to or conferred upon you by disclosure or delivery
 of the Materials, either expressly, by implication, inducement, estoppel
 or otherwise. Any license under such intellectual property rights must
 be express and approved by Intel in writing.

 Unless otherwise agreed by Intel in writing, you may not remove or alter this notice
 or any other notice embedded in Materials by Intel or Intel's suppliers or licensors
 in any way.
*/

#pragma once

#include "Logger.h"
#include "gna2-common-impl.h"

#include <stdexcept>
#include <string>
#include <stdexcept>
#include <unordered_map>

namespace GNA
{

/**
 * Custom exception with device open error support
 */
class GnaException : public std::runtime_error
{
public:
    explicit GnaException(Gna2Status status) :
        std::runtime_error{ Logger::StatusToString(StatusMap.at(status)) },
        Status{ status },
        LegacyStatus { StatusMap.at(status) }
    {}

    virtual inline Gna2Status GetStatus() const
    {
        return Status;
    }

    virtual inline gna_status_t GetLegacyStatus() const
    {
        Log->Error(LegacyStatus);
        return LegacyStatus;
    }

    inline void Print() const
    {
        Log->Error(LegacyStatus, " GnaException");
    }

    virtual ~GnaException() {};

protected:
    Gna2Status Status;
    gna_status_t LegacyStatus;

};

/**
 * Custom exception for tensor build errors
 */
class GnaTensorException : public GnaException
{
public:

    GnaTensorException(const GnaException& e, gna_tensor_dim dimension) :
        GnaException{e},
        Dimension{dimension}
    {}

    virtual inline gna_status_t GetLegacyStatus() const override
    {
        Log->Error(LegacyStatus, " Tensor build failed on dimension: %u", Dimension); // TODO:3: tensor dims names
        return LegacyStatus;
    }

    inline void Print() const
    {
        Log->Error(LegacyStatus, " GnaTensorException: Tensor build failed on dimension: %u", Dimension); // TODO:3: tensor dims names
    }

    virtual ~GnaTensorException() = default;

protected:
    gna_tensor_dim Dimension;
};


/**
 * Custom exception for model build errors
 */
class GnaModelException : public GnaTensorException
{
public:

    GnaModelException(const GnaException& e, uint32_t layerId) :
        GnaTensorException{e, GNA_DIM_S},
        LayerId{layerId}
    {}

    GnaModelException(const GnaTensorException& e, uint32_t layerId) :
        GnaTensorException{e},
        LayerId{layerId}
    {}

    virtual inline gna_status_t GetLegacyStatus() const override
    {
        Log->Error(LegacyStatus, " Model build failed on layer: %d", LayerId);
        return LegacyStatus;
    }

    inline void Print() const
    {
        Log->Error(LegacyStatus, " GnaModelException: Model build failed on layer: %d", LayerId);
    }

    virtual ~GnaModelException() = default;

protected:
    uint32_t LayerId;
};

}
