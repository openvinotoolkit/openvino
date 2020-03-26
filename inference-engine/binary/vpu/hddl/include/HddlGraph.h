//
// Copyright © 2017-2018 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version October 2018). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#ifndef __HDDL_API_HDDL_GRAPH_H__
#define __HDDL_API_HDDL_GRAPH_H__

#include <string>
#include <memory>

#include "HddlCommon.h"

namespace hddl {

class HddlGraphImpl;

class HDDL_EXPORT_API HddlGraph
{
public:
    typedef std::shared_ptr<HddlGraph> Ptr;
    ~HddlGraph();

    HddlGraph(const HddlGraph&) = delete;
    HddlGraph& operator=(const HddlGraph&) = delete;

    std::string getName();
    std::string getPath();

    const void* getData();
    size_t      getDataSize();

    size_t      getInputSize();
    size_t      getOutputSize();
    size_t      getAuxSize(HddlAuxInfoType infoType);

private:
    HddlGraph();

    friend class  HddlClientImpl;
    HddlGraphImpl *m_impl;
};

}
#endif
