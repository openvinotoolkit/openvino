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

#ifndef __HDDL_INFER_DATA__
#define __HDDL_INFER_DATA__

#include <HddlCommon.h>
#include <HddlBlob.h>

#include <memory>
#include <vector>
#include <functional>

namespace hddl {
class HddlInferDataImpl;

class HDDL_EXPORT_API HddlInferData
{
public:
    typedef std::shared_ptr<HddlInferData> Ptr;

    static HddlInferData::Ptr makeInferData(HddlBlob* in, HddlBlob* out, HddlAuxInfoType auxInfoType = AUX_INFO_NONE);
    ~HddlInferData();

    HddlInferData(const HddlInferData&) = delete;
    HddlInferData& operator=(const HddlInferData&) = delete;

    void           setUserData(const void* data);
    void           setCallback(std::function<void(HddlInferData::Ptr, void*)> callback);

    HddlBlob*      getInputBlob();
    HddlBlob*      getOutputBlob();
    const HddlAuxBlob::Ptr   getAuxInfoBlob();

    HddlStatusCode getInferStatusCode();
    HddlTaskHandle getTaskHandle();

    void*          getUserData();

private:
    friend class HddlClientImpl;
    friend class HddlTask;

    HddlInferData(HddlBlob* in, HddlBlob* out, HddlAuxInfoType auxInfoType);

    HddlInferDataImpl* m_impl;
};
} // namespace hddl

#endif
