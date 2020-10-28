// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <map>

#include "details/ie_irelease.hpp"
#include "ie_version.hpp"
#include "ie_common.h"
#include "ie_blob.h"

namespace InferenceEngine {

/**
 * @class IShapeInferImpl
 * @brief This class provides interface for the implementation with the custom execution code
 */
class IShapeInferImpl {
public:
    /**
     * @brief A shared pointer to a IShapeInferImpl object
     */
    using Ptr = std::shared_ptr<IShapeInferImpl>;

    virtual ~IShapeInferImpl() = default;

    /**
     * @brief check that reshape can be applied, that parameters and shapes are valid
     */
    virtual StatusCode inferShapes(const std::vector<Blob::CPtr>& /*inBlobs*/,
                                   const std::map<std::string, std::string>& /*params*/,
                                   const std::map<std::string, Blob::Ptr>& /*blobs*/,
                                   std::vector<SizeVector>& /*outShapes*/, ResponseDesc* /*resp*/) noexcept {
        return NOT_IMPLEMENTED;
    }  // For backward-compatibility
};

/**
 * @class IShapeInferExtension
 * @brief This class is the reader extension interface to provide implementation for shape propagation
 */
class IShapeInferExtension : public InferenceEngine::details::IRelease {
public:
    /**
     * @brief Gets extension version information and stores in versionInfo
     * @param versionInfo Pointer to version info, will be set by plugin
     */
    virtual void GetVersion(const InferenceEngine::Version*& versionInfo) const noexcept = 0;

    /**
     * @brief Cleans resources up
     */
    virtual void Unload() noexcept = 0;

    /**
     * The method will be removed in 2021.1 release.
     * @brief Fills passed array with types of layers which shape infer implementations are included in the extension
     *
     * @param types Array to store the layer types
     * @param size Size of the layer types array
     * @param resp Response descriptor
     * @return Status code
     */
    virtual StatusCode getShapeInferTypes(char**& types, unsigned int& size, ResponseDesc* resp) noexcept = 0;

    /**
     * @brief Gets shape propagation implementation for the given string-type of CNNLayer
     *
     * @param impl the vector with implementations which is ordered by priority
     * @param type A type of CNNLayer
     * @param resp response descriptor
     * @return status code
     */
    virtual StatusCode getShapeInferImpl(IShapeInferImpl::Ptr& impl, const char* type, ResponseDesc* resp) noexcept = 0;
};

/**
 * This API will be removed in 2021.1 release.
 * @brief A shared pointer to a IShapeInferExtension interface
 */
using IShapeInferExtensionPtr = std::shared_ptr<IShapeInferExtension>;

}  // namespace InferenceEngine
