// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "frontend_defs.hpp"
#include "input_model.hpp"
#include "openvino/core/any.hpp"
#include "openvino/core/extension.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/op_extension.hpp"

namespace ov {
namespace frontend {
/// \brief An interface for identifying a frontend for a particular framework.
/// Provides an ability to load and convert of input model
class FRONTEND_API FrontEnd {
public:
    typedef std::shared_ptr<FrontEnd> Ptr;

    FrontEnd();

    virtual ~FrontEnd();

    /// \brief Validates if FrontEnd can recognize model with parameters specified.
    /// Same parameters should be used to load model.
    /// \param vars Any number of parameters of any type. What kind of parameters
    /// are accepted is determined by each FrontEnd individually, typically it is
    /// std::string containing path to the model file. For more information please
    /// refer to specific FrontEnd documentation.
    /// \return true if model recognized, false - otherwise.
    template <typename... Types>
    inline bool supported(const Types&... vars) const {
        return supported_impl({ov::Any(vars)...});
    }

    /// \brief Loads an input model by any specified arguments. Each FrontEnd separately
    /// defines what arguments it can accept.
    /// \param vars Any number of parameters of any type. What kind of parameters
    /// are accepted is determined by each FrontEnd individually, typically it is
    /// std::string containing path to the model file. For more information please
    /// refer to specific FrontEnd documentation.
    /// \return Loaded input model.
    template <typename... Types>
    inline InputModel::Ptr load(const Types&... vars) const {
        return load_impl({ov::Any{vars}...});
    }

    inline InputModel::Ptr load(const ov::AnyVector& vars) const {
        return load_impl(vars);
    }

    /// \brief Completely convert and normalize entire function, throws if it is not
    /// possible
    /// \param model Input model
    /// \return fully converted nGraph function
    virtual std::shared_ptr<ov::Model> convert(InputModel::Ptr model) const;

    /// \brief Completely convert the remaining, not converted part of a function.
    /// \param partiallyConverted partially converted nGraph function
    virtual void convert(std::shared_ptr<ov::Model> partially_converted) const;

    /// \brief Convert only those parts of the model that can be converted leaving others
    /// as-is. Converted parts are not normalized by additional transformations; normalize
    /// function or another form of convert function should be called to finalize the
    /// conversion process.
    /// \param model Input model
    /// \return partially converted nGraph function
    virtual std::shared_ptr<ov::Model> convert_partially(InputModel::Ptr model) const;

    /// \brief Convert operations with one-to-one mapping with decoding nodes.
    /// Each decoding node is an nGraph node representing a single FW operation node with
    /// all attributes represented in FW-independent way.
    /// \param model Input model
    /// \return nGraph function after decoding
    virtual std::shared_ptr<ov::Model> decode(InputModel::Ptr model) const;

    /// \brief Runs normalization passes on function that was loaded with partial conversion
    /// \param function partially converted nGraph function
    virtual void normalize(std::shared_ptr<ov::Model> function) const;

    /// \brief Gets name of this FrontEnd. Can be used by clients
    /// if frontend is selected automatically by FrontEndManager::load_by_model
    ///
    /// \return Current frontend name. Empty string if not implemented
    virtual std::string get_name() const;

    /// \brief Register base extension in the FrontEnd
    /// \param extension base extension
    virtual void add_extension(const std::shared_ptr<ov::Extension>& extension);
    /// \brief Register base extensions in the FrontEnd
    /// \param extensions vector of extensions
    void add_extension(const std::vector<std::shared_ptr<ov::Extension>>& extensions);
    /// \brief Registers extension
    /// \param library_path path to library with ov::Extension
    void add_extension(const std::string& library_path);

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
    /// \brief Registers extension
    /// \param library_path path to library with ov::Extension
    void add_extension(const std::wstring& library_path);
#endif

    /// @brief Registers extension
    /// @param extension Extension class which is inherited from ov::BaseOpExtension class
    template <class T, typename std::enable_if<std::is_base_of<ov::Extension, T>::value, bool>::type = true>
    void add_extension(const T& extension) {
        std::shared_ptr<ov::Extension> ext = std::make_shared<T>(extension);
        add_extension(ext);
    }
    /// @brief Registers extensions
    /// @param extension Extension class which is inherited from ov::Extension class
    template <class T,
              class... Targs,
              typename std::enable_if<std::is_base_of<ov::Extension, T>::value, bool>::type = true>
    void add_extension(const T& extension, Targs... args) {
        std::shared_ptr<ov::Extension> ext = std::make_shared<T>(extension);
        add_extension(ext);
        add_extension(args...);
    }

protected:
    virtual bool supported_impl(const std::vector<ov::Any>& variants) const;
    virtual InputModel::Ptr load_impl(const std::vector<ov::Any>& variants) const;
};

template <>
inline bool FrontEnd::supported(const std::vector<ov::Any>& variants) const {
    return supported_impl(variants);
}

}  // namespace frontend

}  // namespace ov
