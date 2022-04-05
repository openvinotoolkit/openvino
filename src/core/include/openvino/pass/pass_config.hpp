// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <list>
#include <memory>
#include <vector>

#include "ngraph/compatibility.hpp"
#include "openvino/core/core_visibility.hpp"
#include "openvino/core/deprecated.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/node.hpp"

namespace ov {
namespace pass {
using param_callback = std::function<bool(const std::shared_ptr<const ::ov::Node>)>;
using param_callback_map = std::map<ov::DiscreteTypeInfo, param_callback>;

/// \brief Class representing a transformations config that is used for disabling/enabling
/// transformations registered inside pass::Manager and also allows to set callback for all
/// transformations or for particular transformation.
///
/// When pass::Manager is created all passes registered inside this manager including nested
/// passes will share the same instance of PassConfig class.
/// To work with this class first you need to get shared instance of this class by calling
/// manager.get_pass_config() method. Then you will be able to disable/enable passes based
/// on transformations type_info. For example:
///
///     pass::Manager manager;
///     manager.register_pass<CommonOptimizations>();
///     auto pass_config = manager.get_pass_config();
///     pass_config->disable<ConvertGELU>(); // this will disable nested pass inside
///                                          // CommonOptimizations pipeline
///     manager.run_passes(f);
///
/// Sometimes it is needed to call transformation inside other transformation manually. And
/// for that case before running transformation you need manually check that this pass is
/// not disabled and then you need to set current PassConfig instance to this
/// transformation. For example:
///
///     // Inside MatcherPass callback or inside FunctionPass run_on_function() method
///     // you need to call get_pass_config() method to get shared instance of PassConfig
///     auto pass_config = get_pass_config();
///
///     // Before running nested transformation you need to check is it disabled or not
///     if (!pass_config->is_disabled<ConvertGELU>()) {
///         auto pass = ConvertGELU();
///         pass->set_pass_config(pass_config);
///         pass.apply(node);
///     }
///
/// Following this logic inside your transformations you will guaranty that transformations
/// will be executed in a right way.
/// \ingroup ov_pass_cpp_api
class OPENVINO_API PassConfig {
public:
    /// \brief Disable transformation by its type_info
    /// \param type_info Transformation type_info
    void disable(const DiscreteTypeInfo& type_info);
    /// \brief Disable transformation by its class type (based on type_info)
    template <class T, typename std::enable_if<!ngraph::HasTypeInfoMember<T>::value, bool>::type = true>
    void disable() {
        disable(T::get_type_info_static());
    }
    template <class T, typename std::enable_if<ngraph::HasTypeInfoMember<T>::value, bool>::type = true>
    void disable() {
        disable(T::type_info);
    }

    /// \brief Enable transformation by its type_info
    /// \param type_info Transformation type_info
    void enable(const DiscreteTypeInfo& type_info);
    /// \brief Enable transformation by its class type (based on type_info)
    template <class T, typename std::enable_if<!ngraph::HasTypeInfoMember<T>::value, bool>::type = true>
    void enable() {
        enable(T::get_type_info_static());
    }
    template <class T, typename std::enable_if<ngraph::HasTypeInfoMember<T>::value, bool>::type = true>
    void enable() {
        OPENVINO_SUPPRESS_DEPRECATED_START
        enable(T::type_info);
        OPENVINO_SUPPRESS_DEPRECATED_END
    }

    /// \brief Set callback for all kind of transformations
    void set_callback(const param_callback& callback) {
        m_callback = callback;
    }
    template <typename... Args>
    typename std::enable_if<sizeof...(Args) == 0>::type set_callback(const param_callback& callback) {}

    /// \brief Set callback for particular transformation class types
    ///
    /// Example below show how to set callback for one or multiple passes using this method.
    ///
    ///     pass_config->set_callback<ov::pass::ConvertBatchToSpace,
    ///                               ov::pass::ConvertSpaceToBatch>(
    ///              [](const_node_ptr &node) -> bool {
    ///                   // Disable transformations for cases when input shape rank is not
    ///                   equal to 4
    ///                   const auto input_shape_rank =
    ///                   node->get_output_partial_shape(0).rank().get_length();
    ///                   if (input_shape_rank != 4) {
    ///                       return false;
    ///                   }
    ///                   return true;
    ///               });
    ///
    /// Note that inside transformations you must provide code that work with this callback.
    /// See example below:
    ///
    ///     if (transformation_callback(node)) {
    ///         return false; // exit from transformation
    ///     }
    ///
    template <typename T,
              class... Args,
              typename std::enable_if<!ngraph::HasTypeInfoMember<T>::value, bool>::type = true>
    void set_callback(const param_callback& callback) {
        m_callback_map[T::get_type_info_static()] = callback;
        set_callback<Args...>(callback);
    }
    template <typename T,
              class... Args,
              typename std::enable_if<ngraph::HasTypeInfoMember<T>::value, bool>::type = true>
    void set_callback(const param_callback& callback) {
        OPENVINO_SUPPRESS_DEPRECATED_START
        m_callback_map[T::type_info] = callback;
        OPENVINO_SUPPRESS_DEPRECATED_END
        set_callback<Args...>(callback);
    }

    /// \brief Get callback for given transformation type_info
    /// \param type_info Transformation type_info
    ///
    /// In case if callback wasn't set for given transformation type then global callback
    /// will be returned. But if even global callback wasn't set then default callback will
    /// be returned.
    param_callback get_callback(const DiscreteTypeInfo& type_info) const;

    /// \brief Get callback for given transformation class type
    /// \return callback lambda function
    template <class T, typename std::enable_if<ngraph::HasTypeInfoMember<T>::value, bool>::type = true>
    param_callback get_callback() const {
        OPENVINO_SUPPRESS_DEPRECATED_START
        return get_callback(T::type_info);
        OPENVINO_SUPPRESS_DEPRECATED_END
    }
    template <class T, typename std::enable_if<!ngraph::HasTypeInfoMember<T>::value, bool>::type = true>
    param_callback get_callback() const {
        return get_callback(T::get_type_info_static());
    }

    /// \brief Check either transformation type is disabled or not
    /// \param type_info Transformation type_info
    /// \return true if transformation type was disabled and false otherwise
    bool is_disabled(const DiscreteTypeInfo& type_info) const {
        return m_disabled.count(type_info);
    }

    /// \brief Check either transformation class type is disabled or not
    /// \return true if transformation type was disabled and false otherwise
    template <class T, typename std::enable_if<ngraph::HasTypeInfoMember<T>::value, bool>::type = true>
    bool is_disabled() const {
        OPENVINO_SUPPRESS_DEPRECATED_START
        return is_disabled(T::type_info);
        OPENVINO_SUPPRESS_DEPRECATED_END
    }
    template <class T, typename std::enable_if<!ngraph::HasTypeInfoMember<T>::value, bool>::type = true>
    bool is_disabled() const {
        return is_disabled(T::get_type_info_static());
    }

    /// \brief Check either transformation type is force enabled or not
    /// \param type_info Transformation type_info
    /// \return true if transformation type was force enabled and false otherwise
    bool is_enabled(const DiscreteTypeInfo& type_info) const {
        return m_enabled.count(type_info);
    }

    /// \brief Check either transformation class type is force enabled or not
    /// \return true if transformation type was force enabled and false otherwise
    template <class T, typename std::enable_if<ngraph::HasTypeInfoMember<T>::value, bool>::type = true>
    bool is_enabled() const {
        OPENVINO_SUPPRESS_DEPRECATED_START
        return is_enabled(T::type_info);
        OPENVINO_SUPPRESS_DEPRECATED_END
    }
    template <class T, typename std::enable_if<!ngraph::HasTypeInfoMember<T>::value, bool>::type = true>
    bool is_enabled() const {
        return is_enabled(T::get_type_info_static());
    }

    void add_disabled_passes(const PassConfig& rhs);

private:
    param_callback m_callback = [](const std::shared_ptr<const ::ov::Node>&) {
        return false;
    };
    param_callback_map m_callback_map;
    std::unordered_set<DiscreteTypeInfo> m_disabled;
    std::unordered_set<DiscreteTypeInfo> m_enabled;
};
}  // namespace pass
}  // namespace ov
