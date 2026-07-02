# NPU Properties
What is a property from NPU Plugin's POV and Step-by-step guide on how to add one  
Practical manual for NPU plugin properties

## Table of Contents

- [Glossary](#glossary)
- [Structure of a property (Class hierarchy)](#structure-of-a-property-class-hierarchy)
    - [Properties With Option vs Properties Without Option](#properties-with-option-vs-properties-without-option)
  - [OptionBase<T>](#optionbaset)
  - [OptionDesc](#optiondesc)
  - [Config](#config)
  - [FilteredConfig](#filteredconfig)
  - [Properties](#properties)
- [How to add a new public (option backed) property](#how-to-add-a-new-public-option-backed-property)
  - [Step 1. Define the new property](#step-1-define-the-new-property)
  - [Step 2. Define the internal option descriptor](#step-2-define-the-internal-option-descriptor)
  - [Step 3. Register the new option](#step-3-register-the-new-option)
  - [Step 4. Link the new property to the new option](#step-4-link-the-new-property-to-the-new-option)
    - [For plugin](#for-plugin)
    - [For compiled-model (if required)](#for-compiled-model-if-required)
  - [Step 5. Python bindings](#step-5-python-bindings)
  - [Step 6. Update documentation](#step-6-update-documentation)
- [How to add a new public property without option (callback-backed/read-only)](#how-to-add-a-new-public-property-without-option-callback-backedread-only)
  - [Step 1. Define the new property](#step-1-define-the-new-property-1)
    - [Step 2. Define and register a callback function for the new property](#step-2-define-and-register-a-callback-function-for-the-new-property)
  - [Step 3. Python bindings](#step-3-python-bindings)
  - [Step 4. Update documentation](#step-4-update-documentation)
- [Compiled-model properties](#compiled-model-properties)
- [Special cases](#special-cases)
  - [SC.1 Adding a new property which requires custom functions](#sc1-adding-a-new-property-which-requires-custom-functions)
    - [SC.2 Adding a new property without option which requires customization](#sc2-adding-a-new-property-without-option-which-requires-customization)
  - [SC.3 Filtering out options at registration phase](#sc3-filtering-out-options-at-registration-phase)
- [Removing a public property](#removing-a-public-property)

## Glossary

| Name     | Description | Example |
|:--------:|:---------   |:--------: |
| Property | a plugin interface which can be set or read | `ov::log::level` | 
| Option   | a configuration entry in our internal configuration.</br>Consists of an OptionBase template descriptor + a template OptionValue value. | `LOG_LEVEL` |
| Config   | Our internal database of configuration keys and their values | |
| Option-backed property | A property mapped to an Option entry in Config (`register_property*` helpers with config argument). | `ov::hint::performance_mode` |
| Property without option (callback-backed/read-only) | A property that is not mapped to Config and is implemented by callback logic (usually runtime/backend queries). | `ov::device::pci_info` |
| Compiler | Npu compiler as viewed from the plugin's perspective.</br>Can be Compiler-In-Driver or Compiler-In-Plugin | `CID` |
| "Anonymous" property</br>OR</br>compiler-private property | A setting from application level which the plugin has no knowledge of</br>(it is not registered, plugin is not aware of its datatype)</br> but which the compiler reports as supported via is_supported() API. | N/A |

<br>

## Structure of a property (Class hierarchy)

![Properties Class Hierarchy](./img/properties_class_hierarchy.png)

<br>

### Properties With Option vs Properties Without Option
As it can be observed in the above class hierarchy diagram, a Property is a public interface to an internal information, the top layer of abstraction.  
A property can be implemented either through Config options or through callback logic.
The main difference is that while option-backed properties are entries in our internal database (Config) which can be modified at any time, properties without option
do not exist in the internal database and are computed/read on demand from backend/driver/OS.
Examples in NPU Plugin are: driver version, compiler version, device architecture, pci information, gops, uuid, luid, etc.
To summarize:
A property can be implemented in **one** of the 2 ways:
- Option-backed property: has an entry in internal Config and is managed through Option descriptors/helpers.
- Property without option: implemented through callback logic (typically read-only runtime/backend information).

### OptionBase\<T\> 
Implements the option descriptor. This class contains all the details of a config option: name, datatype, default value, parser, public/private, mutability, compiler version (for legacy support), etc. This serves as the key in our configuration map. 

Class definition in npu_plugin/al/include/config/config.hpp:  
```cpp
struct OptionBase { 
    using ValueType = T; 

    // `ActualOpt` must implement the following method: 
    // static std::string_view key() 

    static constexpr std::string_view getTypeName() { 
        if constexpr (TypePrinter<T>::hasName()) { 
            return TypePrinter<T>::name(); 
        } 
        static_assert(TypePrinter<T>::hasName(), 
                      "Options type is not a standard type, please add `getTypeName()` to your option"); 
    } 
    // Overload this to provide environment variable support. 
    static std::string_view envVar() { 
        return ""; 
    } 

    // Overload this to provide deprecated keys names. 
    static std::vector<std::string_view> deprecatedKeys() { 
        return {}; 
    } 

    // Overload this to provide default value if it wasn't specified by user. 
    // If it is std::nullopt - exception will be thrown in case of missing option access. 
    static std::optional<T> defaultValue() { 
        return std::nullopt; 
    } 

    // Overload this to provide more specific parser. 
    static ValueType parse(std::string_view val) { 
        return OptionParser<ValueType>::parse(val); 
    } 

    // Overload this to provide more specific validation 
    static void validateValue(const ValueType&) {} 

    // Overload this to provide more specific implementation. 
    static OptionMode mode() { 
        return OptionMode::Both; 
    } 

    // Overload this for private options. 
    static bool isPublic() { 
        return false; 
    } 

    // Overload this for read-only properties 
    static ov::PropertyMutability mutability() { 
        return ov::PropertyMutability::RW; 
    } 

    /// Overload this for options conditioned by compiler version 
    static uint32_t compilerSupportVersion() { 
        return ONEAPI_MAKE_VERSION(0, 0); 
    } 

    static std::string toString(const ValueType& val) { 
        return OptionPrinter<ValueType>::toString(val); 
    } 
}; 
```

### OptionDesc  
is storage for the registered options This is the base map which stores the available optionBase descriptors.  
This layer implements the option database manipulation functions: add/remove/has/reset
```` Note: This layer is static, intialized once in plugin constructor with all the options the Plugin has knowledge of. ````

### Config
is the highlevel configuration "database" which implements the mapping between OptionBase and templatized OptionValue.
Maps and stores the user-defined values for each entry in OptionsDesc layer.
Implements the top level configuration manipulation functions:
get/update/has/getString/toString/fromString and handles typecasts, typeverification, parsing and conversions.
```` Note: This layer is static, initialized once in plugin constructor with the provided (also static) OptionDesc. ````

### FilteredConfig
is a derivative class of Config, used only by NPU Plugin, which implements additional filtering layers atop of the base config,
such as enabling/disabling keys based on their availability/support on the current system configuration.
```` Note: This layer dynamically changes based on system configuration and compiler_type ````

### Properties
is the top level class and serves as the NPU Plugin's interface to OpenVino and the application layer.
It's main purpose is to implement get_property and set_property interfaces and the callback functions of each property.
```` Note: This layer dynamically changes based on system configuration and compiler_type ````

<br><br>

# How to add a new public (option backed) property

The following steps how to add a new simple property which maps to a compiler configuration option.  
_*simple in this context means that it has no special callback function required for it, just set/get_  

## Step 1. Define the new property
First step is to define the new property's name, datatype and string-name in the public header in  
```bash
src/inference/include/openvino/runtime/intel_npu/properties.hpp
```  
Example:  
```cpp
static constexpr ov::Property<ExampleType,ov::PropertyMutability::RW> example_property{"NPU_EXAMPLE_PROPERTY"};
```
Notes:  
- please note the NPU_ prefix in the property's string name. This is mandatory for npu-only private properties 
- mutability is Read-Write
- datatype of the property is enum ExampleType { VAL1, VAL2, VAL3 }  

## Step 2. Define the internal option descriptor
Second step is to define the optionDesc class for this property in  
```bash
src/plugins/intel_npu/src/al/include/intel_npu/config/options.hpp
```  
Example:  
```cpp
// 
// EXAMPLE_PROPERTY 
//  
struct EXAMPLE_PROPERTY final : OptionBase<EXAMPLE_PROPERTY, ov::intel_npu::ExampleType> {  

    static std::string_view key() { 
        return ov::intel_npu::example_property.name();
        } 

    static constexpr std::string_view getTypeName() { 
        return "ov::intel_npu::ExampleType"; 
    } 

    static ov::intel_npu::ExampleType defaultValue() { 
        return ov::intel_npu::ExampleType::VAL3; 
    } 

    static uint32_t compilerSupportVersion() { 
        return ONEAPI_MAKE_VERSION(5, 5); 
    } 

    static bool isPublic() { 
        return true; 
    } 

    static OptionMode mode() { 
        return OptionMode::Both; 
    } 
     
    static ov::PropertyMutability mutability() { 
        return ov::PropertyMutability::RW; 
    } 
     
    static std::string_view envVar() { 
        return "IE_NPU_EXAMPLE_PROPERTY"; 
    } 

    static ov::intel_npu::ExampleType parse(std::string_view val) { 
        if (val == "VAL1") { 
            return ov::intel_npu::ExampleType::VAL1; 
        } else if (val == "VAL2") { 
            return ov::intel_npu::ExampleType::VAL2; 
        } else if (val == "VAL3") { 
            return ov::intel_npu::ExampleType::VAL3; 
        } 

        OPENVINO_THROW("Value '", val, "'is not a valid EXAMPLE_PROPERTY option"); 
    } 

    static std::string toString(const ov::intel_npu::ExampleType& val) { 
        std::stringstream strStream; 

        strStream << val; 

        return strStream.str(); 
    } 
}; 
```
Notes:  
- key(): needs to return the string name of the property (the NPU_EXAMPLE_PROPERTY defined in the property at step 1)  
- getTypeName: returns the type name as a human-readable string  
- defaultValue: returns the option's default value (if there was no user-defined value set, config.get or get_property(EXAMPLE_PROPERTY) will call this function)
- compilerSupportVersion: the compiler version from which this key is supported by compiler  
- isPublic: defines whether the option is a **public or a private** one  
- mode: defines the OptionMode of this option. Can be:  
    - CompileTime (for options used ONLY by the compiler)  
    - Runtime (for options only used by plugin and runtime)  
    - Both (for options used by both).  
    **Only options of CompileTime and Both will be sent to compiler at model compilation.**  
- mutability: whether the option is **Read-Write** or **Read-Only**  
- envVar: environment variable (if needed) for this property. The config manager will check if the options have envVar defined. For each option which has envVar, it will look in environment variables and update the option value from there at init.  
- parse: string to custom datatype parser. If the property will be set with a string value, this parser will convert it into the internal datatype.  
- toString: for converting the option value from the custom datatype to string 

**(!!)** None of the member functions are mandatory to be defined.  
If any is missing, the default function will be used for its call, as defined in the OptionsBase class  
(see class **OptionBase** in `src/plugins/intel_npu/src/al/include/intel_npu/config/config.hpp` or Class Hierarchy section above)

## Step 3. Register the new option
Third step is to register the new option in the plugin:  
**src/plugins/intel_npu/src/plugin/src/plugin.cpp > function init_config(...)**
```cpp
    register_options<
        /* existing options... */
        EXAMPLE_PROPERTY
    >(options, config);
``` 
Notes:  
at this point, the npu plugin will take care of registering and managing the option in the internal configuration.  
It ensures that it is enabled or disabled based on the current system/environment/application configuration. 

## Step 4. Link the new property to the new option
Fourth step is to create and register the Property (which is basically the interface to this configuration option) for both Plugin and CompiledModel (if needed) 
### For plugin
src/plugins/intel_npu/src/plugin/src/plugin_property_manager.cpp > function PluginPropertyManager::registerProperties()
```cpp
register_property<EXAMPLE_PROPERTY>(_config, _properties, ov::intel_npu::example_property.name());
```
**Explanation:**
this helper function registers a property with the name ov::intel_npu::example_property.name()  
which maps to our internal configuration named EXAMPLE_PROPERTY, and is supported when the option is available.  
and has a simple callback function of config.get<EXAMPLE_PROPERTY>()
### For compiled-model (if required)
src/plugins/intel_npu/src/plugin/src/compiled_model_property_manager.cpp > function CompiledModelPropertyManager::registerProperties()
```cpp
register_property_as_read_only_mark_supported_if_set<EXAMPLE_PROPERTY>(_config,
                                                                        _properties,
                                                                        ov::intel_npu::example_property.name());
```
**Explanation:**
this helper function registers the compiled-model property with the name ov::intel_npu::example_property.name()
which maps to our internal configuration named EXAMPLE_PROPERTY
and has a simple callback function of config.get<EXAMPLE_PROPERTY>()
**(!!) ONLY** if it has been previously explicitly set at compile time.

## Step 5. Python bindings
In order for the property to be exposed in Python API, add python wrapper for the new property in pyOpenvino  
src/bindings/python/src/pyopenvino/core/properties/properties.cpp:  
In section // submodule npu  
```cpp
    wrap_property_RW(m_intel_npu, ov::intel_npu::example_property, "example_property"); 
```

## Step 6. Update documentation
Document the new property in the appropriate sections (+ additional information, if required) in:  
```bash
docs/articles_en/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.rst 
src/plugins/intel_npu/README.md 
```

<br><br>

# How to add a new public property without option (callback-backed/read-only)
As described in the first paragraph, these properties do not have an entry in our internal Config and are returned through callback logic at get_property calls.  
This means we do not need to define an Option nor register an Option for them.  

## Step 1. Define the new property
First step is to define the new property's name, datatype and string-name in the public header in  
```bash
src/inference/include/openvino/runtime/intel_npu/properties.hpp
```  
Example:  
```cpp
static constexpr ov::Property<ExampleType,ov::PropertyMutability::RO> example_property{"NPU_EXAMPLE_PROPERTY"};
```
Notes:  
- please note the NPU_ prefix in the property's string name. This is mandatory for npu-only private properties 
- mutability is Read-Only
- datatype of the property is enum ExampleType { VAL1, VAL2, VAL3 } 

## Step 2. Define and register a callback function for the new property
You need to register the new property and define a callback function in the owner-specific property manager.
For plugin: src/plugins/intel_npu/src/plugin/src/plugin_property_manager.cpp > function PluginPropertyManager::registerProperties()
For compiled-model: src/plugins/intel_npu/src/plugin/src/compiled_model_property_manager.cpp > function CompiledModelPropertyManager::registerProperties()

For properties without option, prefer a support-gated registration so the getter is only used when backend/runtime requirements are available:
```cpp
    const auto has_backend = [this]() {
        return _backend != nullptr;
    };

    register_property_with_support_and_custom_function(_properties,
                                                       ov::intel_npu::example_property.name(),
                                                       has_backend,
                                                       true,
                                                       [this](const ov::AnyMap&) {
                                                           return utils::getDriverVersion(_backend);
                                                       });
```
**Explanation**
this helper function registers a property with the name **ov::intel_npu::example_property (NPU_EXAMPLE_PROPERTY)**, which is public and included in supported_properties when the support predicate returns true.  
and calls `utils::getDriverVersion(_backend)` each time get_property is queried. The getter receives query-time
arguments as `ov::AnyMap`; use them when the property supports extra inputs, or ignore them when it does not.
Note: the first argument is the property name string (`property.name()`), not the property object itself.

## Step 3. Python bindings
In order for the property to be exposed in Python API, add python wrapper for the new property in pyOpenvino  
src/bindings/python/src/pyopenvino/core/properties/properties.cpp:  
In section // submodule npu  
```cpp
    wrap_property_RO(m_intel_npu, ov::intel_npu::example_property, "example_property"); 
```

## Step 4. Update documentation
Document the new property in the appropriate sections (+ additional information, if required) in:  
```bash
docs/articles_en/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.rst
src/plugins/intel_npu/README.md
```

<br><br>

# Compiled-model properties
By internal convention, what needs to be included in compiled-model properties gets decided based on the following statements:
- every option which has an effect on model compilation (meaning options with mode=OptionMode::CompileTime or OptionMode::Both) need to be included
- options (with some specific exceptions) should be publicly advertised in compiled-model's supported_properties **ONLY** if they have been explicitly set prior to model compilation.
- compiled-model properties (with a few specific exceptions) are all READ-ONLY, for the reason that the model has already been compiled.
This is to ensure that we only expose settings we are sure were taken into account by compiler.

For read-only config-backed properties in compiled-model, property_registration.hpp provides two related helper functions:

#### register_property_as_read_only<OPT_TYPE>(config, properties, propertyName)
Use this when the property should be read-only in compiled-model and advertised whenever the option is available.
Example:
```cpp
    register_property_as_read_only<LOG_LEVEL>(_config,
                                              _properties,
                                              ov::log::level.name());
```

#### register_property_as_read_only_mark_supported_if_set<OPT_TYPE>(config, properties, propertyName)
Use this when the property should be exposed only if a value for it exists in config (default values are not materialized in config and are read from OptionsDesc).
Example:
```cpp
    register_property_as_read_only_mark_supported_if_set<COMPILATION_MODE>(_config,
                                                                            _properties,
                                                                            ov::intel_npu::compilation_mode.name());
```

<br><br>

# Special cases
## SC.1 Adding a new property which requires custom functions
If the new property requires a custom callback function, only Step 4. changes.
Instead of using register_property helper function, you can choose from the following helper functions:

#### register_property_with_custom_visibility<OPT_TYPE>(config, properties, propertyName, isPublic)
This can be used when callback is standard, but visibility (public/private) is custom.
Instead of using automatically the value from the option descriptor, one can pass a runtime bool to determine
whether the property will be public (included in supported_properties) or private.
Example:
```cpp
    register_property_with_custom_visibility<RUN_INFERENCES_SEQUENTIALLY>(
        _config,
        _properties,
        ov::intel_npu::run_inferences_sequentially.name(),
        [&] { return _backend && _backend->getInitStructs(); }());
```

#### register_property_with_custom_function(config, properties, propertyName, getter)
This helper function can be used whenever a custom callback function/implementation is required for this property,
provided as a lambda function. The getter receives query-time arguments as `ov::AnyMap` and must return an
`ov::Any` value. The standard callback function just returns the value from config.
Example:
```cpp
    register_property_with_custom_function(_config, _properties, ov::intel_npu::stepping.name(),
        [this](const ov::AnyMap&) {
            if (!_config.has<STEPPING>()) {
                const auto specifiedDeviceName = _config.get<intel_npu::DEVICE_ID>();
                return static_cast<int64_t>(utils::getSteppingNumber(_backend, specifiedDeviceName));
            }
            return _config.get<STEPPING>();
        });
```

## SC.2 Adding a new property without option which requires customization
Apart from register_property_with_custom_function, two additional helper functions are available for properties without option:

#### register_property_with_support_and_custom_function(properties, propertyName, isSupported, isPublic, getter)
Registers a property and gates it through isSupported.
Use this when availability depends on runtime condition (e.g. backend capability check).
Example:
```cpp
    const auto has_backend_and_valid_device = [this]() {
        if (_backend == nullptr) {
            return false;
        }

        try {
            const auto specifiedDeviceName = _config.get<intel_npu::DEVICE_ID>();
            return utils::getDeviceById(_backend, specifiedDeviceName) != nullptr;
        } catch (...) {
            return false;
        }
    };

    register_property_with_support_and_custom_function(
        _properties,
        ov::device::full_name.name(),
        has_backend_and_valid_device,
        true,
        [this](const ov::AnyMap&) {
            return utils::getFullDeviceName(_backend, _config.get<intel_npu::DEVICE_ID>());
        });
```

The same pattern can be used for support-gating compiler-specific properties without option:
```cpp
    register_property_with_support_and_custom_function(
        _properties,
        ov::intel_npu::compiler_version.name(),
        [this]() {
            try {
                auto compilerType = _config.get<COMPILER_TYPE>();
                auto deviceId = _config.get<DEVICE_ID>();
                auto device = utils::getDeviceById(_backend, deviceId);

                auto compilationPlatform = utils::getCompilationPlatform(
                    _config.get<PLATFORM>(),
                    device == nullptr ? std::move(deviceId) : device->getName(),
                    _backend == nullptr ? std::vector<std::string>() : _backend->getDeviceNames());

                CompilerAdapterFactory factory;
                return factory.getCompiler(_backend, compilerType, compilationPlatform) != nullptr;
            } catch (...) {
                return false;
            }
        },
        true,
        [this](const ov::AnyMap&) {
            auto compilerType = _config.get<COMPILER_TYPE>();
            auto deviceId = _config.get<DEVICE_ID>();
            auto device = utils::getDeviceById(_backend, deviceId);

            auto compilationPlatform = utils::getCompilationPlatform(
                _config.get<PLATFORM>(),
                device == nullptr ? std::move(deviceId) : device->getName(),
                _backend == nullptr ? std::vector<std::string>() : _backend->getDeviceNames());

            CompilerAdapterFactory factory;
            auto dummyCompiler = factory.getCompiler(_backend, compilerType, compilationPlatform);
            return dummyCompiler->get_version();
        });
```

#### register_property_with_support_custom_function_and_args(properties, propertyName, isSupported, isPublic, getter)
Same as register_property_with_support_and_custom_function, but for properties whose getter also receives an ov::AnyMap of additional
arguments at get_property call time. Use this for properties such as `ov::compatibility_check` that accept
extra input arguments.
Example:
```cpp
    register_property_with_support_custom_function_and_args(
        _properties,
        ov::compatibility_check.name(),
        [this]() {
            return _compatibilityCheckFiltered && _compatibilityCheckSupported;
        },
        true,
        [this](const ov::AnyMap& arguments) {
            return validateCompatibilityDescriptor(_backend, arguments);
        });
```

## SC.3 Register all options, gate availability with enable flags
Enable/disable options based on runtime/backend support.
Do not skip registration for normal capability gating. Keep options known to the stack and control availability via config.enable(...).
For WORKLOAD_TYPE, this is exactly how availability is gated when backend support is missing.
Implementation point:
src/plugins/intel_npu/src/plugin/src/plugin.cpp > function init_config(...)
Example:
```cpp
    register_options</* base options..., */ WORKLOAD_TYPE>(options, config);

    config.enable(ov::workload_type.name(), backend != nullptr && backend->isCommandQueueExtSupported());
```
With this pattern, WORKLOAD_TYPE remains registered, but is disabled whenever backend is absent or command queue extension is not supported.

<br><br>

# Removing a public property
Removing a public property can be done by removing everything added in section "How to add a new public property" step-by-step.
