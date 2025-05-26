# NPU Properties
What is a property from NPU Plugin's POV and Step-by-step guide on how to add one

## Glossary

| Name     | Description | Example |
|:--------:|:---------   |:--------: |
| Property | a plugin interface which can be set or read. Can map to options or metrics` | `ov::log::level` | 
| Option   | a configuration entry in our internal configuration.</br>Consists of an OptionBase template descriptor + a template OptionValue value. | `LOG_LEVEL` |
| Config   | Our internal database of configuration keys and their values | `_globalConfig` |
| Metric   | A property which does not map to any configuration key and configuration entry in our internal config.</br>Usually a static value directly read from driver or OS. | `ov::device::pci_info` |
| Compiler | Npu compiler as viewed from the plugin’s perspective.</br>Can be Compiler-In-Driver or Compiler-In-Plugin | `CID` | 
| “Anonymous” property</br>OR</br>compiler-private property | A setting from application level which the plugin has no knowledge of</br>(it is not registered, plugin is not aware of its datatype)</br> but which the compiler reports as supported via is_supported() API. | N/A |

## Structure of a property (Class hierarchy)

![Properties Class Hierarchy](./img/properties_class_hierarchy.png)

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

    // Overload this for read-only options (metrics) 
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
is the highlevel configuration “database” which implements the mapping between OptionBase and templatized OptionValue.  
Maps and stores the user-defined values for each entry in OptionsDesc layer.  
Implements the top level configuration manipulation functions:  
get/update/has/getString/toString/fromString and handles typecasts, typeverification, parsing and conversions.  
```` Note: This layer is static, initialized once in plugin constructor with the provided (also static) OptionDesc. ````

### FilteredConfig 
is a derivative class of Config, used only by NPU Plugin, which implements additional filtering layers atop of the base config,  
such as enabling/disabling keys based on their availability/support on the current system configuration.  
```` Note: This layer dynamically changes based on system configuration and compiler_type ````

### Properties
is the top level class and serves as the NPU Plugin’s interface to OpenVino and the application layer.  
It’s main purpose is to implement get_property and set_property interfaces and the callback functions of each property.  
```` Note: This layer dynamically changes based on system configuration and compiler_type ````
  
# How to add a new public property

The following steps how to add a new simple property which maps to a compiler configuration option.  
_*simple in this context means that it has no special callback function required for it, just set/get_  

## Step 1. Define the new property
First step is to define the new property's name, datatype and string-name in the public header in  
```bash
openvino/src/inference/include/openvino/runtime/intel_npu/properties.hpp
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
openvino/src/plugins/intel_npu/al/config/options.hpp
```  
Example:  
```cpp
// 
// EXAMPLE_PROPERTY 
//  
struct EXAMPLE_PROPERTY final : OptionBase<EXAMPLE_PROPERTY, ov::intel_npu::example_property> {  

    static std::string_view key() { 
        return ov::intel_npu::example_property.name();
        } 

    static constexpr std::string_view getTypeName() { 
        return "ov::intel_npu::ExampleType"; 
    } 

    static ov::intel_npu::BatchMode defaultValue() { 
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
- defaultValue: returns the option's default value (if there was no user-defined value set, config.get or get_property(EXAMPLE_PROPERTY) will call this function  
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
(see class **OptionBase** in *npu_plugin/al/include/config/config.hpp* or Class Hierarchy section above)

## Step 3. Register the new option
Third step is to register the new option in the plugin:  
**plugin.cpp > function init_options()**:
```cpp
    REGISTER_OPTION(EXAMPLE_PROPERTY);
``` 
Notes:  
at this point, the npu plugin will take care of registering and managing the option in the internal configuration.  
It ensures that it is enabled or disabled based in the current system/environment/application configuration. 

## Step 4. Link the new property to the new option
Fourth step is to create and register the Property (which is basicly the interface to this configuration option) for both Plugin and CompiledModel (if needed) 
### For plugin
npu_plugin/plugin/src/properties.cpp > function registerPluginProperties()
```cpp
TRY_REGISTER_SIMPLE_PROPERTY(ov::intel_npu::example_property, EXAMPLE_PROPERTY);
```
**Explanation:**
this macro will register (if all conditions are met) a property with the name ov::intel_npu::example_property.name()  
which maps to our internal configuration named EXAMPLE_PROPERTY  
and has a simple callback function of config.get<EXAMPLE_PROPERTY>()
### For compiled-model (if required)
npu_plugin/plugin/src/properties.cpp > function registerCompiledModelProperties()
```cpp
TRY_REGISTER_COMPILEDMODEL_PROPERTY_IFSET(ov::intel_npu::example_property, EXAMPLE_PROPERTY);
```
**Explanation:**
this macro will register (if all conditions are met) a property with the name ov::intel_npu::example_property.name()  
which maps to our internal configuration named EXAMPLE_PROPERTY  
and has a simple callback function of config.get<EXAMPLE_PROPERTY>()  
**(!!) ONLY** if it has been previously explicitly set at compile time

## Step 5. Python bindings
In order for the property to be property exposed in python API, add python wrapper for the new property in pyOpenvino  
openvino/src/bindings/python/src/pyopenvino/core/properties/properties.cpp:  
In section // sumbodule npu  
```cpp
    wrap_property_RW(m_intel_npu, ov::intel_npu::example_property, "example_property"); 
```

## Step 6. Update documentation
Document the new property in the appropaite sections (+ additional information, if required) in:  
```bash
openvino/docs/articles_en/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.rst 
openvino/src/plugins/intel_npu/README.md 
```

# Adding a new property which requires custom functions
If the new property requires a custom callback function, only Step 4. changes.  
Instead of using TRY_REGISTER_SIMPLE_PROPERTY macro, you can choose from the following helper macros:  

### TRY_REGISTER_VARPUB_PROPERTY(OPT_NAME, OPT_TYPE, PROP_VISIBILITY)  
This can be used when callback is standard, but visibility (public/private) is custom.  
Instead of using automaticly the value from optionsBase, one can define a custom function to determine  
whether the property will be public (included in supported_properties) or private and provide as PROP_VISIBILITY parameter. 
### TRY_REGISTER_CUSTOMFUNC_PROPERTY(OPT_NAME, OPT_TYPE, PROP_RETFUNC) 
This macro can be used whenever a custom callback function is required for this property,  
provided as a lambda function as PROP_RETFUNC parameter.  
(Standard callback function just returns the value of the config)  
### TRY_REGISTER_CUSTOM_PROPERTY(OPT_NAME, OPT_TYPE, PROP_VISIBILITY, PROP_MUTABILITY, PROP_RETFUNC)
This macro bypasses all automatic descriptor fetching, availability checks, and compatibility verifications  
and gives you the possibility to register a completely custom property with custom visibility and custom callback function. 

# Removing a public property
Removing a public property can be done by removing everything added in section “How do add a new public property” step-by-step.  