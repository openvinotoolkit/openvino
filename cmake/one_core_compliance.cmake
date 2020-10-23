#has to be set before project
if(MSVC_ONE_CORE_COMPLIANCE)
    message([STATUS] "Building OneCore Compliant Binaries")
    if(DEFINED ENV{varBuildWdkVer})
        # For QB or manual setting - QB sets varBuildWdkVer before build
        message([STATUS] "Using passed SDK version")
        set(CMAKE_SYSTEM_VERSION 10.0.$ENV{varBuildWdkVer}.0 CACHE STRING INTERNAL FORCE)
    else()
        message([STATUS] "Using local SDK version")
        # CMake by default will set highest
        set(CMAKE_SYSTEM_VERSION 10.0 CACHE STRING INTERNAL FORCE)
    endif()

    message("Building for Windows OneCore compliants (using OneCoreUap.lib)")
    # Needed for cmake to not append project dir to $(VC_LibraryPath_VC_x64_OneCore). Because cmake inteprets it as relative path.
    if (X86_64)
        # Older and newer onecore libraries path (depends on WDK version).
        # Forcefull make VS search for C++ libreries in these folders priori to other c++ standard libraries localizations.
        add_link_options("/LIBPATH:\"\$\(VCInstallDir\)/lib/onecore/amd64\"")
        add_link_options("/LIBPATH:\"\$\(VC_LibraryPath_VC_x64_OneCore\)\"")
        set(CMAKE_C_STANDARD_LIBRARIES "\$\(UCRTContentRoot\)lib/\$\(TargetUniversalCRTVersion\)/um/\$\(Platform\)/OneCoreUap.lib" CACHE STRING "" FORCE)
        set(CMAKE_CXX_STANDARD_LIBRARIES "\$\(UCRTContentRoot\)lib/\$\(TargetUniversalCRTVersion\)/um/\$\(Platform\)/OneCoreUap.lib" CACHE STRING "" FORCE)
    else() #x86
        set(CMAKE_C_STANDARD_LIBRARIES "\$\(UCRTContentRoot\)lib/\$\(TargetUniversalCRTVersion\)/um/x86/OneCoreUap.lib" CACHE STRING "" FORCE)
        set(CMAKE_CXX_STANDARD_LIBRARIES "\$\(UCRTContentRoot\)lib/\$\(TargetUniversalCRTVersion\)/um/x86/OneCoreUap.lib" CACHE STRING "" FORCE)
        add_link_options("/LIBPATH:\"\$\(VCInstallDir\)lib/onecore\"")
        add_link_options("/LIBPATH:\"\$\(VC_LibraryPath_VC_x86_OneCore\)\"")
    endif()

    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /NODEFAULTLIB:kernel32.lib /NODEFAULTLIB:user32.lib /NODEFAULTLIB:advapi32.lib /NODEFAULTLIB:ole32.lib /NODEFAULTLIB:mscoree.lib /NODEFAULTLIB:combase.lib")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:kernel32.lib /NODEFAULTLIB:user32.lib /NODEFAULTLIB:advapi32.lib /NODEFAULTLIB:ole32.lib /NODEFAULTLIB:mscoree.lib /NODEFAULTLIB:combase.lib")
    #set(MSVC_IGNORED_LIBRARIES "/NODEFAULTLIB:kernel32.lib /NODEFAULTLIB:user32.lib /NODEFAULTLIB:advapi32.lib /NODEFAULTLIB:ole32.lib /NODEFAULTLIB:mscoree.lib /NODEFAULTLIB:combase.lib")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /I\$\(UniversalCRT_IncludePath\)")
    include(msvc_utils)
    configure_msvc_runtime()
endif()