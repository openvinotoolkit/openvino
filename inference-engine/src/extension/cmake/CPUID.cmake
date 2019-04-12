# the module will build and run cpuid utility, which store detected
# host processor features into cpuid.txt file in form:
# FEATURE [not] supported
# variable HAVE_CPUID_INFO set in case of success
# if variable HAVE_CPUID_INFO is set then it is possible
# to test HAVE_SSE42/HAVE_AVX2 variables

include (CheckCXXSourceRuns)

if(NOT WIN32 AND NOT APPLE)
    set(CMAKE_REQUIRED_FLAGS "-std=c++11")
endif()

check_cxx_source_runs(
"
// InstructionSet.cpp
// Compile by using: cl /EHsc /W4 InstructionSet.cpp
// processor: x86, x64
// Uses the __cpuid intrinsic to get information about
// CPU extended instruction set support.
//
// source origin:
// https://msdn.microsoft.com/en-us/library/hskdteyh.aspx
// https://gcc.gnu.org/git/?p=gcc.git;a=blob_plain;f=gcc/config/i386/driver-i386.c


#include <iostream>
#include <fstream>
#include <vector>
#include <bitset>
#include <array>
#include <string>
#ifdef WIN32
#include <intrin.h>
#else
#include <cpuid.h>
#include <string.h>
#endif

class InstructionSet
{
    // forward declarations
    class InstructionSet_Internal;

public:
    // getters
    static std::string Vendor(void) { return CPU_Rep.vendor_; }
    static std::string Brand(void) { return CPU_Rep.brand_; }

    static bool SSE3(void) { return CPU_Rep.f_1_ECX_[0]; }
    static bool PCLMULQDQ(void) { return CPU_Rep.f_1_ECX_[1]; }
    static bool MONITOR(void) { return CPU_Rep.f_1_ECX_[3]; }
    static bool SSSE3(void) { return CPU_Rep.f_1_ECX_[9]; }
    static bool FMA(void) { return CPU_Rep.f_1_ECX_[12]; }
    static bool CMPXCHG16B(void) { return CPU_Rep.f_1_ECX_[13]; }
    static bool SSE41(void) { return CPU_Rep.f_1_ECX_[19]; }
    static bool SSE42(void) { return CPU_Rep.f_1_ECX_[20]; }
    static bool MOVBE(void) { return CPU_Rep.f_1_ECX_[22]; }
    static bool POPCNT(void) { return CPU_Rep.f_1_ECX_[23]; }
    static bool AES(void) { return CPU_Rep.f_1_ECX_[25]; }
    static bool XSAVE(void) { return CPU_Rep.f_1_ECX_[26]; }
    static bool OSXSAVE(void) { return CPU_Rep.f_1_ECX_[27]; }
    static bool AVX(void) { return CPU_Rep.f_1_ECX_[28]; }
    static bool F16C(void) { return CPU_Rep.f_1_ECX_[29]; }
    static bool RDRAND(void) { return CPU_Rep.f_1_ECX_[30]; }

    static bool MSR(void) { return CPU_Rep.f_1_EDX_[5]; }
    static bool CX8(void) { return CPU_Rep.f_1_EDX_[8]; }
    static bool SEP(void) { return CPU_Rep.f_1_EDX_[11]; }
    static bool CMOV(void) { return CPU_Rep.f_1_EDX_[15]; }
    static bool CLFSH(void) { return CPU_Rep.f_1_EDX_[19]; }
    static bool MMX(void) { return CPU_Rep.f_1_EDX_[23]; }
    static bool FXSR(void) { return CPU_Rep.f_1_EDX_[24]; }
    static bool SSE(void) { return CPU_Rep.f_1_EDX_[25]; }
    static bool SSE2(void) { return CPU_Rep.f_1_EDX_[26]; }

    static bool FSGSBASE(void) { return CPU_Rep.f_7_EBX_[0]; }
    static bool BMI1(void) { return CPU_Rep.f_7_EBX_[3]; }
    static bool HLE(void) { return CPU_Rep.isIntel_ && CPU_Rep.f_7_EBX_[4]; }
    static bool AVX2(void) { return CPU_Rep.f_7_EBX_[5]; }
    static bool BMI2(void) { return CPU_Rep.f_7_EBX_[8]; }
    static bool ERMS(void) { return CPU_Rep.f_7_EBX_[9]; }
    static bool INVPCID(void) { return CPU_Rep.f_7_EBX_[10]; }
    static bool RTM(void) { return CPU_Rep.isIntel_ && CPU_Rep.f_7_EBX_[11]; }
    static bool AVX512F(void) { return CPU_Rep.f_7_EBX_[16]; }
    static bool RDSEED(void) { return CPU_Rep.f_7_EBX_[18]; }
    static bool ADX(void) { return CPU_Rep.f_7_EBX_[19]; }
    static bool AVX512PF(void) { return CPU_Rep.f_7_EBX_[26]; }
    static bool AVX512ER(void) { return CPU_Rep.f_7_EBX_[27]; }
    static bool AVX512CD(void) { return CPU_Rep.f_7_EBX_[28]; }
    static bool SHA(void) { return CPU_Rep.f_7_EBX_[29]; }

    static bool PREFETCHWT1(void) { return CPU_Rep.f_7_ECX_[0]; }

    static bool LAHF(void) { return CPU_Rep.f_81_ECX_[0]; }
    static bool LZCNT(void) { return CPU_Rep.isIntel_ && CPU_Rep.f_81_ECX_[5]; }
    static bool ABM(void) { return CPU_Rep.isAMD_ && CPU_Rep.f_81_ECX_[5]; }
    static bool SSE4a(void) { return CPU_Rep.isAMD_ && CPU_Rep.f_81_ECX_[6]; }
    static bool XOP(void) { return CPU_Rep.isAMD_ && CPU_Rep.f_81_ECX_[11]; }
    static bool TBM(void) { return CPU_Rep.isAMD_ && CPU_Rep.f_81_ECX_[21]; }

    static bool SYSCALL(void) { return CPU_Rep.isIntel_ && CPU_Rep.f_81_EDX_[11]; }
    static bool MMXEXT(void) { return CPU_Rep.isAMD_ && CPU_Rep.f_81_EDX_[22]; }
    static bool RDTSCP(void) { return CPU_Rep.isIntel_ && CPU_Rep.f_81_EDX_[27]; }
    static bool _3DNOWEXT(void) { return CPU_Rep.isAMD_ && CPU_Rep.f_81_EDX_[30]; }
    static bool _3DNOW(void) { return CPU_Rep.isAMD_ && CPU_Rep.f_81_EDX_[31]; }

private:
    static const InstructionSet_Internal CPU_Rep;

    class InstructionSet_Internal
    {
    public:
        InstructionSet_Internal()
            : nIds_{ 0 },
            nExIds_{ 0 },
            isIntel_{ false },
            isAMD_{ false },
            f_1_ECX_{ 0 },
            f_1_EDX_{ 0 },
            f_7_EBX_{ 0 },
            f_7_ECX_{ 0 },
            f_81_ECX_{ 0 },
            f_81_EDX_{ 0 },
            data_{},
            extdata_{}
        {
#ifdef WIN32
            std::array<int, 4> cpui;
#else
            std::array<unsigned int, 4> cpui;
#endif

            // Calling __cpuid with 0x0 as the function_id argument
            // gets the number of the highest valid function ID.
#ifdef WIN32
            __cpuid(cpui.data(), 0);
#else
            cpui[0] = __get_cpuid_max(0, &cpui[1]);
#endif
            nIds_ = cpui[0];

            for (int i = 0; i <= nIds_; ++i)
            {
#ifdef WIN32
                __cpuidex(cpui.data(), i, 0);
#else
                __cpuid_count(i, 0, cpui[0], cpui[1], cpui[2], cpui[3]);
#endif

                data_.push_back(cpui);
            }

            // Capture vendor string
            char vendor[0x20];
            memset(vendor, 0, sizeof(vendor));
            *reinterpret_cast<int*>(vendor + 0) = data_[0][1];
            *reinterpret_cast<int*>(vendor + 4) = data_[0][3];
            *reinterpret_cast<int*>(vendor + 8) = data_[0][2];
            vendor_ = vendor;
            if (vendor_ == \"GenuineIntel\")
            {
                isIntel_ = true;
            }
            else if (vendor_ == \"AuthenticAMD\")
            {
                isAMD_ = true;
            }

            // load bitset with flags for function 0x00000001
            if (nIds_ >= 1)
            {
                f_1_ECX_ = data_[1][2];
                f_1_EDX_ = data_[1][3];
            }

            // load bitset with flags for function 0x00000007
            if (nIds_ >= 7)
            {
                f_7_EBX_ = data_[7][1];
                f_7_ECX_ = data_[7][2];
            }

            // Calling __cpuid with 0x80000000 as the function_id argument
            // gets the number of the highest valid extended ID.
#ifdef WIN32
            __cpuid(cpui.data(), 0x80000000);
#else
            __cpuid(0x80000000, cpui[0], cpui[1], cpui[2], cpui[3]);
#endif
            nExIds_ = cpui[0];

            char brand[0x40];
            memset(brand, 0, sizeof(brand));

            for (int i = 0x80000000; i <= nExIds_; ++i)
            {
#ifdef WIN32
                __cpuidex(cpui.data(), i, 0);
#else
                __cpuid_count(i, 0, cpui[0], cpui[1], cpui[2], cpui[3]);
#endif
                extdata_.push_back(cpui);
            }

            // load bitset with flags for function 0x80000001
            if ((unsigned)nExIds_ >= 0x80000001)
            {
                f_81_ECX_ = extdata_[1][2];
                f_81_EDX_ = extdata_[1][3];
            }

            // Interpret CPU brand string if reported
            if ((unsigned)nExIds_ >= 0x80000004)
            {
                memcpy(brand +  0, extdata_[2].data(), sizeof(cpui));
                memcpy(brand + 16, extdata_[3].data(), sizeof(cpui));
                memcpy(brand + 32, extdata_[4].data(), sizeof(cpui));
                brand_ = brand;
            }
        };

        int nIds_;
        int nExIds_;
        std::string vendor_;
        std::string brand_;
        bool isIntel_;
        bool isAMD_;
        std::bitset<32> f_1_ECX_;
        std::bitset<32> f_1_EDX_;
        std::bitset<32> f_7_EBX_;
        std::bitset<32> f_7_ECX_;
        std::bitset<32> f_81_ECX_;
        std::bitset<32> f_81_EDX_;
#ifdef WIN32
        std::vector<std::array<int, 4>> data_;
        std::vector<std::array<int, 4>> extdata_;
#else
        std::vector<std::array<unsigned int, 4>> data_;
        std::vector<std::array<unsigned int, 4>> extdata_;
#endif
    };
};

// Initialize static member data
const InstructionSet::InstructionSet_Internal InstructionSet::CPU_Rep;

// Print out supported instruction set extensions
int main()
{
    std::ofstream fo(\"${CMAKE_BINARY_DIR}/cpuid.txt\");
    auto& outstream = fo;//std::cout;

    auto support_message = [&outstream](std::string isa_feature, bool is_supported) {
        outstream << isa_feature << (is_supported ? \" supported\" : \" not supported\") << std::endl;
    };

    std::cout << InstructionSet::Vendor() << std::endl;
    std::cout << InstructionSet::Brand() << std::endl;

    support_message(\"3DNOW\",       InstructionSet::_3DNOW());
    support_message(\"3DNOWEXT\",    InstructionSet::_3DNOWEXT());
    support_message(\"ABM\",         InstructionSet::ABM());
    support_message(\"ADX\",         InstructionSet::ADX());
    support_message(\"AES\",         InstructionSet::AES());
    support_message(\"AVX\",         InstructionSet::AVX());
    support_message(\"AVX2\",        InstructionSet::AVX2());
    support_message(\"AVX512CD\",    InstructionSet::AVX512CD());
    support_message(\"AVX512F\",     InstructionSet::AVX512F());
    support_message(\"AVX512ER\",    InstructionSet::AVX512ER());
    support_message(\"AVX512PF\",    InstructionSet::AVX512PF());
    support_message(\"BMI1\",        InstructionSet::BMI1());
    support_message(\"BMI2\",        InstructionSet::BMI2());
    support_message(\"CLFSH\",       InstructionSet::CLFSH());
    support_message(\"CMPXCHG16B\",  InstructionSet::CMPXCHG16B());
    support_message(\"CX8\",         InstructionSet::CX8());
    support_message(\"ERMS\",        InstructionSet::ERMS());
    support_message(\"F16C\",        InstructionSet::F16C());
    support_message(\"FMA\",         InstructionSet::FMA());
    support_message(\"FSGSBASE\",    InstructionSet::FSGSBASE());
    support_message(\"FXSR\",        InstructionSet::FXSR());
    support_message(\"HLE\",         InstructionSet::HLE());
    support_message(\"INVPCID\",     InstructionSet::INVPCID());
    support_message(\"LAHF\",        InstructionSet::LAHF());
    support_message(\"LZCNT\",       InstructionSet::LZCNT());
    support_message(\"MMX\",         InstructionSet::MMX());
    support_message(\"MMXEXT\",      InstructionSet::MMXEXT());
    support_message(\"MONITOR\",     InstructionSet::MONITOR());
    support_message(\"MOVBE\",       InstructionSet::MOVBE());
    support_message(\"MSR\",         InstructionSet::MSR());
    support_message(\"OSXSAVE\",     InstructionSet::OSXSAVE());
    support_message(\"PCLMULQDQ\",   InstructionSet::PCLMULQDQ());
    support_message(\"POPCNT\",      InstructionSet::POPCNT());
    support_message(\"PREFETCHWT1\", InstructionSet::PREFETCHWT1());
    support_message(\"RDRAND\",      InstructionSet::RDRAND());
    support_message(\"RDSEED\",      InstructionSet::RDSEED());
    support_message(\"RDTSCP\",      InstructionSet::RDTSCP());
    support_message(\"RTM\",         InstructionSet::RTM());
    support_message(\"SEP\",         InstructionSet::SEP());
    support_message(\"SHA\",         InstructionSet::SHA());
    support_message(\"SSE\",         InstructionSet::SSE());
    support_message(\"SSE2\",        InstructionSet::SSE2());
    support_message(\"SSE3\",        InstructionSet::SSE3());
    support_message(\"SSE4.1\",      InstructionSet::SSE41());
    support_message(\"SSE4.2\",      InstructionSet::SSE42());
    support_message(\"SSE4a\",       InstructionSet::SSE4a());
    support_message(\"SSSE3\",       InstructionSet::SSSE3());
    support_message(\"SYSCALL\",     InstructionSet::SYSCALL());
    support_message(\"TBM\",         InstructionSet::TBM());
    support_message(\"XOP\",         InstructionSet::XOP());
    support_message(\"XSAVE\",       InstructionSet::XSAVE());
    return 0;
}
"
HAVE_CPUID_INFO
)

if(HAVE_CPUID_INFO)
    set(_CPUID_INFO "${CMAKE_BINARY_DIR}/cpuid.txt")
    set(HAVE_AVX512F FALSE)
    set(HAVE_AVX2    FALSE)
    set(HAVE_SSE42   FALSE)

    file(STRINGS ${_CPUID_INFO} _FEATURES)

    ext_message(STATUS "Host CPU features:")

    foreach(FEATURE IN ITEMS ${_FEATURES})

        ext_message(STATUS "  ${FEATURE}")

        string(COMPARE EQUAL "${FEATURE}" "AVX512F supported" _FEATURE_FOUND)
        if(${_FEATURE_FOUND})
            if (NOT ${CMAKE_CXX_COMPILER_ID} STREQUAL GNU OR CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 4.9)
                set(HAVE_AVX512F ${_FEATURE_FOUND})
            else()
                ext_message(WARNING "Compiler doesn't support AVX512 instructuion set")
            endif()
        endif()
        string(COMPARE EQUAL "${FEATURE}" "AVX2 supported" _FEATURE_FOUND)
        if(${_FEATURE_FOUND})
            set(HAVE_AVX2 ${_FEATURE_FOUND})
        endif()
        string(COMPARE EQUAL "${FEATURE}" "SSE4.2 supported" _FEATURE_FOUND)
        if(${_FEATURE_FOUND})
            set(HAVE_SSE42 ${_FEATURE_FOUND})
        endif()
    endforeach(FEATURE)

    unset(_FEATURE_FOUND)
    unset(_CPUID_INFO)
    unset(_FEATURES)
endif()
