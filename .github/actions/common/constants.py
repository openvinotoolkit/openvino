from enum import Enum


class EventType(Enum):
    pre_commit = 'pre_commit'
    commit = 'commit'


productTypes = (
    'public_linux_debian_10_arm_release',
    'public_linux_fedora_29_x86_64_release',
    'public_linux_ubuntu_20_04_x86_64_release',
    'public_linux_ubuntu_20_04_arm64_release',
    'public_linux_ubuntu_22_04_x86_64_release',
    'public_linux_ubuntu_22_04_dpcpp_x86_64_release',
    'public_linux_ubuntu_24_04_x86_64_release',
    'public_windows_vs2019_release',
    'public_windows_vs2019_debug',
    'public_windows_vs2022_release',
    'public_windows_vs2022_debug',
    'public_manylinux2014_x86_64_release',
    'public_macos_x86_64_release',
)
ProductType = Enum('ProductType', {t.upper(): t for t in productTypes})


platformKeys = (
    'centos7_x86_64',
    'debian10_armhf',
    'rhel8_x86_64',
    'ubuntu20_arm64',
    'ubuntu20_x86_64',
    'ubuntu22_x86_64',
    'ubuntu24_x86_64',
    'macos_12_6_arm64',
    'macos_12_6_x86_64',
    'windows_x86_64',
)
PlatformKey = Enum('PlatformKey', {t.upper(): t for t in platformKeys})

PlatformMapping = {
    PlatformKey.DEBIAN10_ARMHF: ProductType.PUBLIC_LINUX_DEBIAN_10_ARM_RELEASE,
    PlatformKey.UBUNTU20_X86_64: ProductType.PUBLIC_LINUX_UBUNTU_20_04_X86_64_RELEASE,
    PlatformKey.UBUNTU20_ARM64: ProductType.PUBLIC_LINUX_UBUNTU_20_04_ARM64_RELEASE,
    PlatformKey.UBUNTU22_X86_64: ProductType.PUBLIC_LINUX_UBUNTU_22_04_X86_64_RELEASE,
    PlatformKey.UBUNTU24_X86_64: ProductType.PUBLIC_LINUX_UBUNTU_24_04_X86_64_RELEASE,
    PlatformKey.WINDOWS_X86_64: ProductType.PUBLIC_WINDOWS_VS2022_RELEASE,
    PlatformKey.MACOS_12_6_X86_64: ProductType.PUBLIC_MACOS_X86_64_RELEASE,
}
