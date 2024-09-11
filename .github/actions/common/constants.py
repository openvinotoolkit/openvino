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
    'public_windows_vs2019_Release',
    'public_windows_vs2019_Debug',
)
ProductType = Enum('ProductType', {t.upper(): t for t in productTypes})
