architecture_map = {
    'x86': 'x86',
    'x86_64': 'x86_64',
    'ppc32be': 'ppc',
    'ppc32': 'ppc',
    'ppc64le': 'ppc64',
    'ppc64': 'ppc64',
    'armv4': 'arm',
    'armv4i': 'arm',
    'armv5el': 'arm',
    'armv5hf': 'arm',
    'armv6': 'arm',
    'armv7': 'arm',
    'armv7hf': 'arm',
    'armv7s': 'arm',
    'armv7k': 'arm',
    'armv8': 'arm64',
    'armv8_32': 'arm64',
    'armv8.3': 'arm64',
    'sparc': 'sparc',
    'sparcv9': 'sparc64',
    'mips': 'mips',
    'mips64': 'mips64',
    'avr': 'avr',
    's390': 's390x',
    's390x': 's390x',
    'asm.js': None,
    'wasm': None,
    'wasm64': None,
    'sh4le': 'sh'
}


build_variant_map = {
    'Debug': 'debug',
    'Release': 'release',
    'RelWithDebInfo': 'profiling',
    'MinSizeRel': 'release'
}


optimization_map = {
    'MinSizeRel': 'small'
}


cxx_language_version_map = {
    '98': 'c++98',
    'gnu98': 'c++98',
    '11': 'c++11',
    'gnu11': 'c++11',
    '14': 'c++14',
    'gnu14': 'c++14',
    '17': 'c++17',
    'gnu17': 'c++17',
    '20': 'c++20',
    'gnu20': 'c++20'
}


target_platform_map = {
    'Windows': 'windows',
    'WindowsStore': 'windows',
    'WindowsCE': 'windows',
    'Linux': 'linux',
    'Macos': 'macos',
    'Android': 'android',
    'iOS': 'ios',
    'watchOS': 'watchos',
    'tvOS': 'tvos',
    'visionOS': 'xros',
    'FreeBSD': 'freebsd',
    'SunOS': 'solaris',
    'AIX': 'aix',
    'Emscripten': None,
    'Arduino': 'none',
    'Neutrino': 'qnx',
}


runtime_library_map = {
    'static': 'static',
    'dynamic': 'dynamic',
    'MD': 'dynamic',
    'MT': 'static',
    'MDd': 'dynamic',
    'MTd': 'static',
}
