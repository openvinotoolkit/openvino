# Keep everything private until we review what is really needed and refactor passing "conanfile"
# from conan.tools.apple.apple import apple_dot_clean
# from conan.tools.apple.apple import apple_sdk_name
# from conan.tools.apple.apple import apple_deployment_target_flag
from conan.tools.apple.apple import fix_apple_shared_install_name, is_apple_os, to_apple_arch, XCRun
from conan.tools.apple.xcodedeps import XcodeDeps
from conan.tools.apple.xcodebuild import XcodeBuild
from conan.tools.apple.xcodetoolchain import XcodeToolchain
