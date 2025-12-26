from conan.api.output import ConanOutput, Color


def print_profiles(profile_host, profile_build):
    out = ConanOutput()
    out.title("Input profiles")
    out.info("Profile host:", fg=Color.BRIGHT_CYAN)
    out.info(profile_host.dumps())
    out.info("Profile build:", fg=Color.BRIGHT_CYAN)
    out.info(profile_build.dumps())
