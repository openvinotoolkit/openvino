class ConanException(Exception):
    """ Generic conan exception """
    def __init__(self, msg=None, remote=None):
        self.remote = remote
        super().__init__(msg)

    def __str__(self):
        msg = super().__str__()
        if self.remote:
            return f"{msg}. [Remote: {self.remote.name}]"
        return msg


class ConanInvalidConfiguration(ConanException):
    """
    This binary, for the requested configuration and package-id cannot be built
    """
    pass


class ConanMigrationError(ConanException):
    pass
