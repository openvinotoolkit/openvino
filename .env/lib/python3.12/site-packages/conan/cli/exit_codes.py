# Exit codes for conan command:
SUCCESS = 0                             # 0: Success (done)
ERROR_GENERAL = 1                       # 1: General ConanException error (done)
ERROR_MIGRATION = 2                     # 2: Migration error
USER_CTRL_C = 3                         # 3: Ctrl+C
USER_CTRL_BREAK = 4                     # 4: Ctrl+Break
ERROR_SIGTERM = 5                       # 5: SIGTERM
ERROR_INVALID_CONFIGURATION = 6         # 6: Invalid configuration (done)
ERROR_UNEXPECTED = 7
