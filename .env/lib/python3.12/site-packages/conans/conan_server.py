import argparse
import os

from conans.server.launcher import ServerLauncher


def run():
    parser = argparse.ArgumentParser(description='Launch the server')
    parser.add_argument('--migrate', default=False, action='store_true',
                        help='Run the pending migrations')
    parser.add_argument('--server_dir', '-d', default=None,
                        help='Specify where to store server config and data.')
    args = parser.parse_args()
    launcher = ServerLauncher(force_migration=args.migrate,
                              server_dir=args.server_dir or os.environ.get("CONAN_SERVER_HOME"))
    launcher.launch()


if __name__ == '__main__':
    run()
