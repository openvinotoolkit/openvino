import json
import os
from collections import OrderedDict

from conan.api.conan_api import ConanAPI
from conan.api.model import Remote, LOCAL_RECIPES_INDEX
from conan.api.output import cli_out_write, Color
from conan.cli import make_abs_path
from conan.cli.command import conan_command, conan_subcommand, OnceArgument
from conan.cli.commands.list import remote_color, error_color, recipe_color, \
    reference_color
from conan.errors import ConanException


def _print_remotes_json(remotes):
    info = [{"name": r.name,
             "url": r.url,
             "verify_ssl": r.verify_ssl,
             "enabled": not r.disabled,
             "allowed_packages": r.allowed_packages,
             "recipes_only": r.recipes_only}
            for r in remotes]
    cli_out_write(json.dumps(info, indent=4))


def _print_remote_list(remotes):
    for r in remotes:
        output_str = str(r)
        cli_out_write(output_str)


def _print_remote_user_list(results):
    for remote_name, result in results.items():
        cli_out_write(f"{remote_name}:", fg=remote_color)
        if result["user_name"] is None:
            cli_out_write("  No user", fg=error_color)
        else:
            cli_out_write("  Username: ", fg=recipe_color, endline="")
            cli_out_write(result["user_name"], fg=reference_color)
            cli_out_write("  authenticated: ", fg=recipe_color, endline="")
            cli_out_write(result["authenticated"], fg=reference_color)


def _print_remote_user_set(results):
    for remote_name, result in results.items():
        from_user = "'{}'".format(result["previous_info"]["user_name"])
        from_user += " (anonymous)" \
            if not result["previous_info"]["authenticated"] else " (authenticated)"
        to_user = "'{}'".format(result["info"]["user_name"])
        to_user += " (anonymous)" \
            if not result["info"]["authenticated"] else " (authenticated)"
        message = "Changed user of remote '{}' from {} to {}".format(remote_name, from_user, to_user)
        cli_out_write(message)


def _print_remotes_users_json(results):
    cli_out_write(json.dumps(list(results.values())))


@conan_subcommand(formatters={"text": _print_remote_list, "json": _print_remotes_json})
def remote_list(conan_api: ConanAPI, parser, subparser, *args):  # noqa
    """
    List current remotes.
    """
    parser.parse_args(*args)
    return conan_api.remotes.list(only_enabled=False)


@conan_subcommand()
def remote_add(conan_api, parser, subparser, *args):
    """
    Add a remote.
    """
    subparser.add_argument("name", help="Name of the remote to add")
    subparser.add_argument("url", help="Url of the remote")
    subparser.add_argument("--insecure", dest="secure", action='store_false',
                           help="Allow insecure server connections when using SSL")
    subparser.add_argument("--index", action=OnceArgument, type=int,
                           help="Insert the remote at a specific position in the remote list")
    subparser.add_argument("-f", "--force", action='store_true',
                           help="Force the definition of the remote even if duplicated")
    subparser.add_argument("-ap", "--allowed-packages", action="append", default=None,
                           help="Add recipe reference pattern to list of allowed packages for "
                                "this remote")
    subparser.add_argument("-t", "--type", choices=[LOCAL_RECIPES_INDEX],
                           help="Define the remote type")
    subparser.add_argument("--recipes-only", action="store_true", default=False,
                           help="Disallow binary downloads from this remote, only recipes "
                                "will be downloaded")

    subparser.set_defaults(secure=True)
    args = parser.parse_args(*args)

    url_folder = make_abs_path(args.url)
    remote_type = args.type or (LOCAL_RECIPES_INDEX if os.path.isdir(url_folder) else None)
    url = url_folder if remote_type == LOCAL_RECIPES_INDEX else args.url
    r = Remote(args.name, url, args.secure, disabled=False, remote_type=remote_type,
               allowed_packages=args.allowed_packages,
               recipes_only=args.recipes_only)
    conan_api.remotes.add(r, force=args.force, index=args.index)


@conan_subcommand(formatters={"text": _print_remote_list})
def remote_remove(conan_api, parser, subparser, *args):
    """
    Remove remotes.
    """
    subparser.add_argument("remote", help="Name of the remote to remove. "
                                          "Accepts 'fnmatch' style wildcards.")  # to discuss
    args = parser.parse_args(*args)
    remotes = conan_api.remotes.remove(args.remote)
    cli_out_write("Removed remotes:")
    return remotes


@conan_subcommand()
def remote_update(conan_api, parser, subparser, *args):
    """
    Update a remote.
    """
    subparser.add_argument("remote", help="Name of the remote to update")
    subparser.add_argument("--url", action=OnceArgument, help="New url for the remote")
    subparser.add_argument("--secure", dest="secure", action='store_true',
                           help="Don't allow insecure server connections when using SSL")
    subparser.add_argument("--insecure", dest="secure", action='store_false',
                           help="Allow insecure server connections when using SSL")
    subparser.add_argument("--index", action=OnceArgument, type=int,
                           help="Insert the remote at a specific position in the remote list")
    subparser.add_argument("-ap", "--allowed-packages", action="append", default=None,
                           help="Add recipe reference pattern to the list of allowed packages "
                                "for this remote")
    subparser.add_argument("--recipes-only", default=None, const="True", nargs="?",
                           choices=["True", "False"],
                           help="Disallow binary downloads from this remote, only recipes will "
                                "be downloaded")

    subparser.set_defaults(secure=None)
    args = parser.parse_args(*args)
    if (args.url is None and args.secure is None and args.index is None and
            args.allowed_packages is None and args.recipes_only is None):
        subparser.error("Please add at least one argument to update")
    args.recipes_only = None if args.recipes_only is None else args.recipes_only == "True"
    conan_api.remotes.update(args.remote, args.url, args.secure, index=args.index,
                             allowed_packages=args.allowed_packages,
                             recipes_only=args.recipes_only)


@conan_subcommand()
def remote_rename(conan_api, parser, subparser, *args):
    """
    Rename a remote.
    """
    subparser.add_argument("remote", help="Current name of the remote")
    subparser.add_argument("new_name", help="New name for the remote")
    args = parser.parse_args(*args)
    conan_api.remotes.rename(args.remote, args.new_name)


@conan_subcommand(formatters={"text": _print_remote_list, "json": _print_remotes_json})
def remote_enable(conan_api, parser, subparser, *args):
    """
    Enable all the remotes matching a pattern.
    """
    subparser.add_argument("remote", help="Pattern of the remote/s to enable. "
                                          "The pattern uses 'fnmatch' style wildcards.")
    args = parser.parse_args(*args)
    return conan_api.remotes.enable(args.remote)


@conan_subcommand(formatters={"text": _print_remote_list, "json": _print_remotes_json})
def remote_disable(conan_api, parser, subparser, *args):
    """
    Disable all the remotes matching a pattern.
    """
    subparser.add_argument("remote", help="Pattern of the remote/s to disable. "
                                          "The pattern uses 'fnmatch' style wildcards.")
    args = parser.parse_args(*args)
    return conan_api.remotes.disable(args.remote)


# ### User related commands

@conan_subcommand(formatters={"text": _print_remote_user_list, "json": _print_remotes_users_json})
def remote_list_users(conan_api, parser, subparser, *args):  # noqa
    """
    List the users logged into all the remotes.
    """
    parser.parse_args(*args)
    remotes = conan_api.remotes.list()
    ret = OrderedDict()
    if not remotes:
        raise ConanException("No remotes defined")
    for r in remotes:
        ret[r.name] = conan_api.remotes.user_info(r)

    return ret


@conan_subcommand(formatters={"text": _print_remote_user_set, "json": _print_remotes_users_json})
def remote_login(conan_api, parser, subparser, *args):
    """
    Login into the specified remotes matching a pattern.
    """
    subparser.add_argument("remote", help="Pattern or name of the remote to login into. "
                                          "The pattern uses 'fnmatch' style wildcards.")
    subparser.add_argument("username", nargs="?", help='Username')
    subparser.add_argument("-p", "--password", nargs='?', type=str, action=OnceArgument,
                           help='User password. Use double quotes if password with spacing, '
                                'and escape quotes if existing. If empty, the password is '
                                'requested interactively (not exposed)')

    args = parser.parse_args(*args)
    remotes = conan_api.remotes.list(pattern=args.remote, only_enabled=False)
    if not remotes:
        raise ConanException("There are no remotes matching the '{}' pattern".format(args.remote))
    return conan_api.remotes.login(remotes, args.username, args.password)


@conan_subcommand(formatters={"text": _print_remote_user_set, "json": _print_remotes_users_json})
def remote_set_user(conan_api, parser, subparser, *args):
    """
    Associate a username with a remote matching a pattern without performing the authentication.
    """
    subparser.add_argument("remote", help="Pattern or name of the remote. "
                                          "The pattern uses 'fnmatch' style wildcards.")
    subparser.add_argument("username", help='Username')

    args = parser.parse_args(*args)
    remotes = conan_api.remotes.list(pattern=args.remote)
    if not remotes:
        raise ConanException("There are no remotes matching the '{}' pattern".format(args.remote))

    ret = OrderedDict()
    for r in remotes:
        previous_info = conan_api.remotes.user_info(r)
        if previous_info["user_name"] != args.username:
            conan_api.remotes.user_logout(r)
            conan_api.remotes.user_set(r, args.username)
        ret[r.name] = {"previous_info": previous_info, "info": conan_api.remotes.user_info(r)}
    return ret


@conan_subcommand(formatters={"text": _print_remote_user_set, "json": _print_remotes_users_json})
def remote_logout(conan_api, parser, subparser, *args):
    """
    Clear the existing credentials for the specified remotes matching a pattern.
    """
    subparser.add_argument("remote", help="Pattern or name of the remote to logout. "
                                          "The pattern uses 'fnmatch' style wildcards.")
    args = parser.parse_args(*args)
    remotes = conan_api.remotes.list(pattern=args.remote)
    if not remotes:
        raise ConanException("There are no remotes matching the '{}' pattern".format(args.remote))

    ret = OrderedDict()
    for r in remotes:
        previous_info = conan_api.remotes.user_info(r)
        conan_api.remotes.user_logout(r)
        info = conan_api.remotes.user_info(r)
        ret[r.name] = {"previous_info": previous_info, "info": info}
    return ret


def _print_auth(remotes):
    for remote_name, msg in remotes.items():
        if msg is None:
            cli_out_write(f"{remote_name}: No user defined")
        else:
            cli_out_write(f"{remote_name}:")
            for k, v in msg.items():
                cli_out_write(f"    {k}: {v}", fg=Color.BRIGHT_RED if k == "error" else Color.WHITE)


def _print_auth_json(results):
    cli_out_write(json.dumps(results))


@conan_subcommand(formatters={"text": _print_auth, "json": _print_auth_json})
def remote_auth(conan_api, parser, subparser, *args):
    """
    Authenticate in the defined remotes. Use CONAN_LOGIN_USERNAME* and CONAN_PASSWORD* variables
    if available.
    Ask for username and password interactively in case (re-)authentication is required and there are
    no CONAN_LOGIN* and CONAN_PASSWORD* variables available which could be used.
    Usually you'd use this method over conan remote login for scripting which needs to run in CI
    and locally.
    """
    subparser.add_argument("remote", help="Pattern or name of the remote/s to authenticate against."
                                          " The pattern uses 'fnmatch' style wildcards.")
    subparser.add_argument("--with-user", action="store_true",
                           help="Only try to auth in those remotes that already "
                                "have a username or a CONAN_LOGIN_USERNAME* env-var defined")
    subparser.add_argument("--force", action="store_true",
                           help="Force authentication for anonymous-enabled repositories. "
                                "Can be used for force authentication in case your Artifactory "
                                "instance has anonymous access enabled and Conan would not ask "
                                "for username and password even for non-anonymous repositories "
                                "if not yet authenticated.")
    args = parser.parse_args(*args)
    remotes = conan_api.remotes.list(pattern=args.remote)
    if not remotes:
        raise ConanException("There are no remotes matching the '{}' pattern".format(args.remote))

    results = {}
    for r in remotes:
        try:
            results[r.name] = {"user": conan_api.remotes.user_auth(r, args.with_user, args.force)}
        except Exception as e:
            results[r.name] = {"error": str(e)}
    return results


@conan_command(group="Consumer")
def remote(conan_api, parser, *args):  # noqa
    """
    Manage the remote list and the users authenticated on them.
    """
