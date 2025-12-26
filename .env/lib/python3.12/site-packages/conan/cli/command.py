import argparse
import os
import textwrap
from contextlib import redirect_stdout

from conan.api.output import ConanOutput
from conan.errors import ConanException


class OnceArgument(argparse.Action):
    """Allows declaring a parameter that can have only one value, by default argparse takes the
    latest declared and it's very confusing.
    """

    def __call__(self, parser, namespace, values, option_string=None):
        if getattr(namespace, self.dest) is not None and self.default is None:
            msg = '{o} can only be specified once'.format(o=option_string)
            raise argparse.ArgumentError(None, msg)
        setattr(namespace, self.dest, values)


class SmartFormatter(argparse.HelpFormatter):

    def _fill_text(self, text, width, indent):
        text = textwrap.dedent(text)
        return ''.join(indent + line for line in text.splitlines(True))


class BaseConanCommand:
    def __init__(self, method, formatters=None):
        self._formatters = {"text": lambda x: None}
        self._method = method
        self._name = None
        if formatters:
            for kind, action in formatters.items():
                if callable(action):
                    self._formatters[kind] = action
                else:
                    raise ConanException("Invalid formatter for {}. The formatter must be"
                                         "a valid function".format(kind))
        if method.__doc__:
            self._doc = method.__doc__
        else:
            raise ConanException("No documentation string defined for command: '{}'. Conan "
                                 "commands should provide a documentation string explaining "
                                 "its use briefly.".format(self._name))

    @staticmethod
    def _init_core_options(parser):
        # Define possible levels, including "" for verbose
        possible_levels = list(ConanOutput.valid_log_levels().keys())
        possible_levels.pop(possible_levels.index(None))
        parser.add_argument("-v", default="status", nargs='?',
                            help="Level of detail of the output. Valid options from less verbose "
                                 "to more verbose: -vquiet, -verror, -vwarning, -vnotice, -vstatus, "
                                 "-v or -vverbose, -vv or -vdebug, -vvv or -vtrace",
                            choices=possible_levels,
                            )
        parser.add_argument("-cc", "--core-conf", action="append",
                            help="Define core configuration, overwriting global.conf "
                                 "values. E.g.: -cc core:non_interactive=True")

    @property
    def _help_formatters(self):
        """
        Formatters that are shown as available in help, 'text' formatter
        should not appear
        """
        return [formatter for formatter in self._formatters if formatter != "text"]

    def _init_formatters(self, parser):
        formatters = self._help_formatters
        if formatters:
            help_message = "Select the output format: {}".format(", ".join(formatters))
            parser.add_argument('-f', '--format', action=OnceArgument, help=help_message)

        parser.add_argument("--out-file", action=OnceArgument,
                            help="Write the output of the command to the specified file instead of "
                                 "stdout.")

    @property
    def name(self):
        return self._name

    @property
    def method(self):
        return self._method

    @property
    def doc(self):
        return self._doc

    def _format(self, parser, info, *args):
        parser_args, _ = parser.parse_known_args(*args)

        formatarg = getattr(parser_args, "format", None) or "text"
        out_file = getattr(parser_args, "out_file", None)

        try:
            formatter = self._formatters[formatarg]
        except KeyError:
            raise ConanException("{} is not a known format. Supported formatters are: {}".format(
                formatarg, ", ".join(self._help_formatters)))

        if out_file:
            if os.path.dirname(out_file):
                os.makedirs(os.path.dirname(out_file), exist_ok=True)
            with open(out_file, 'w') as f:
                with redirect_stdout(f):
                    formatter(info)
            ConanOutput().info(f"Formatted output saved to '{out_file}'")
        else:
            formatter(info)

    @staticmethod
    def _dispatch_errors(info):
        if info and isinstance(info, dict):
            if info.get("conan_error"):
                e = info["conan_error"]
                # Storing and launching an exception is better than the string, as it keeps
                # the correct backtrace for debugging.
                if isinstance(e, Exception):
                    raise e
                raise ConanException(e)
            if info.get("conan_warning"):
                ConanOutput().warning(info["conan_warning"])


class ConanArgumentParser(argparse.ArgumentParser):

    def __init__(self, conan_api, *args, **kwargs):
        self._conan_api = conan_api
        super().__init__(*args, **kwargs)

    def parse_args(self, args=None, namespace=None):
        args = super().parse_args(args)
        ConanOutput.define_log_level(args.v)
        if getattr(args, "lockfile_packages", None):
            ConanOutput().error("The --lockfile-packages arg is private and shouldn't be used")

        if args.core_conf:
            self._conan_api._api_helpers.set_core_confs(args.core_conf)  # noqa

        global_conf = self._conan_api._api_helpers.global_conf  # noqa
        # TODO: This might be even better moved to the ConanAPI so users without doing custom
        #  commands can benefit from it
        ConanOutput.set_warnings_as_errors(global_conf.get("core:warnings_as_errors",
                                                           default=[], check_type=list))
        ConanOutput.define_silence_warnings(global_conf.get("core:skip_warnings",
                                                            default=[], check_type=list))
        return args


class ConanCommand(BaseConanCommand):
    def __init__(self, method, group=None, formatters=None):
        super().__init__(method, formatters=formatters)
        self._subcommands = {}
        self._group = group or "Other"
        self._name = method.__name__.replace("_", "-")
        self._prog = self._name

    def add_subcommand(self, subcommand):
        subcommand.set_name(self.name)
        self._subcommands[subcommand.name] = subcommand

    def run_cli(self, conan_api, *args):
        parser = ConanArgumentParser(conan_api, description=self._doc,
                                     prog="conan {}".format(self._prog),
                                     formatter_class=SmartFormatter)
        self._init_formatters(parser)
        self._init_core_options(parser)
        parser.suggest_on_error = True
        info = self._method(conan_api, parser, *args)
        if not self._subcommands:
            return info

        subcommand_parser = parser.add_subparsers(dest='subcommand', help='sub-command help')
        subcommand_parser.required = True

        subcmd = args[0][0]
        try:
            sub = self._subcommands[subcmd]
        except (KeyError, IndexError):  # display help
            raise ConanException(f"Sub command {subcmd} does not exist")
        else:
            sub.set_parser(subcommand_parser, conan_api)
            return sub.run_cli(conan_api, parser, *args)

    def run(self, conan_api, *args):
        parser = ConanArgumentParser(conan_api, description=self._doc,
                                     prog="conan {}".format(self._prog),
                                     formatter_class=SmartFormatter)
        self._init_formatters(parser)
        self._init_core_options(parser)
        parser.suggest_on_error = True

        info = self._method(conan_api, parser, *args)

        if not self._subcommands:
            self._format(parser, info, *args)
        else:
            subcommand_parser = parser.add_subparsers(dest='subcommand', help='sub-command help')
            subcommand_parser.required = True

            try:
                sub = self._subcommands[args[0][0]]
            except (KeyError, IndexError):  # display help
                for sub in self._subcommands.values():
                    sub.set_parser(subcommand_parser, conan_api)
                parser.parse_args(*args)
            else:
                sub.set_parser(subcommand_parser, conan_api)
                sub.run(conan_api, parser, *args)
        self._dispatch_errors(info)

    @property
    def group(self):
        return self._group


class ConanSubCommand(BaseConanCommand):
    def __init__(self, method, formatters=None):
        super().__init__(method, formatters=formatters)
        self._parser = None
        self._subcommand_name = method.__name__.replace('_', '-')

    def run_cli(self, conan_api, parent_parser, *args):
        return self._method(conan_api, parent_parser, self._parser, *args)

    def run(self, conan_api, parent_parser, *args):
        info = self._method(conan_api, parent_parser, self._parser, *args)
        # It is necessary to do it after calling the "method" otherwise parser not complete
        self._format(parent_parser, info, *args)
        self._dispatch_errors(info)

    def set_name(self, parent_name):
        self._name = self._subcommand_name.replace(f'{parent_name}-', '', 1)

    def set_parser(self, subcommand_parser, conan_api):
        self._parser = subcommand_parser.add_parser(self._name, conan_api=conan_api, help=self._doc)
        self._parser.description = self._doc
        self._init_formatters(self._parser)
        self._init_core_options(self._parser)
        self._parser.suggest_on_error = True


def conan_command(group=None, formatters=None):
    return lambda f: ConanCommand(f, group, formatters=formatters)


def conan_subcommand(formatters=None):
    return lambda f: ConanSubCommand(f, formatters=formatters)
