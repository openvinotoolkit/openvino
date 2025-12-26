import getpass
import sys

from conan.errors import ConanException


class UserInput:
    """Class to interact with the user, used to show messages and ask for information"""

    def __init__(self, non_interactive):
        """
        Params:
            ins: input stream
            out: ConanOutput, should have "write" method
        """
        self._ins = sys.stdin
        # FIXME: circular include, move "color_enabled" function to better location
        from conan.api.output import ConanOutput
        self._out = ConanOutput()
        self._interactive = not non_interactive

    def _raise_if_non_interactive(self):
        if not self._interactive:
            raise ConanException("Conan interactive mode disabled")

    def raw_input(self):
        self._raise_if_non_interactive()
        return input()

    def request_login(self, remote_name, username=None):
        """Request user to input their name and password
        :param remote_name:
        :param username If username is specified it only request password"""
        self._raise_if_non_interactive()
        if not username:
            self._out.login_msg(f"Remote '{remote_name}' username: ")
            username = self.get_username()

        self._out.login_msg("Please enter a password for "
                            f"user '{username}' on remote '{remote_name}': ")
        try:
            pwd = self.get_password()
        except ConanException:
            raise
        except Exception as e:
            raise ConanException('Cancelled pass %s' % e)
        return username, pwd

    def get_username(self):
        """Overridable for testing purpose"""
        return self.raw_input()

    @staticmethod
    def get_password():
        """Overridable for testing purpose"""
        try:
            return getpass.getpass("")
        except BaseException:  # For KeyboardInterrupt too
            raise ConanException("Interrupted interactive password input")

    def request_string(self, msg, default_value=None):
        """Request user to input a msg
        :param default_value:
        :param msg Name of the msg
        """
        self._raise_if_non_interactive()

        if default_value:
            self._out.write('%s (%s): ' % (msg, default_value))
        else:
            self._out.write('%s: ' % msg)
        s = self._ins.readline().replace("\n", "")
        if default_value is not None and s == '':
            return default_value
        return s

    def request_boolean(self, msg, default_option=None):
        """Request user to input a boolean"""
        ret = None
        while ret is None:
            if default_option:
                s = self.request_string("%s (YES/no)" % msg)
            elif default_option is False:
                s = self.request_string("%s (NO/yes)" % msg)
            else:
                s = self.request_string("%s (yes/no)" % msg)
            if default_option is not None and s == '':
                return default_option
            if s.lower() in ['yes', 'y']:
                ret = True
            elif s.lower() in ['no', 'n']:
                ret = False
            else:
                self._out.error(f"{s} is not a valid answer")
        return ret
