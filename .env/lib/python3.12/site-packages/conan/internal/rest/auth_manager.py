"""
Collaborate with RestApiClient to make remote anonymous and authenticated calls.
Uses user_input to request user's login and password and obtain a token for calling authenticated
methods if receives AuthenticationException from RestApiClient.


Flow:
    Directly invoke a REST method in RestApiClient, example: get_conan.
    if receives AuthenticationException (not open method) will ask user for login and password
    (with LOGIN_RETRIES retries) and retry to call with the new token.
"""

from conan.api.output import ConanOutput
from conan.internal.rest.remote_credentials import RemoteCredentials
from conan.internal.rest.rest_client import RestApiClient
from conan.internal.errors import AuthenticationException, ForbiddenException
from conan.errors import ConanException

LOGIN_RETRIES = 3


class _RemoteCreds:
    def __init__(self, localdb):
        self._localdb = localdb

    def get(self, remote, msg=True):
        creds = getattr(remote, "_creds", None)
        if creds is None:
            user, token, _ = self._localdb.get_login(remote.url)
            creds = user, token
            if msg:
                usermsg = f"with user '{user}'" if user else "anonymously"
                ConanOutput().info(f"Connecting to remote '{remote.name}' {usermsg}")
            setattr(remote, "_creds", creds)
        return creds

    def set(self, remote, user, token):
        setattr(remote, "_creds", (user, token))
        ConanOutput().success(f"Authenticated in remote '{remote.name}' with user '{user}'")
        self._localdb.store(user, token, None, remote.url)


class ConanApiAuthManager:

    def __init__(self, requester, cache_folder, localdb, global_conf):
        self._requester = requester
        self._creds = _RemoteCreds(localdb)
        self._global_conf = global_conf
        self._cache_folder = cache_folder

    def call_rest_api_method(self, remote, method_name, *args, **kwargs):
        """Handles AuthenticationException and request user to input a user and a password"""
        user, token = self._creds.get(remote, msg=(method_name != "authenticate"))
        rest_client = RestApiClient(remote, token, self._requester, self._global_conf)

        if method_name == "authenticate":
            return self._authenticate(rest_client, remote, *args, **kwargs)

        try:
            ret = getattr(rest_client, method_name)(*args, **kwargs)
            return ret
        except ForbiddenException as e:
            raise ForbiddenException(f"Permission denied for user: '{user}': {e}")
        except AuthenticationException:
            # User valid but not enough permissions
            # token is None when you change user with user command
            # Anonymous is not enough, ask for a user
            ConanOutput().info(f"Remote '{remote.name}' needs authentication, obtaining credentials")
            if self._get_credentials_and_authenticate(rest_client, user, remote):
                return self.call_rest_api_method(remote, method_name, *args, **kwargs)

    def _get_credentials_and_authenticate(self, rest_client, user, remote):
        """Try LOGIN_RETRIES to obtain a password from user input for which
        we can get a valid token from api_client. If a token is returned,
        credentials are stored in localdb and rest method is called"""
        creds = RemoteCredentials(self._cache_folder, self._global_conf)
        for _ in range(LOGIN_RETRIES):
            input_user, input_password, interactive = creds.auth(remote)
            try:
                self._authenticate(rest_client, remote, input_user, input_password)
            except AuthenticationException:
                out = ConanOutput()
                if user is None:
                    out.error('Wrong user or password', error_type="exception")
                else:
                    out.error(f'Wrong password for user "{user}"', error_type="exception")
                if not interactive:
                    raise AuthenticationException(f"Authentication error in remote '{remote.name}'")
            else:
                return True
        raise AuthenticationException("Too many failed login attempts, bye!")

    def _authenticate(self, rest_client, remote, user, password):
        try:
            token = rest_client.authenticate(user, password)
        except UnicodeDecodeError:
            raise ConanException("Password contains not allowed symbols")

        # Store result in DB
        self._creds.set(remote, user, token)
