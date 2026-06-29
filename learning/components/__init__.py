"""
Importing this package registers every functional component.

The env only needs the registries populated; importing these modules for their
side effect (the @REGISTRY.register decorators) is how that happens.  Add a new
component module here so it is discoverable by name from the config.
"""

from learning.components import actions        # noqa: F401
from learning.components import commands        # noqa: F401
from learning.components import observations    # noqa: F401
from learning.components import rewards         # noqa: F401
from learning.components import terminations    # noqa: F401
