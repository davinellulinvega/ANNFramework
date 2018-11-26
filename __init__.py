# Only define the __all__ variable to control the damages done by any `from PrimEmoArch import *` statement.
# It only gives access to the top most concept of a Network, which is configured via the settings.py configuration file, while the rest is
# hidden away and has to be imported separately.
__all__ = ['Network']
