"""Looks up the installed version of this package."""

from datetime import date
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("trendify2")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

__license__ = "MIT"
__summary__ = "Tools for generating data products, storing, and interacting with them."
__copyright__ = f"© {date.today().year} Talbot Knighton and David Gable"
__url__ = "https://github.com/Hydrowelder/trendify"
