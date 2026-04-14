from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version("pycistarget")
except PackageNotFoundError:
    pass
