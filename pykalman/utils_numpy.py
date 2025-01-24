"""Utility functions to handle numpy 2."""

import pkg_resources


def _check_numpy_2():
    try:
        # Get the installed version of numpy
        numpy_version = pkg_resources.get_distribution("numpy").version

        # Split version into parts for checking
        version_parts = numpy_version.split(".")
        major_version = int(version_parts[0])

        # Check if major version is 2 and consider RC versions
        return major_version == 2

    except Exception:
        return False


numpy2 = _check_numpy_2()


def newbyteorder(arr, new_order):
    """Change the byte order of an array.

    Parameters
    ----------
    arr : ndarray
        Input array.
    new_order : str
        Byte order to force.

    Returns
    -------
    arr : ndarray
        Array with new byte order.
    """
    if numpy2:
        return arr.view(arr.dtype.newbyteorder(new_order))
    else:
        return arr.newbyteorder(new_order)
