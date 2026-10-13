"""Utility functions to handle numpy 2."""


def newbyteorder(arr, new_order):
    """Change the byte order of an array while preserving its values.

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
    return arr.astype(arr.dtype.newbyteorder(new_order), copy=False)
