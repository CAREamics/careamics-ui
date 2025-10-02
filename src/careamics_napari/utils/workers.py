import platform


def get_num_workers():
    """Utility function to set dataloader's num_workers based on OS."""
    if platform.system() == "Windows" or platform.system() == "Darwin":
        return 0
    else:
        return 3  # or any other number suitable for your system
