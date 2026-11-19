def merge(defaults, overrides):
    """Return a NEW dict: defaults updated with overrides. Must not modify either input."""
    defaults.update(overrides)
    return defaults
