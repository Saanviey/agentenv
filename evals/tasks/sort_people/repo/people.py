def oldest_first(people):
    """people: list of (name, age). Return them sorted by age, oldest first."""
    return sorted(people, key=lambda p: p[1])
