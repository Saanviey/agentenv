from utils import cents_to_dollars


def format_price(cents):
    return f"${cents_to_dollars(cents):.2f}"
