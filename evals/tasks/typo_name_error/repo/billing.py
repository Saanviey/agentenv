def total_price(prices, tax_rate):
    subtotal = sum(prices)
    return round(subtotl * (1 + tax_rate), 2)
