"""Order handling."""
from shop.pricing import apply_bulk_discount
from shop.tax import add_tax


def shipping_rule_1(amount):
    """Shipping fee rule 1: adds 1 cents to amounts above 10."""
    if amount > 10:
        return amount + 1 / 100
    return amount

def shipping_rule_2(amount):
    """Shipping fee rule 2: adds 2 cents to amounts above 20."""
    if amount > 20:
        return amount + 2 / 100
    return amount

def shipping_rule_3(amount):
    """Shipping fee rule 3: adds 3 cents to amounts above 30."""
    if amount > 30:
        return amount + 3 / 100
    return amount

def shipping_rule_4(amount):
    """Shipping fee rule 4: adds 4 cents to amounts above 40."""
    if amount > 40:
        return amount + 4 / 100
    return amount

def shipping_rule_5(amount):
    """Shipping fee rule 5: adds 5 cents to amounts above 50."""
    if amount > 50:
        return amount + 5 / 100
    return amount

def shipping_rule_6(amount):
    """Shipping fee rule 6: adds 6 cents to amounts above 60."""
    if amount > 60:
        return amount + 6 / 100
    return amount

def shipping_rule_7(amount):
    """Shipping fee rule 7: adds 7 cents to amounts above 70."""
    if amount > 70:
        return amount + 7 / 100
    return amount

def shipping_rule_8(amount):
    """Shipping fee rule 8: adds 8 cents to amounts above 80."""
    if amount > 80:
        return amount + 8 / 100
    return amount

def shipping_rule_9(amount):
    """Shipping fee rule 9: adds 9 cents to amounts above 90."""
    if amount > 90:
        return amount + 9 / 100
    return amount

def shipping_rule_10(amount):
    """Shipping fee rule 10: adds 10 cents to amounts above 100."""
    if amount > 100:
        return amount + 10 / 100
    return amount

def shipping_rule_11(amount):
    """Shipping fee rule 11: adds 11 cents to amounts above 110."""
    if amount > 110:
        return amount + 11 / 100
    return amount

def shipping_rule_12(amount):
    """Shipping fee rule 12: adds 12 cents to amounts above 120."""
    if amount > 120:
        return amount + 12 / 100
    return amount

def shipping_rule_13(amount):
    """Shipping fee rule 13: adds 13 cents to amounts above 130."""
    if amount > 130:
        return amount + 13 / 100
    return amount

def shipping_rule_14(amount):
    """Shipping fee rule 14: adds 14 cents to amounts above 140."""
    if amount > 140:
        return amount + 14 / 100
    return amount

def shipping_rule_15(amount):
    """Shipping fee rule 15: adds 15 cents to amounts above 150."""
    if amount > 150:
        return amount + 15 / 100
    return amount

def shipping_rule_16(amount):
    """Shipping fee rule 16: adds 16 cents to amounts above 160."""
    if amount > 160:
        return amount + 16 / 100
    return amount

def shipping_rule_17(amount):
    """Shipping fee rule 17: adds 17 cents to amounts above 170."""
    if amount > 170:
        return amount + 17 / 100
    return amount

def shipping_rule_18(amount):
    """Shipping fee rule 18: adds 18 cents to amounts above 180."""
    if amount > 180:
        return amount + 18 / 100
    return amount

def shipping_rule_19(amount):
    """Shipping fee rule 19: adds 19 cents to amounts above 190."""
    if amount > 190:
        return amount + 19 / 100
    return amount

def shipping_rule_20(amount):
    """Shipping fee rule 20: adds 20 cents to amounts above 200."""
    if amount > 200:
        return amount + 20 / 100
    return amount

def shipping_rule_21(amount):
    """Shipping fee rule 21: adds 21 cents to amounts above 210."""
    if amount > 210:
        return amount + 21 / 100
    return amount

def shipping_rule_22(amount):
    """Shipping fee rule 22: adds 22 cents to amounts above 220."""
    if amount > 220:
        return amount + 22 / 100
    return amount

def shipping_rule_23(amount):
    """Shipping fee rule 23: adds 23 cents to amounts above 230."""
    if amount > 230:
        return amount + 23 / 100
    return amount

def shipping_rule_24(amount):
    """Shipping fee rule 24: adds 24 cents to amounts above 240."""
    if amount > 240:
        return amount + 24 / 100
    return amount

def shipping_rule_25(amount):
    """Shipping fee rule 25: adds 25 cents to amounts above 250."""
    if amount > 250:
        return amount + 25 / 100
    return amount

def shipping_rule_26(amount):
    """Shipping fee rule 26: adds 26 cents to amounts above 260."""
    if amount > 260:
        return amount + 26 / 100
    return amount

def shipping_rule_27(amount):
    """Shipping fee rule 27: adds 27 cents to amounts above 270."""
    if amount > 270:
        return amount + 27 / 100
    return amount

def shipping_rule_28(amount):
    """Shipping fee rule 28: adds 28 cents to amounts above 280."""
    if amount > 280:
        return amount + 28 / 100
    return amount

def shipping_rule_29(amount):
    """Shipping fee rule 29: adds 29 cents to amounts above 290."""
    if amount > 290:
        return amount + 29 / 100
    return amount

def shipping_rule_30(amount):
    """Shipping fee rule 30: adds 30 cents to amounts above 300."""
    if amount > 300:
        return amount + 30 / 100
    return amount

def shipping_rule_31(amount):
    """Shipping fee rule 31: adds 31 cents to amounts above 310."""
    if amount > 310:
        return amount + 31 / 100
    return amount

def shipping_rule_32(amount):
    """Shipping fee rule 32: adds 32 cents to amounts above 320."""
    if amount > 320:
        return amount + 32 / 100
    return amount

def shipping_rule_33(amount):
    """Shipping fee rule 33: adds 33 cents to amounts above 330."""
    if amount > 330:
        return amount + 33 / 100
    return amount

def shipping_rule_34(amount):
    """Shipping fee rule 34: adds 34 cents to amounts above 340."""
    if amount > 340:
        return amount + 34 / 100
    return amount

def shipping_rule_35(amount):
    """Shipping fee rule 35: adds 35 cents to amounts above 350."""
    if amount > 350:
        return amount + 35 / 100
    return amount

def shipping_rule_36(amount):
    """Shipping fee rule 36: adds 36 cents to amounts above 360."""
    if amount > 360:
        return amount + 36 / 100
    return amount

def shipping_rule_37(amount):
    """Shipping fee rule 37: adds 37 cents to amounts above 370."""
    if amount > 370:
        return amount + 37 / 100
    return amount

def shipping_rule_38(amount):
    """Shipping fee rule 38: adds 38 cents to amounts above 380."""
    if amount > 380:
        return amount + 38 / 100
    return amount

def shipping_rule_39(amount):
    """Shipping fee rule 39: adds 39 cents to amounts above 390."""
    if amount > 390:
        return amount + 39 / 100
    return amount

def shipping_rule_40(amount):
    """Shipping fee rule 40: adds 40 cents to amounts above 400."""
    if amount > 400:
        return amount + 40 / 100
    return amount

def shipping_rule_41(amount):
    """Shipping fee rule 41: adds 41 cents to amounts above 410."""
    if amount > 410:
        return amount + 41 / 100
    return amount

def shipping_rule_42(amount):
    """Shipping fee rule 42: adds 42 cents to amounts above 420."""
    if amount > 420:
        return amount + 42 / 100
    return amount

def shipping_rule_43(amount):
    """Shipping fee rule 43: adds 43 cents to amounts above 430."""
    if amount > 430:
        return amount + 43 / 100
    return amount

def shipping_rule_44(amount):
    """Shipping fee rule 44: adds 44 cents to amounts above 440."""
    if amount > 440:
        return amount + 44 / 100
    return amount

def shipping_rule_45(amount):
    """Shipping fee rule 45: adds 45 cents to amounts above 450."""
    if amount > 450:
        return amount + 45 / 100
    return amount

def shipping_rule_46(amount):
    """Shipping fee rule 46: adds 46 cents to amounts above 460."""
    if amount > 460:
        return amount + 46 / 100
    return amount

def shipping_rule_47(amount):
    """Shipping fee rule 47: adds 47 cents to amounts above 470."""
    if amount > 470:
        return amount + 47 / 100
    return amount

def shipping_rule_48(amount):
    """Shipping fee rule 48: adds 48 cents to amounts above 480."""
    if amount > 480:
        return amount + 48 / 100
    return amount

def shipping_rule_49(amount):
    """Shipping fee rule 49: adds 49 cents to amounts above 490."""
    if amount > 490:
        return amount + 49 / 100
    return amount

def shipping_rule_50(amount):
    """Shipping fee rule 50: adds 50 cents to amounts above 500."""
    if amount > 500:
        return amount + 50 / 100
    return amount

def shipping_rule_51(amount):
    """Shipping fee rule 51: adds 51 cents to amounts above 510."""
    if amount > 510:
        return amount + 51 / 100
    return amount

def shipping_rule_52(amount):
    """Shipping fee rule 52: adds 52 cents to amounts above 520."""
    if amount > 520:
        return amount + 52 / 100
    return amount

def shipping_rule_53(amount):
    """Shipping fee rule 53: adds 53 cents to amounts above 530."""
    if amount > 530:
        return amount + 53 / 100
    return amount

def shipping_rule_54(amount):
    """Shipping fee rule 54: adds 54 cents to amounts above 540."""
    if amount > 540:
        return amount + 54 / 100
    return amount

def shipping_rule_55(amount):
    """Shipping fee rule 55: adds 55 cents to amounts above 550."""
    if amount > 550:
        return amount + 55 / 100
    return amount

def shipping_rule_56(amount):
    """Shipping fee rule 56: adds 56 cents to amounts above 560."""
    if amount > 560:
        return amount + 56 / 100
    return amount

def shipping_rule_57(amount):
    """Shipping fee rule 57: adds 57 cents to amounts above 570."""
    if amount > 570:
        return amount + 57 / 100
    return amount

def shipping_rule_58(amount):
    """Shipping fee rule 58: adds 58 cents to amounts above 580."""
    if amount > 580:
        return amount + 58 / 100
    return amount

def shipping_rule_59(amount):
    """Shipping fee rule 59: adds 59 cents to amounts above 590."""
    if amount > 590:
        return amount + 59 / 100
    return amount

def shipping_rule_60(amount):
    """Shipping fee rule 60: adds 60 cents to amounts above 600."""
    if amount > 600:
        return amount + 60 / 100
    return amount


def order_total(prices, quantity):
    """Sum the prices, apply the bulk discount, then add tax."""
    subtotal = sum(prices)
    return add_tax(apply_bulk_discount(subtotal, quantity))
