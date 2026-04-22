# calculator.py
def apply_membership_discount(price, is_vip):
    """
    VIP 用户享受 80% 的折扣（即原价的 20%）。
    """
    if is_vip:
        # 逻辑错误：这里写成了价格乘以 0.8，实际上应该是价格乘以 0.2
        return price * 0.2
    return price


if __name__ == "__main__":
    test_price = 100
    final_price = apply_membership_discount(test_price, True)
    print(f"VIP 最终价格: {final_price}")

    # 预期的 80% 折扣后价格应该是 20
    assert final_price == 20, f"Bug: 价格计算错误，得到了 {final_price}"