"""
Bài tập thực hành Buổi 3 — Luật Hebb – Hopfield khử nhiễu ký hiệu giao thông
Học phần 111135 — Đại học Lạc Hồng

Bối cảnh: Camera giao thông chụp ký hiệu trên bảng báo nhưng ảnh bị nhiễu vài
điểm ảnh (bụi, mưa, ánh sáng). Dùng mạng Hopfield để phục hồi đúng ký hiệu đã
lưu trong ba ký hiệu: UP (Đi thẳng), LEFT (Rẽ trái), CROSS (Cấm/Dừng).

Hướng dẫn:
- Điền code vào các chỗ có "TODO".
- Chạy trực tiếp bằng: python bai-tap/buoi3_hebb_hopfield.py
- Mỗi bài có hàm kiểm tra tự động (assert); nếu chạy xong không có lỗi
  AssertionError là bạn đã làm đúng.

Quy ước đánh số 16 ô (trái -> phải, trên -> dưới):
    1  2  3  4
    5  6  7  8
    9  10 11 12
    13 14 15 16
Sáng = +1, tối = -1.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Dữ liệu chung — KHÔNG cần sửa
# ---------------------------------------------------------------------------

PATTERNS = {
    "UP": np.array([
        -1, 1, 1,-1,
         1, 1, 1, 1,
        -1, 1, 1,-1,
        -1, 1, 1,-1,
    ], dtype=float),
    "LEFT": np.array([
         1,-1,-1,-1,
         1, 1, 1, 1,
         1,-1,-1,-1,
         1,-1,-1,-1,
    ], dtype=float),
    "CROSS": np.array([
         1,-1,-1, 1,
        -1, 1, 1,-1,
        -1, 1, 1,-1,
         1,-1,-1, 1,
    ], dtype=float),
}


def sign(value: float, current: float) -> float:
    """Hàm dấu: +1 nếu value>0, -1 nếu value<0, giữ nguyên current nếu value=0."""
    if value > 0:
        return 1.0
    if value < 0:
        return -1.0
    return current


def energy(W: np.ndarray, x: np.ndarray) -> float:
    """Năng lượng Hopfield: E(x) = -1/2 * x^T W x."""
    return -0.5 * x @ W @ x


def build_weight_matrix(patterns: dict) -> np.ndarray:
    """Dựng ma trận trọng số W (16x16) bằng luật Hebb từ 3 ký hiệu UP, LEFT, CROSS."""
    xiUP, xiLEFT, xiCROSS = patterns["UP"], patterns["LEFT"], patterns["CROSS"]
    W = np.outer(xiUP, xiUP) + np.outer(xiLEFT, xiLEFT) + np.outer(xiCROSS, xiCROSS)
    np.fill_diagonal(W, 0)
    return W


W = build_weight_matrix(PATTERNS)


def get_value(name: str, index_1based: int) -> float:
    """Lấy giá trị (+1/-1) của một ô trên một ký hiệu, index tính từ 1."""
    return PATTERNS[name][index_1based - 1]


# ---------------------------------------------------------------------------
# Bài A — Luật Hebb: tính tay bốn trọng số (2 điểm)
# ---------------------------------------------------------------------------

def bai_a() -> dict:
    """Trả về dict {(i, j): w_ij} cho 4 cặp ô yêu cầu."""
    pairs = [(1, 13), (1, 4), (1, 5), (1, 2)]
    result = {}
    for i, j in pairs:
        tich_UP = get_value("UP", i) * get_value("UP", j)
        tich_LEFT = get_value("LEFT", i) * get_value("LEFT", j)
        tich_CROSS = get_value("CROSS", i) * get_value("CROSS", j)
        w_ij = tich_UP + tich_LEFT + tich_CROSS
        result[(i, j)] = w_ij
    return result


def check_bai_a() -> None:
    result = bai_a()
    expected = {(1, 13): 3, (1, 4): 1, (1, 5): -1, (1, 2): -3}
    for pair, expected_w in expected.items():
        actual_w = result[pair]
        assert actual_w == expected_w, (
            f"Bài A sai tại w{pair}: bạn tính {actual_w}, đáp án đúng {expected_w}"
        )
        i, j = pair
        assert actual_w == W[i - 1, j - 1], f"Bài A: w{pair} không khớp với W thật"
    print("Bài A: OK — cả 4 trọng số Hebb đều đúng.")


# ---------------------------------------------------------------------------
# Bài B — Hopfield: vá ký hiệu Rẽ trái (LEFT) bị nhiễu 2 điểm (3 điểm)
# ---------------------------------------------------------------------------

def bai_b() -> np.ndarray:
    """Tạo input nhiễu, tính net, vá tuần tự ô 6 rồi ô 14. Trả về x sau khi vá."""
    x0 = PATTERNS["LEFT"].copy()
    x0[5] *= -1  # Lật ô 6
    x0[13] *= -1 # Lật ô 14

    net_6 = W[5] @ x0
    net_14 = W[13] @ x0
    print(f"  net_6 = {net_6}, net_14 = {net_14}")

    x_result = x0.copy()
    x_result[5] = sign(W[5] @ x_result, x_result[5])
    x_result[13] = sign(W[13] @ x_result, x_result[13])
    return x_result


def check_bai_b() -> None:
    x_result = bai_b()
    assert np.array_equal(x_result, PATTERNS["LEFT"]), (
        "Bài B sai: sau khi vá, x chưa khớp với ký hiệu LEFT sạch."
    )
    print("Bài B: OK — vá đúng cả hai ô, ảnh trở về ký hiệu LEFT sạch.")


# ---------------------------------------------------------------------------
# Bài C — Năng lượng: theo dõi E khi vá ký hiệu Cấm (CROSS) (3 điểm)
# ---------------------------------------------------------------------------

def bai_c() -> tuple:
    """Trả về (E_truoc, E_sau, x_sau_khi_va) khi vá ô 6 của CROSS bị lật."""
    x_c = PATTERNS["CROSS"].copy()
    x_c[5] *= -1

    E_before = energy(W, x_c)

    net_6 = W[5] @ x_c
    x_c[5] = sign(net_6, x_c[5])

    E_after = energy(W, x_c)

    return E_before, E_after, x_c


def check_bai_c() -> None:
    E_before, E_after, x_c = bai_c()
    assert E_before == -92, f"Bài C sai: E trước khi vá phải là -92, bạn tính {E_before}"
    assert E_after == -114, f"Bài C sai: E sau khi vá phải là -114, bạn tính {E_after}"
    assert E_after <= E_before, "Bài C sai: năng lượng không được tăng sau khi vá."
    assert np.array_equal(x_c, PATTERNS["CROSS"]), (
        "Bài C sai: x sau khi vá chưa khớp ký hiệu CROSS sạch."
    )
    print(f"Bài C: OK — E giảm từ {E_before} xuống {E_after}, khớp ký hiệu CROSS sạch.")


# ---------------------------------------------------------------------------
# Bài D — Tổng hợp: Hamming hay Hopfield? (2 điểm)
# ---------------------------------------------------------------------------

def bai_d() -> dict:
    """Trả về dict {so_thu_tu: "Hamming" | "Hopfield"}."""
    return {
        1: "Hamming",
        2: "Hopfield",
        3: "Hopfield",
        4: "Hamming",
    }


def check_bai_d() -> None:
    answers = bai_d()
    expected = {1: "Hamming", 2: "Hopfield", 3: "Hopfield", 4: "Hamming"}
    for q, expected_ans in expected.items():
        actual_ans = answers.get(q)
        assert actual_ans == expected_ans, (
            f"Bài D sai ở câu {q}: bạn chọn '{actual_ans}', đáp án đúng '{expected_ans}'"
        )
    print("Bài D: OK — chọn đúng mạng cho cả 4 tình huống.")


# ---------------------------------------------------------------------------
# Chạy toàn bộ kiểm tra
# ---------------------------------------------------------------------------

def main() -> None:
    print("Bắt đầu kiểm tra bài tập Buổi 3 (ký hiệu giao thông)...\n")
    check_bai_a()
    check_bai_b()
    check_bai_c()
    check_bai_d()
    print("\nTất cả các bài đều đúng! 10/10 điểm.")


if __name__ == "__main__":
    main()
