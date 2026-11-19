# Dữ liệu

Bộ dữ liệu **UCI Covertype (ID 31)**: 581.012 ô lưới 30 m × 30 m, 54 thuộc tính, 7 lớp.
Nguồn: <https://archive.ics.uci.edu/dataset/31>.

| File | Trong git? | Ghi chú |
| :--- | :---: | :--- |
| `raw/covtype.data` | Không (72 MB) | Bảng gốc, không có dòng tiêu đề |
| `raw/covtype.data.gz` | Không (11 MB) | Bản nén của file trên |
| `raw/covtype.info` | Có | Mô tả thuộc tính, bảng mã 40 loại đất (ELU) |
| `raw/old_covtype.info` | Có | Phiên bản mô tả cũ của UCI |

## Lấy dữ liệu

Không cần tải thủ công. Nếu không tìm thấy `raw/covtype.data` hoặc `raw/covtype.data.gz`, notebook sẽ tự tải bằng
`sklearn.datasets.fetch_covtype()`. Bản tải này đã được đối chiếu và **trùng khớp từng giá trị, cùng thứ tự dòng**
với `covtype.data` (581.012 × 55), nên các phép chia dữ liệu dùng `random_state` cho cùng kết quả.

Muốn dùng file cục bộ: tải `covtype.data.gz` từ trang UCI ở trên và đặt vào `data/raw/`.
