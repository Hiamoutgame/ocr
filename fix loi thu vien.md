fix lỗi bên trong thư viện 
For the code present, we get this error:
```
Import "pkg_resources" could not be resolved
```
Fix it, verify, and then give a concise explanation.

```

from importlib.metadata import version as _get_version

from .cached_download import cached_download
from .cached_download import md5sum
from .download import download
from .download_folder import download_folder
from .extractall import extractall

__author__ = "Kentaro Wada <www.kentaro.wada@gmail.com>"
__version__ = _get_version("gdown")

```


| Tiêu chí        | t_ocr.py                         | t_recognizer.py                          | full_pipeline.py                                      |
|-----------------|----------------------------------|------------------------------------------|--------------------------------------------------------|
| Mục đích        | Chỉ OCR (detect + nhận chữ)      | Chỉ detect layout hoặc cấu trúc bảng     | OCR + layout + bảng → 1 file markdown gần PDF          |
| Layout model    | Không dùng                       | Có (LayoutRecognizer hoặc TSR)           | Có (LayoutRecognizer + TSR)                            |
| OCR             | Có (toàn ảnh)                    | Chỉ khi mode=tsr (trong bảng)            | Có (từng vùng layout)                                  |
| Output chính    | Ảnh vẽ box + .txt + .md (bullet) | Ảnh vẽ box + (nếu tsr) .md bảng          | Ảnh không vẽ; mỗi trang *_full.md + 1 file <doc>_layout.md |
| Đa GPU          | Có (trio)                        | Không                                    | Không                                                  |