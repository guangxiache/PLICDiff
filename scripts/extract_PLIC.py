"""兼容 shim：保留旧导入路径 'from extract_PLIC import PLIC'。

真实实现已合并至 datasets/extract_PLIC.py（训练与采样共用）。
采样侧调用示例：
    plic = PLIC()
    pli, pocket_path = plic.pli_processor(lig, protein, already_cropped=False, return_pocket=True)
训练侧（pocket 已切好）：
    pli = plic.pli_processor(lig, pocket)          # already_cropped=True 为默认
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from datasets.extract_PLIC import PLIC  # noqa: F401
