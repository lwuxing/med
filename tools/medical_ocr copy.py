from collections.abc import Generator
from typing import Any, List, Tuple, Generator

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from dify_plugin.config.logger_format import plugin_logger_handler

import io
import base64
import logging
import concurrent.futures
import requests
import numpy as np
from PIL import Image
from rapidocr import RapidOCR, OCRVersion, LangDet
from rapidocr import ModelType as OCRModelType
from rapidocr.utils import RapidOCROutput
from rapid_layout import EngineType, ModelType, RapidLayout, RapidLayoutInput

try:
    from enum import StrEnum
except ImportError:
    from enum import Enum

    class StrEnum(str, Enum):
        def __str__(self):
            return str(self.value)

        def __format__(self, format_spec):
            return format(self.value, format_spec)


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(plugin_logger_handler)


class OcrType(StrEnum):
    TEXT_REC = "text_rec"
    TABLE_REC = "table_rec"


def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)

    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height

    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2g - x1g) * (y2g - y1g)
    union_area = area1 + area2 - inter_area

    if union_area == 0:
        return 0
    return inter_area / union_area


def is_contained(inner, outer):
    ix1, iy1, ix2, iy2 = inner
    ox1, oy1, ox2, oy2 = outer
    return ox1 <= ix1 and oy1 <= iy1 and ox2 >= ix2 and oy2 >= iy2


def filter_boxes_by_area_and_nms(
    boxes: List[List[float]], iou_thresh: float = 0.5
) -> List[List[float]]:
    """
    对矩形框进行面积排序，并基于包含关系 + IoU > 阈值 进行 NMS 过滤
    :param boxes: 输入框列表，每个是 [x1, y1, x2, y2]
    :param iou_thresh: IoU 阈值，默认 0.5
    :return: 过滤后的保留框列表
    """
    # 计算面积并排序
    contours_with_area: List[Tuple[List[float], float]] = []
    for box in boxes:
        x1, y1, x2, y2 = box
        area = (x2 - x1) * (y2 - y1)
        contours_with_area.append((box, area))
    contours_with_area.sort(key=lambda x: x[1], reverse=True)

    # 过滤包含 + IoU > 阈值
    filtered = []
    for i, (bbox_i, area_i) in enumerate(contours_with_area):
        keep = True
        for bbox_j, _ in filtered:
            if is_contained(bbox_i, bbox_j):
                keep = False
                break
            if compute_iou(bbox_i, bbox_j) > iou_thresh:
                keep = False
                break
        if keep:
            filtered.append((bbox_i, area_i))

    # 返回保留的框（去掉面积）
    return [box for box, _ in filtered]


def sort_ocr_results_by_lines(ocr_result_obj, y_thresh=10):
    """
    支持 RapidOCR 的结果对象结构，按行整理输出。

    :param ocr_result_obj: OCR 返回的对象，具有 .boxes 和 .txts 属性
    :param y_thresh: 同一行 y 坐标最大容忍差（像素）
    :return: List[List[str]] 每行为一个文本列表
    """
    boxes = ocr_result_obj.boxes
    txts = ocr_result_obj.txts

    items = []

    for text, box in zip(txts, boxes):
        if text is None or not isinstance(box, (np.ndarray, list)):
            continue
        box = np.array(box)
        x_center = float(np.mean(box[:, 0]))
        y_center = float(np.mean(box[:, 1]))
        items.append((text, x_center, y_center))

    # 按 y 坐标聚类成“行”
    lines = []
    for text, x, y in sorted(items, key=lambda x: x[2]):
        placed = False
        for line in lines:
            if abs(line[0][2] - y) < y_thresh:
                line.append((text, x, y))
                placed = True
                break
        if not placed:
            lines.append([(text, x, y)])

    # 每行按 x 排序并提取文本
    sorted_lines = []
    for line in lines:
        line_sorted = sorted(line, key=lambda x: x[1])  # 按 x 排序
        sorted_lines.append([text for text, _, _ in line_sorted])

    return sorted_lines


class MedicalOcrTool(Tool):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ocr_engine = RapidOCR(
            params={
                # "Det.engine_type": EngineType.ONNXRUNTIME,
                # "Det.lang_type": LangDet.CH,
                # "Det.model_type": OCRModelType.MOBILE,
                "Det.ocr_version": OCRVersion.PPOCRV5,
                "EngineConfig.onnxruntime.use_cuda": True,  
                "EngineConfig.onnxruntime.cuda_ep_cfg.device_id": 0, 
            }
        )
        self.layout_engine = RapidLayout(
            cfg=RapidLayoutInput(model_type=ModelType.DOCLAYOUT_D4LA)
        )

    def text_ocr(self, image_path: str):
        results = self.ocr_engine(image_path)
        if isinstance(results, RapidOCROutput):
            results = [results]

        content = ""
        for result in results:
            text = result.txts
            content += " ".join(text) + "\n"
        return content

    def table_rec(self, image_path: str):

        results = self.layout_engine(image_path)

        table_regions = []
        for bbox, class_name in zip(results.boxes, results.class_names):
            if class_name in ["Table", "RegionList"]:
                table_regions.append(bbox)

        filtered_boxes = filter_boxes_by_area_and_nms(table_regions)

        all_lines = []

        for i, bbox in enumerate(filtered_boxes, start=1):
            bbox = [int(x) for x in bbox]
            cropped_img = results.img[bbox[1] : bbox[3], bbox[0] : bbox[2], :]

            ocr_result = self.ocr_engine(cropped_img)

            # ocr_result[1] 是 OCR 返回的文字内容
            lines = sort_ocr_results_by_lines(ocr_result)

            all_lines.extend(lines)

        content = ""
        for row in all_lines:
            content += " ".join(row) + "\n"
        return content

    @staticmethod
    def _load_img_from_source(source: str):
        if any(source.startswith(prefix) for prefix in ["http://", "https://"]):
            response = requests.get(source)
            img = Image.open(io.BytesIO(response.content))
        else:
            # 处理 base64 字符串
            if "base64," in source:
                source = source.split("base64,", 1)[1]

            imbyte = base64.b64decode(source)
            img = Image.open(io.BytesIO(imbyte))

        return img

    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:

        uploaded_files = tool_parameters.get("file")
        logger.debug(f"uploaded_files: {uploaded_files}")
        if not uploaded_files:
            yield self.create_text_message("请上传文件")
            return

        _type = tool_parameters.get("ocr_type")
        logger.debug(f"_type: {_type}")

        if _type not in (OcrType.TEXT_REC, OcrType.TABLE_REC):
            yield self.create_text_message("请选择正确的处理类型")
            return

        contents = []

        if _type == OcrType.TEXT_REC:
            for uploaded_file in uploaded_files:
                url = uploaded_file.url
                try:
                    img = self._load_img_from_source(url)
                    content = self.text_ocr(img)
                    contents.append(content)
                except Exception as e:
                    yield self.create_text_message(f"图片处理错误: {e}")
                    return

        elif _type == OcrType.TABLE_REC:
            for uploaded_file in uploaded_files:
                url = uploaded_file.url
                try:
                    img = self._load_img_from_source(url)
                    content = self.table_rec(img)
                    contents.append(content)
                except Exception as e:
                    yield self.create_text_message(f"图片处理错误: {e}")
                    return
        else:
            yield self.create_text_message("请选择正确的处理类型")
            return

        contents = "\n".join(contents)
        print(contents)
        yield self.create_text_message(contents)
