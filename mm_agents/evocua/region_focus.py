"""RegionFocus-style visual test-time grounding refinement for EvoCUA.

This module implements a lightweight, training-free wrapper inspired by
"Visual Test-time Scaling for GUI Agent Grounding".  It is intentionally
self-contained so it can be inserted between EvoCUA's action generation and
OSWorld's ``env.step`` without changing the base agent.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from PIL import Image

logger = logging.getLogger("desktopenv.region_focus")


_CLICK_RE = re.compile(
    r"pyautogui\.(?P<fn>click|rightClick|middleClick|doubleClick|tripleClick)"
    r"\(\s*(?P<x>-?\d+(?:\.\d+)?)\s*,\s*(?P<y>-?\d+(?:\.\d+)?)"
    r"(?P<rest>\s*(?:,[^)]*)?)\)",
    re.IGNORECASE,
)

_CLICK_KW_RE = re.compile(
    r"pyautogui\.(?P<fn>click|rightClick|middleClick|doubleClick|tripleClick)"
    r"\(\s*x\s*=\s*(?P<x>-?\d+(?:\.\d+)?)\s*,\s*y\s*=\s*(?P<y>-?\d+(?:\.\d+)?)"
    r"(?P<rest>\s*(?:,[^)]*)?)\)",
    re.IGNORECASE,
)


@dataclass
class ClickAction:
    fn: str
    x: int
    y: int
    rest: str
    match_start: int
    match_end: int


@dataclass
class RegionFocusConfig:
    enabled: bool = False
    crop_ratios: Tuple[float, ...] = (0.45, 0.65)
    min_crop_size: int = 384
    max_calls: int = 2
    confidence_threshold: float = 0.0
    save_debug: bool = True

    @classmethod
    def from_args(cls, args: Any) -> "RegionFocusConfig":
        ratios = os.environ.get("EVO_REGION_FOCUS_CROP_RATIOS") or getattr(args, "region_focus_crop_ratios", "0.45,0.65")
        parsed_ratios: List[float] = []
        if isinstance(ratios, str):
            for part in ratios.split(","):
                part = part.strip()
                if not part:
                    continue
                try:
                    value = float(part)
                except ValueError:
                    logger.warning("Ignoring invalid RegionFocus crop ratio: %s", part)
                    continue
                if 0 < value <= 1:
                    parsed_ratios.append(value)
        elif isinstance(ratios, Iterable):
            parsed_ratios = [float(x) for x in ratios if 0 < float(x) <= 1]

        if not parsed_ratios:
            parsed_ratios = [0.45, 0.65]

        return cls(
            enabled=bool(getattr(args, "enable_region_focus", False)) or os.environ.get("EVO_REGION_FOCUS", "0").lower() in {"1", "true", "yes", "on"},
            crop_ratios=tuple(parsed_ratios),
            min_crop_size=int(os.environ.get("EVO_REGION_FOCUS_MIN_CROP_SIZE") or getattr(args, "region_focus_min_crop_size", 384)),
            max_calls=max(1, int(os.environ.get("EVO_REGION_FOCUS_MAX_CALLS") or getattr(args, "region_focus_max_calls", 2))),
            confidence_threshold=float(os.environ.get("EVO_REGION_FOCUS_CONFIDENCE_THRESHOLD") or getattr(args, "region_focus_confidence_threshold", 0.0)),
            save_debug=(os.environ.get("EVO_REGION_FOCUS_SAVE_DEBUG") or str(getattr(args, "region_focus_save_debug", True))).lower() not in {"0", "false", "no", "off"},
        )


def parse_click_action(action: str) -> Optional[ClickAction]:
    """Return the first click-like pyautogui action in ``action``."""
    if not isinstance(action, str):
        return None

    match = _CLICK_RE.search(action) or _CLICK_KW_RE.search(action)
    if not match:
        return None

    try:
        x = int(round(float(match.group("x"))))
        y = int(round(float(match.group("y"))))
    except Exception:
        return None

    return ClickAction(
        fn=match.group("fn"),
        x=x,
        y=y,
        rest=match.groupdict().get("rest") or "",
        match_start=match.start(),
        match_end=match.end(),
    )


def replace_click_coordinates(action: str, click: ClickAction, x: int, y: int) -> str:
    """Replace the parsed click coordinates while preserving the click type."""
    x = int(round(x))
    y = int(round(y))
    new_call = f"pyautogui.{click.fn}({x}, {y}{click.rest})"
    return action[: click.match_start] + new_call + action[click.match_end :]


def _encode_png(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort JSON object extraction from an LLM response."""
    if not text:
        return None

    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    candidates = [cleaned]
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))

    for cand in candidates:
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


class RegionFocusRefiner:
    """Training-free click refinement using zoomed local crops.

    The refiner only touches click-like pyautogui actions.  If parsing, model
    calling, or JSON extraction fails, it returns the original action so that
    the evaluation loop remains safe.
    """

    def __init__(self, config: RegionFocusConfig):
        self.config = config

    def refine_action(
        self,
        *,
        agent: Any,
        obs: Dict[str, Any],
        instruction: str,
        model_response: str,
        action: str,
        step_idx: int,
        action_timestamp: str,
        debug_dir: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        info: Dict[str, Any] = {
            "enabled": self.config.enabled,
            "changed": False,
            "original_action": action,
            "refined_action": action,
            "reason": "disabled",
            "candidates": [],
        }

        if not self.config.enabled:
            return action, info

        click = parse_click_action(action)
        if click is None:
            info["reason"] = "not_click_action"
            return action, info

        try:
            screenshot = Image.open(io.BytesIO(obs["screenshot"])).convert("RGB")
        except Exception as exc:
            logger.warning("RegionFocus failed to open screenshot: %s", exc)
            info["reason"] = f"screenshot_open_failed: {exc}"
            return action, info

        width, height = screenshot.size
        click.x = int(_clip(click.x, 0, width - 1))
        click.y = int(_clip(click.y, 0, height - 1))

        crops = self._build_crops(width, height, click.x, click.y)
        selected: Optional[Dict[str, Any]] = None

        for crop_idx, box in enumerate(crops[: self.config.max_calls], start=1):
            left, top, right, bottom = box
            crop = screenshot.crop(box)
            if self.config.save_debug and debug_dir:
                self._save_crop_debug(crop, debug_dir, step_idx, action_timestamp, crop_idx)

            prompt = self._build_prompt(
                instruction=instruction,
                model_response=model_response,
                action=action,
                click=click,
                crop_box=box,
                crop_size=crop.size,
            )

            try:
                response = agent.call_llm(
                    {
                        "model": getattr(agent, "model"),
                        "messages": [
                            {
                                "role": "system",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": "You are a precise GUI grounding refiner. Return only valid JSON.",
                                    }
                                ],
                            },
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": f"data:image/png;base64,{_encode_png(crop)}"},
                                    },
                                    {"type": "text", "text": prompt},
                                ],
                            },
                        ],
                        "max_tokens": 512,
                        "temperature": 0.0,
                        "top_p": 0.9,
                    }
                )
            except Exception as exc:
                logger.warning("RegionFocus LLM call failed on crop %s: %s", crop_idx, exc)
                info["candidates"].append({"crop_idx": crop_idx, "error": str(exc), "box": box})
                continue

            candidate = self._parse_candidate(response, box, crop.size)
            candidate.update(
                {
                    "crop_idx": crop_idx,
                    "box": list(box),
                    "raw_response": response[:1000] if isinstance(response, str) else str(response)[:1000],
                }
            )
            info["candidates"].append(candidate)

            if candidate.get("valid") and candidate.get("confidence", 0.0) >= self.config.confidence_threshold:
                if selected is None or candidate.get("confidence", 0.0) > selected.get("confidence", 0.0):
                    selected = candidate

        if selected is None:
            info["reason"] = "no_valid_candidate"
            return action, info

        refined_x = int(_clip(selected["abs_x"], 0, width - 1))
        refined_y = int(_clip(selected["abs_y"], 0, height - 1))
        refined_action = replace_click_coordinates(action, click, refined_x, refined_y)

        info.update(
            {
                "changed": refined_action != action,
                "refined_action": refined_action,
                "reason": "refined",
                "selected": selected,
            }
        )
        logger.info("RegionFocus refined action: %s -> %s", action, refined_action)
        return refined_action, info

    def _build_crops(self, width: int, height: int, x: int, y: int) -> List[Tuple[int, int, int, int]]:
        boxes: List[Tuple[int, int, int, int]] = []
        min_dim = min(width, height)
        for ratio in self.config.crop_ratios:
            crop_size = max(self.config.min_crop_size, int(round(min_dim * ratio)))
            crop_w = min(width, crop_size)
            crop_h = min(height, crop_size)
            left = int(round(x - crop_w / 2))
            top = int(round(y - crop_h / 2))
            left = int(_clip(left, 0, max(0, width - crop_w)))
            top = int(_clip(top, 0, max(0, height - crop_h)))
            right = left + crop_w
            bottom = top + crop_h
            box = (left, top, right, bottom)
            if box not in boxes:
                boxes.append(box)
        return boxes

    def _build_prompt(
        self,
        *,
        instruction: str,
        model_response: str,
        action: str,
        click: ClickAction,
        crop_box: Tuple[int, int, int, int],
        crop_size: Tuple[int, int],
    ) -> str:
        left, top, right, bottom = crop_box
        crop_w, crop_h = crop_size
        local_x = int(_clip(click.x - left, 0, crop_w - 1))
        local_y = int(_clip(click.y - top, 0, crop_h - 1))
        local_grid_x = int(round(local_x / max(1, crop_w - 1) * 999))
        local_grid_y = int(round(local_y / max(1, crop_h - 1) * 999))

        return f"""
The image is a zoomed crop from a desktop GUI screenshot. Refine the click target for the same intended action.

Original user task:
{instruction}

Original agent response:
{model_response[:2000] if isinstance(model_response, str) else model_response}

Original executable action:
{action}

Crop metadata:
- Crop box in original screenshot: left={left}, top={top}, right={right}, bottom={bottom}
- Crop size: {crop_w}x{crop_h}
- The original planned click is approximately at crop-grid coordinate ({local_grid_x}, {local_grid_y}) on a 0-999 grid.

Return only one JSON object with this schema:
{{"x": <0-999>, "y": <0-999>, "confidence": <0.0-1.0>, "reason": "short reason"}}

Rules:
- x and y must be coordinates in this crop on a 0-999 grid.
- Choose the center of the UI element that should be clicked.
- If the original planned click is already correct, return the same target center.
- If the correct target is not visible in this crop, return confidence 0.0 and keep x/y near the original planned click.
- Do not output markdown, code fences, or extra text.
""".strip()

    def _parse_candidate(
        self,
        response: str,
        crop_box: Tuple[int, int, int, int],
        crop_size: Tuple[int, int],
    ) -> Dict[str, Any]:
        left, top, _, _ = crop_box
        crop_w, crop_h = crop_size
        obj = _extract_json_object(response)
        if not obj:
            return {"valid": False, "error": "json_parse_failed"}

        try:
            raw_x = float(obj.get("x"))
            raw_y = float(obj.get("y"))
        except Exception:
            return {"valid": False, "error": "missing_xy", "json": obj}

        # Prefer RegionFocus's 0-999 coordinate convention; also tolerate 0-1
        # normalized or raw crop-pixel coordinates to make the wrapper robust.
        if 0.0 <= raw_x <= 1.0 and 0.0 <= raw_y <= 1.0:
            local_x = raw_x * max(1, crop_w - 1)
            local_y = raw_y * max(1, crop_h - 1)
            coord_mode = "0-1"
        elif 0.0 <= raw_x <= 1000.0 and 0.0 <= raw_y <= 1000.0:
            local_x = raw_x / 999.0 * max(1, crop_w - 1)
            local_y = raw_y / 999.0 * max(1, crop_h - 1)
            coord_mode = "0-999"
        else:
            local_x = raw_x
            local_y = raw_y
            coord_mode = "pixel"

        confidence = obj.get("confidence", 0.0)
        try:
            confidence = float(confidence)
        except Exception:
            confidence = 0.0
        confidence = float(_clip(confidence, 0.0, 1.0))

        local_x = _clip(local_x, 0, crop_w - 1)
        local_y = _clip(local_y, 0, crop_h - 1)
        return {
            "valid": True,
            "x": raw_x,
            "y": raw_y,
            "coord_mode": coord_mode,
            "local_x": int(round(local_x)),
            "local_y": int(round(local_y)),
            "abs_x": int(round(left + local_x)),
            "abs_y": int(round(top + local_y)),
            "confidence": confidence,
            "reason": obj.get("reason", ""),
        }

    def _save_crop_debug(
        self,
        crop: Image.Image,
        debug_dir: str,
        step_idx: int,
        action_timestamp: str,
        crop_idx: int,
    ) -> None:
        try:
            region_dir = os.path.join(debug_dir, "region_focus")
            os.makedirs(region_dir, exist_ok=True)
            crop.save(os.path.join(region_dir, f"step_{step_idx}_{action_timestamp}_crop{crop_idx}.png"))
        except Exception as exc:
            logger.debug("Failed to save RegionFocus debug crop: %s", exc)
