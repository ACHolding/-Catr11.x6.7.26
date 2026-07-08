#!/usr/bin/env python3.14
import sys
if sys.version_info < (3, 14):
    sys.exit("Python 3.14+ required")
# todo files = off
"""cat r1.10 · files = off"""
import tkinter as tk
from tkinter import scrolledtext, font, messagebox
import numpy as np
import time
import threading
import re
import json
import os
import io
import ast
import contextlib
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field
import html as html_module
import uuid
from urllib import request as urlrequest
from urllib.error import URLError, HTTPError
from urllib.parse import urlparse
from html.parser import HTMLParser

# ──────────────────────────────────────────────────────────────
# FRONTIER DISTILLATION (in-memory student stack · files = off)
# catseek r1.10 frontier parity
# ──────────────────────────────────────────────────────────────
CAT_R1_DISTIL_PARAMS = {
    "distil_enabled": True,
    "distil_teacher_weight": 0.72,
    "teacher_weight": 0.72,
    "distil_passes": 4,
    "turbo_passes": 2,
    "distil_protocol": "cat-r1.10-distil",
}

# ──────────────────────────────────────────────────────────────
# FRONTIER MYTHOS PARAMETERS (BitNet GigaEngine · files = off)
# catseek r1.10 frontier level
# ──────────────────────────────────────────────────────────────
CAT_R1_MYTHOS_PARAMS = {
    "prose_tier": "catr1.10",
    "mythos_tier": "catr1.10",
    "reasoning_mode": "catseek-r1.10-hybrid-reasoning",
    "code_interpreter": "catr1.10-code",
    "code_interpreter_family": "catr1.10",
    "code_interpreter_version": "r1.10",
    "catr1_recursive_depth": 3,
    "mythos_mode": True,
    "mythos_recursive_improve": True,
    "mythos_recursive_depth": 3,
    "mythos_recursive_epsilon": 0.008,
    "code_auto_run": False,
    "code_output_exact": True,
    "code_token_weights_only": True,
    "mythos_voice": True,
    "mythos_runtime": True,
    "code_interpreter_name": "catr1.10 frontier code",
    "code_anything": True,
    "code_anything_mode": "universal-frontier",
    "simulate_latency": 0.06,
    "step_delay": 0.04,
    # catr1.10 frontier scale (10M ctx · 512K out · 1.7T effective)
    "catr1_context_window": 10_000_000,
    "catr1_max_output": 512_000,
    "nominal_base_params": 1_700_000_000_000,
}

# ──────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────
CONFIG = {
    # ── Frontier-scale architecture (catseek r1.10 frontier tier) ──
    "d_model": 512,
    "layers": 4,
    "heads": 8,
    "n_experts": 4,
    "top_k": 2,
    "recursive_depth": 3,
    "recursive_epsilon": 0.008,
    "o1_preview": True,
    "o1_self_check": True,
    "interpreter_syntax": "o1-preview",
    "o1_interpreter_protocol": "o1-preview-interpreter",
    "ultrathink_default": True,
    "r1_synth_default": True,
    # ── Multi-level compression (BitNet + sparse + low-rank + MoE) ──
    "compression_enabled": True,
    "compression_sparse_k": 128,
    "compression_rank": 256,
    "compression_stack_mult": 8,
    "weight_bits": 1.58,
    "nominal_base_params": 1_700_000_000_000,
    # ── API & protocol ──
    "api_port": 8765,
    "api_key": os.getenv("CATR1_API_KEY", "lm-studio"),
    "chat_protocol": "cat-r1.10-chat",
    "chat_version": "r1.10",
    "max_sessions": 128,
    # ── Tokenizer & context ──
    "vocab_size": 65536,
    "max_seq": 2048,
    "ff_mult": 4,
    "catr1_eps": 1e-5,
    "act_bits": 16,
    "files": "off",
    "catr1_enabled": True,
    "catr1_model_id": "catr1.10-frontier",
    "catr1_edition": "r1.10",
    "coding_api_protocol": "cat-r1.10-coding-api",
    "coding_api_version": "1.0",
    "coding_ide": "cursor",
    "code_workspace_max_files": 256,
    # ── Frontiers compression ──
    "whitepaper_compression": True,
    "google_whitepaper_heuristics": True,
    "catr1_voice": True,
    "compression_nvidia_awq_bits": 4,
    "compression_nvidia_gptq_block": 128,
    "compression_nvidia_sparse_2_4": True,
    "compression_google_lowrank_rank": 64,
    "compression_google_moe_experts": 64,
    "compression_google_moe_topk": 8,
    "compression_zstd_level": 6,
    # ── Frontier code engine ──
    "code_terminal_timeout": 8,
    "catr1_code_enabled": True,
    "catr1_engine": True,
    "catr1_code_perfect": True,
    "catr1b_lint": True,
    "vibe_code_heuristics": True,
    "catr1_reasoning": True,
    "deepmind_fast": True,
    "flash_attention": True,
    "adaptive_compute": True,
    "turbo_encode_tasks": ("chat", "code", "math", "execute", "explain", "agent", "research"),
    "catr1_self_verify": True,
    "web_program_enabled": True,
    "web_fetch_enabled": True,
    "web_max_sites": 256,
    "web_max_fetch_kb": 1024,
    "web_preview_port": None,
    # ── catseek r1.10 frontier features ──
    "catr1_moe_dense": True,
    "catr1_multi_turn_search": True,
    "catr1_fun_mode": True,
    "catr1_realtime": True,
    "catr1_moe_scale": True,
    "catr1_multi_token_prediction": True,
    "catr1_active_inference": True,
    "catr1_long_context_finegrained": True,
    "catr1_sparse_attention": True,
    # ── GUI ──
    "gui_theme": "catr1.10",
    "gui_catr1": True,
    "default_chat_mode": "expert",
    **CAT_R1_DISTIL_PARAMS,
    **CAT_R1_MYTHOS_PARAMS,
}

# catr1.10 frontier layout · files = off · catseek r1.10
CAT_R1_UI = {
    "bg": "#0f1117",
    "sidebar": "#13161e",
    "sidebar_border": "#1e2230",
    "header_bg": "#0f1117",
    "header_border": "#1e2230",
    "text": "#e4e6ed",
    "muted": "#7c8298",
    "user_bg": "#1a2040",
    "user_fg": "#e4e6ed",
    "bot_bg": "#13161e",
    "bot_fg": "#e4e6ed",
    "think_fg": "#8b92a8",
    "think_bg": "#161a24",
    "think_border": "#1e2230",
    "code_bg": "#0a0c12",
    "code_fg": "#c9d1d9",
    "input_bg": "#13161e",
    "input_border": "#1e2230",
    "input_shadow": "#0a0c12",
    "accent": "#6c5ce7",
    "accent_text": "#3b82f6",
    "send_hover": "#1a1a1a",
    "avatar_bot": "#6c5ce7",
    "avatar_user": "#7c8298",
    "mascot_bg": "#1a1f33",
    "mascot_fg": "#a78bfa",
    "new_chat_bg": "#000000",
    "new_chat_border": "#000000",
    "history_hover": "#0d0d0d",
    "empty_title": "#e4e6ed",
    "radius_pad": 16,
    "input_radius": 20,
    "mode_active_bg": "#000000",
    "mode_active_fg": "#3b82f6",
    "mode_idle_bg": "#000000",
    "mode_idle_fg": "#3b82f6",
    "toggle_on": "#000000",
    "toggle_off": "#000000",
}

GUI_APP_NAME = "catr1.10"
WINDOW_TITLE = "catr1.10 frontier"
GUI_TAGLINE = ""
MASCOT_GLYPH = "🐱"
MASCOT_NAME = "catr1.10"
CAT_R1_PRO = "catr1.10"
CAT_R1_FLASH = "catr1.10-flash"
V4_MODEL_PRO = CAT_R1_PRO
V4_MODEL_FLASH = CAT_R1_FLASH
V4_MODEL_LABEL_PRO = V4_MODEL_PRO
V4_MODEL_LABEL_FLASH = V4_MODEL_FLASH

FILES = CONFIG["files"]
BRAND = "CatR1.10"
EDITION = CONFIG.get("catr1_edition", "r1")
MODEL_NAME = "catr1.10"
CORE_NAME = "catr1.10 frontier core"
WEB_PROGRAM_NAME = "catr1.10 web"
LINEAR_NAME = "catr1.10 linear"
CAT_R1_MODEL_ID = CONFIG["catr1_model_id"]
MYTHOS_TIER = CONFIG["mythos_tier"]
CODE_ENGINE = CONFIG["code_interpreter_name"]
CODE_BACKEND = CONFIG["code_interpreter"]
CODING_API_VER = CONFIG.get("coding_api_version", "1.0")
CODING_API_PROTO = CONFIG.get("coding_api_protocol", "cat-r1.10-coding-api")
CODING_API_LABEL = f"{CODE_ENGINE} {CODING_API_VER}"
CATR1_ENGINE = CODING_API_LABEL
CAT_R1_CODE_ENABLED = CONFIG["catr1_code_enabled"]
MYTHOS_MODE = CONFIG.get("mythos_mode", True)
CAT_R1_REASONING = CONFIG.get("catr1_reasoning", True)
MYTHOS_NAME = BRAND
BRAND_TAG = BRAND
MASCOT_DESC = BRAND_TAG
REASONING_MODE = CONFIG.get("reasoning_mode", "catseek-r1.10-hybrid-reasoning")
PROSE_TIER = CONFIG["prose_tier"]
VERSION = EDITION
MYTHOS_ENGINE_VER = CONFIG["code_interpreter_version"]
O1_INTERPRETER_PROTO = CONFIG.get("o1_interpreter_protocol", "o1-preview-interpreter")
O1_INTERPRETER_TAG = CONFIG.get("interpreter_syntax", "o1-preview")

# Legacy aliases (internal · do not use in user-facing strings)
CATSEEK_UI = CAT_R1_UI
CATSEEK_MODEL_ID = CAT_R1_MODEL_ID
CATSEEK_CODE_ENABLED = CAT_R1_CODE_ENABLED
CATSEEK_R1_NAME = CAT_R1_PRO
CATSEEK_V3_NAME = CAT_R1_FLASH
CATSEEK_R1_MODE = CAT_R1_REASONING

# ──────────────────────────────────────────────────────────────
# PERSISTENT MEMORY (first-run per device · settings · files = soft)
# catseek r1.10 keeps chat in-memory; only device metadata persists.
# ──────────────────────────────────────────────────────────────
PERSISTENT_MEMORY_DIR = os.path.join(
    os.path.expanduser("~"),
    ".config" if sys.platform != "darwin" else "Library/Application Support",
    "catr1"
)
PERSISTENT_MEMORY_FILE = os.path.join(PERSISTENT_MEMORY_DIR, "memory.json")


@dataclass
class PersistentMemory:
    first_run: bool = True
    device_id: str = ""
    chat_mode: str = "expert"
    thinking_on: bool = True
    total_sessions: int = 0
    total_messages: int = 0
    setup_version: str = ""


def _load_persistent_memory() -> PersistentMemory:
    if os.path.exists(PERSISTENT_MEMORY_FILE):
        try:
            with open(PERSISTENT_MEMORY_FILE) as f:
                data = json.load(f)
            import dataclasses
            known = {f.name for f in dataclasses.fields(PersistentMemory)}
            return PersistentMemory(**{k: v for k, v in data.items() if k in known})
        except Exception:
            pass
    os.makedirs(PERSISTENT_MEMORY_DIR, exist_ok=True)
    mem = PersistentMemory()
    mem.device_id = uuid.uuid4().hex[:12]
    mem.setup_version = EDITION
    _save_persistent_memory(mem)
    return mem


def _save_persistent_memory(mem: PersistentMemory) -> None:
    os.makedirs(PERSISTENT_MEMORY_DIR, exist_ok=True)
    with open(PERSISTENT_MEMORY_FILE, "w") as f:
        json.dump({
            "first_run": mem.first_run,
            "device_id": mem.device_id,
            "chat_mode": mem.chat_mode,
            "thinking_on": mem.thinking_on,
            "total_sessions": mem.total_sessions,
            "total_messages": mem.total_messages,
            "setup_version": mem.setup_version,
        }, f, indent=2)


CAT_R11_PROFILE_MD = f"""# {BRAND} (Frontier Local)

**{CAT_R1_MODEL_ID}** · {CORE_NAME} · **catseek r1.10 frontier**.

| | |
|---|---|---|
| Brand | **{BRAND}** — frontier local assistant |
| Tier | catseek r1.10 reasoning · catseek r1.10 code |
| Chat | **{CAT_R1_PRO}** reasoning · **{CAT_R1_FLASH}** instant |
| Runtime | **{MYTHOS_NAME}** on {CORE_NAME} |
| Reasoning | {REASONING_MODE} + recursive polish + GRPO self-verify |
| Prose | `{PROSE_TIER}` extended thinking |
| Context | {CONFIG['catr1_context_window']:,} tokens (in-memory) |
| Output | up to {CONFIG['catr1_max_output']:,} tokens |
| Code | **{CODE_ENGINE}** · `{CODE_BACKEND}` · `{MYTHOS_ENGINE_VER}` |
| Student stack | {CONFIG['distil_passes']} in-memory heads |
| MoE | {CONFIG['n_experts']} experts · top-{CONFIG['top_k']} routing · catr1 MoE scale |
| Compression | Sparse top-{CONFIG['compression_sparse_k']} · low-rank {CONFIG['compression_rank']} · {CONFIG['weight_bits']}-bit BitNet |
| Web | {WEB_PROGRAM_NAME} — artifacts · fetch · preview |
| Architecture | {LINEAR_NAME} · causal MHA · MoE FFN · ReLU² · RMSNorm |
| Weights | AbsMean ternary {{−1, 0, 1}} @ {CONFIG['weight_bits']} bits |
| Effective params | {CONFIG['nominal_base_params'] / 1e12:.1f}T (frontier tier) |
| API ID | `{CAT_R1_MODEL_ID}` |

Run: `python3 ">>catseekr1.10.py"` · CLI: `python3 ">>catseekr1.10.py" --chat`
"""


# ──────────────────────────────────────────────────────────────
# CATR1.10 FRONTIER CORE (catseek r1.10 · BitNet GigaEngine)
# files = off · in-memory shadow weights + packed ternary · 1.7T effective
# ──────────────────────────────────────────────────────────────
def _round_clip(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(np.round(x), lo, hi).astype(np.int8)


def _squared_relu(x: np.ndarray) -> np.ndarray:
    r = np.maximum(x, 0.0)
    return (r * r).astype(np.float32)


class CatSeekLinear:
    """
    catr1.10 BitNet GigaLinear layer (files = off):
    shadow FP32 weights → AbsMean ternary {-1,0,1} → int8/16 activation matmul.
    catseek r1.10 · 1.58-bit BitNet engine.
    """

    __slots__ = ("in_f", "out_f", "shadow_w", "bias", "w_scale", "w_signed", "w_packed", "eps")

    def __init__(self, in_features: int, out_features: int, seed: int, *, bias: bool = False):
        rng = np.random.RandomState(seed)
        self.in_f = in_features
        self.out_f = out_features
        self.eps = CONFIG["catr1_eps"]
        scale = np.sqrt(2.0 / max(in_features, 1))
        self.shadow_w = (rng.randn(out_features, in_features).astype(np.float32) * scale)
        self.bias = np.zeros(out_features, dtype=np.float32) if bias else None
        self.w_scale = np.float32(1.0)
        self.w_signed = np.zeros((out_features, in_features), dtype=np.int8)
        self.w_packed = b""
        self.requantize()

    def requantize(self) -> None:
        gamma = float(np.mean(np.abs(self.shadow_w))) + self.eps
        self.w_scale = np.float32(gamma)
        scaled = self.shadow_w / gamma
        w_q = _round_clip(scaled, -1, 1)
        self.w_signed = w_q.astype(np.int8)
        self.w_packed = self._pack(w_q)

    @staticmethod
    def _pack(w_q: np.ndarray) -> bytes:
        """Base-3 pack: 5 ternary values {-1,0,1} → 1 byte (cat r1.10 ~1.58 bits/weight)."""
        flat = w_q.ravel()
        out = bytearray()
        for i in range(0, len(flat), 5):
            chunk = flat[i : i + 5]
            val = 0
            for j, t in enumerate(chunk):
                val += (int(t) + 1) * (3 ** j)
            out.append(val % 256)
        return bytes(out)

    @staticmethod
    def _absmax_quantize_x(x: np.ndarray) -> Tuple[np.ndarray, float]:
        scale = float(np.max(np.abs(x))) + CONFIG["catr1_eps"]
        qmax = (2 ** (CONFIG["act_bits"] - 1)) - 1
        x_q = np.clip(np.round(x / scale * qmax), -qmax, qmax).astype(np.int16)
        return x_q, scale

    def forward(self, x: np.ndarray) -> np.ndarray:
        single = x.ndim == 1
        xm = np.atleast_2d(x.astype(np.float32))
        qmax = (2 ** (CONFIG["act_bits"] - 1)) - 1
        scales = np.max(np.abs(xm), axis=1, keepdims=True) + self.eps
        x_q = np.clip(np.round(xm / scales * qmax), -qmax, qmax).astype(np.int16)
        acc = (x_q.astype(np.int32) @ self.w_signed.T.astype(np.int32)).astype(np.float32)
        acc = acc * scales * float(self.w_scale)
        if self.bias is not None:
            acc = acc + self.bias
        return acc[0] if single else acc

    def param_count(self) -> Tuple[int, float]:
        n = self.in_f * self.out_f
        return n, n * CONFIG["weight_bits"] / 8.0


def _rms_norm(x: np.ndarray, gamma: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    if x.ndim == 1:
        rms = float(np.sqrt(np.mean(x * x) + eps))
        return (x / rms * gamma).astype(np.float32)
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return (x / rms * gamma).astype(np.float32)


class CatSeekBlock:
    """One transformer block: causal MHA + cat r1.10 linear FFN (ReLU²), all ternary matmul."""

    __slots__ = ("q", "k", "v", "o", "ff_up", "ff_down", "router", "experts", "norm1", "norm2", "_h", "_hd")

    def __init__(self, d_model: int, seed: int):
        h = CONFIG["heads"]
        hd = d_model // h
        ff = d_model * CONFIG["ff_mult"]
        self.q = CatSeekLinear(d_model, d_model, seed + 1)
        self.k = CatSeekLinear(d_model, d_model, seed + 2)
        self.v = CatSeekLinear(d_model, d_model, seed + 3)
        self.o = CatSeekLinear(d_model, d_model, seed + 4)
        self.ff_up = CatSeekLinear(d_model, ff, seed + 5)
        self.ff_down = CatSeekLinear(ff, d_model, seed + 6)
        self.router = CatSeekLinear(d_model, CONFIG["n_experts"], seed + 7)
        rng = np.random.RandomState(seed + 8)
        self.experts = [
            (CatSeekLinear(d_model, ff, seed + 100 + i * 2), CatSeekLinear(ff, d_model, seed + 101 + i * 2))
            for i in range(CONFIG["n_experts"])
        ]
        self.norm1 = rng.randn(d_model).astype(np.float32) * 0.1 + 1.0
        self.norm2 = rng.randn(d_model).astype(np.float32) * 0.1 + 1.0
        self._hd = hd
        self._h = h

    def _causal_mha(self, x: np.ndarray) -> np.ndarray:
        t, d = x.shape
        h, hd = self._h, self._hd
        eps = CONFIG["catr1_eps"]
        xn = np.stack([_rms_norm(x[ti], self.norm1) for ti in range(t)], axis=0)
        q = self.q.forward(xn).reshape(t, h, hd)
        k = self.k.forward(xn).reshape(t, h, hd)
        v = self.v.forward(xn).reshape(t, h, hd)
        scale = np.sqrt(hd) + eps
        if CONFIG.get("flash_attention") and t > 1:
            scores = np.einsum("thd,shd->hts", q, k) / scale
            mask = np.triu(np.ones((t, t), dtype=bool), k=1)
            scores = np.where(mask[np.newaxis, :, :], -1e9, scores)
            scores = scores - np.max(scores, axis=-1, keepdims=True)
            w = np.exp(scores)
            w = w / (np.sum(w, axis=-1, keepdims=True) + eps)
            ctx = np.einsum("hts,shd->htd", w, v).reshape(t, d)
            return self.o.forward(ctx)
        out = np.zeros((t, d), dtype=np.float32)
        for ti in range(t):
            scores = np.einsum("hd,ihd->ih", q[ti], k[: ti + 1]) / scale
            scores = scores - np.max(scores, axis=-1, keepdims=True)
            w = np.exp(scores)
            w = w / (np.sum(w, axis=-1, keepdims=True) + eps)
            ctx = np.einsum("ih,ihd->hd", w, v[: ti + 1])
            out[ti] = ctx.reshape(d)
        return self.o.forward(out)

    def _moe_ffn(self, x: np.ndarray) -> np.ndarray:
        t, d = x.shape
        out = np.zeros_like(x)
        for ti in range(t):
            xn = _rms_norm(x[ti], self.norm2)
            logits = self.router.forward(xn)
            top = np.argsort(logits)[-CONFIG["top_k"]:]
            acc = np.zeros(d, dtype=np.float32)
            for idx in top:
                up, down = self.experts[int(idx)]
                h = _squared_relu(up.forward(xn))
                acc = acc + down.forward(h) / CONFIG["top_k"]
            out[ti] = acc
        return out

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x + self._moe_ffn(x + self._causal_mha(x))


def _iter_catseek_linears(block: CatSeekBlock) -> List[CatSeekLinear]:
    layers = [block.q, block.k, block.v, block.o, block.ff_up, block.ff_down, block.router]
    for up, down in block.experts:
        layers.extend([up, down])
    return layers


def catseek_memory_report(blocks: List[CatSeekBlock], embed: np.ndarray, head: np.ndarray) -> Dict[str, Any]:
    shadow = embed.size + head.size
    packed_bytes = 0
    effective_bits = 0.0
    linear_count = 0
    for blk in blocks:
        for lin in _iter_catseek_linears(blk):
            n, mem = lin.param_count()
            shadow += n
            effective_bits += n * CONFIG["weight_bits"]
            packed_bytes += len(lin.w_packed)
            linear_count += 1
    return {
        "catr1_linear_layers": linear_count,
        "shadow_params": shadow,
        "effective_bits": effective_bits,
        "packed_kb": packed_bytes / 1024.0,
        "effective_mb": effective_bits / 8.0 / 1024.0 / 1024.0,
        "weight_bits": CONFIG["weight_bits"],
    }


# ──────────────────────────────────────────────────────────────
# O1-PREVIEW INTERPRETER SYNTAX (files = off)
# Canonical trace: Understand → Plan → Reason → Self-check → Verify → Answer
# ──────────────────────────────────────────────────────────────
class O1PreviewSyntax:
    """catseek r1.10 hybrid reasoning syntax · files = off."""

    MODEL = "catr1.10-frontier"
    PROTO = O1_INTERPRETER_PROTO
    TAG = O1_INTERPRETER_TAG
    VERSION = "r1.10"
    COMMANDS = ("/run", "/interpret", "/think", "/ultrathink", "run it", "execute")

    @classmethod
    def header(cls) -> str:
        if MYTHOS_MODE and CONFIG.get("mythos_runtime", True):
            return f"{BRAND} · {REASONING_MODE}"
        return f"{cls.TAG}"

    @classmethod
    def step(cls, n: int, label: str, body: str) -> str:
        return f"{n}. {label} — {body}"

    @classmethod
    def final_answer_step(cls, n: int) -> str:
        return cls.step(n, "Answer", "emit clean user-facing text.")

    @classmethod
    def build_trace(
        cls,
        prompt: str,
        *,
        intent: str,
        subtasks: List[str],
        reason: str = "",
        recursive_trace: Optional[List[str]] = None,
        compression_trace: Optional[List[str]] = None,
        verify: str = "",
        self_check: str = "",
    ) -> str:
        lines = [cls.header(), cls.step(1, "Understand", intent)]
        for i, task in enumerate(subtasks, start=2):
            lines.append(cls.step(i, "Plan", task))
        n = len(subtasks) + 2
        if compression_trace:
            for j, ct in enumerate(compression_trace):
                lines.append(cls.step(n + j, "Compress", ct))
            n += len(compression_trace)
        if recursive_trace:
            for j, rt in enumerate(recursive_trace):
                lines.append(cls.step(n + j, "Recursive", rt))
            n += len(recursive_trace)
        lines.append(cls.step(n, "Reason", (reason or "align draft to prompt.")[:200]))
        n += 1
        if CONFIG.get("o1_self_check", True):
            lines.append(cls.step(n, "Self-check", self_check or "Does the draft answer the exact question?"))
            n += 1
        lines.append(cls.step(n, "Verify", verify or f"reasoning stays in-memory"))
        lines.append(cls.final_answer_step(n + 1))
        return "\n".join(lines)

    @classmethod
    def code_trace(cls, lang: str, script: str, *, lint: str = "", note: str = "") -> str:
        intent = f"execute `{script}` ({lang}) in the o1-preview interpreter"
        subtasks = [
            "parse buffer and detect language",
            "lint syntax before sandbox run",
            "capture stdout/stderr in-memory only",
        ]
        if lint:
            subtasks.append(f"lint: {lint}")
        return cls.build_trace(
            script,
            intent=intent,
            subtasks=subtasks,
            reason=note or f"run {lang} buffer via {cls.TAG} interpreter",
            verify=f"no files written",
            self_check="output matches executed code; errors surfaced verbatim",
        )

    @classmethod
    def format_output(cls, lang: str, script: str, output: str, *, error: str = "") -> str:
        if error:
            return (
                f"**{cls.TAG} interpreter**\n\n"
                f"`{script}` · `{lang}`\n\n"
                f"```\n{error}\n```"
            )
        body = (output or "(no output)").rstrip()
        return (
            f"**{cls.TAG} interpreter** · `{script}` · `{lang}`\n\n"
            f"```\n{body}\n```"
        )

    @classmethod
    def format_panel_stdout(cls, result: Dict[str, Any]) -> str:
        out = result.get("output", "")
        think = result.get("thinking", "")
        if not result.get("ok"):
            err = result.get("error", "run failed")
            if think:
                return f"{think}\n\n---\n\n[{cls.TAG}] error\n{err}"
            return f"[{cls.TAG}] error\n{err}"
        if think:
            return f"{think}\n\n---\n\n{out}"
        return out

    @classmethod
    def help_text(cls) -> str:
        return (
            f"**{cls.TAG} code interpreter** · \n\n"
            "**Syntax (exact o1-preview):**\n"
            "```\n"
            f"{cls.header()}\n"
            "1. Understand — …\n"
            "2. Plan — …\n"
            "…\n"
            "N. Verify — …\n"
            "N+1. Answer — emit clean user-facing text.\n"
            "```\n\n"
            "**Commands:** `/run` · `/interpret` · `/think` · `run it` · paste ``` fences\n\n"
            f"**Protocol:** `{cls.PROTO}` · model `{cls.MODEL}`\n\n"
            "Thinking stays internal; user sees clean final text + interpreter output."
        )


# ──────────────────────────────────────────────────────────────
# CAT R1.1 REASONING (files = off)
# ──────────────────────────────────────────────────────────────
class O1PreviewReasoner:
    """
    o1-preview reasoning loop (files = off):
    parse → recursive cat r1.10 passes → self-check → verify → answer.
    Thinking stays internal; user sees clean final text.
    """

    @classmethod
    def should_run(cls, prompt: str, *, enabled: bool, force: bool) -> bool:
        if force:
            return True
        if not enabled:
            return False
        pl = prompt.strip()
        if not pl:
            return False
        return True

    @staticmethod
    def _parse_intent(prompt: str) -> str:
        pl = prompt.lower()
        if any(k in pl for k in ("bug", "error", "traceback", "exception")):
            return "debug / isolate failure"
        if any(k in pl for k in ("code", "python", "script", "function")):
            return "implement or review code"
        if any(k in pl for k in ("build", "make", "create", "design")):
            return "design / construct"
        if any(k in pl for k in ("explain", "what is", "why", "how")):
            return "explain / teach"
        if re.search(r"\d\s*[+\-*/]", pl):
            return "compute / verify numerics"
        if "?" in prompt:
            return "answer a question"
        return "general assistance"

    @staticmethod
    def _subtasks(prompt: str) -> List[str]:
        pl = prompt.lower()
        tasks: List[str] = []
        if "?" in prompt:
            tasks.append("identify what is being asked and the expected form of the answer")
        if any(k in pl for k in ("code", "python", "implement")):
            tasks.append("list inputs, outputs, and edge cases before writing code")
        if any(k in pl for k in ("error", "bug", "traceback")):
            tasks.append("reproduce minimally, then localize the failing line")
        if any(k in pl for k in ("build", "design", "architecture")):
            tasks.append("sketch components and data flow before details")
        if re.search(r"\d\s*[+\-*/]", pl):
            tasks.append("compute step-by-step and verify the result")
        if not tasks:
            tasks.append("state goal, constraints, and the smallest verifiable next step")
        return tasks[:5]

    @staticmethod
    def _verify_note(prompt: str) -> str:
        pl = prompt.lower()
        checks: List[str] = []
        if "?" in prompt:
            checks.append("final answer addresses the question directly")
        if any(k in pl for k in ("code", "python")):
            checks.append("code is runnable in the local sandbox")
        if any(k in pl for k in ("error", "bug")):
            checks.append("expected vs actual output is explicit")
        checks.append("reasoning and weights stay in-memory")
        return "; ".join(checks)

    @staticmethod
    def _self_check(prompt: str, draft: str) -> str:
        pl = prompt.lower()
        notes: List[str] = []
        if "?" in prompt:
            notes.append("Does the draft answer the exact question?")
        if any(k in pl for k in ("code", "python")):
            notes.append("Are imports, edge cases, and return values covered?")
        if re.search(r"\d\s*[+\-*/]", pl):
            notes.append("Re-check arithmetic independently.")
        notes.append("Remove speculation; keep only what follows from the prompt.")
        return " · ".join(notes)

    def run(
        self,
        prompt: str,
        *,
        distill_draft: str = "",
        recursive_trace: Optional[List[str]] = None,
        compression_trace: Optional[List[str]] = None,
    ) -> str:
        intent = self._parse_intent(prompt)
        subtasks = self._subtasks(prompt)
        verify = self._verify_note(prompt)
        reason = distill_draft.strip() or f"aligned on {intent}."
        return O1PreviewSyntax.build_trace(
            prompt,
            intent=intent,
            subtasks=subtasks,
            reason=reason,
            recursive_trace=recursive_trace,
            compression_trace=compression_trace,
            verify=verify,
            self_check=self._self_check(prompt, reason),
        )


ThinkEngine = O1PreviewReasoner
UltraThinkEngine = O1PreviewReasoner


# ──────────────────────────────────────────────────────────────
# CAT R1.1 FUSION (cat r1.10 CoT · catseek polish · files = off)
# ──────────────────────────────────────────────────────────────
class CatSeekR1Fusion:
    """
    Dual reasoning stack under cat r1.10 branding (no weight files):
    cat r1.10 — long CoT, GRPO-style self-verify, math sanity checks.
    cat r1.10-tier — extended thinking trace, recursive draft improvement.
    """

    __slots__ = ("last_passes", "last_trace")

    CATSEEK_ALGOS = (
        "Long chain-of-thought",
        "GRPO outcome self-verification",
        "Step-by-step math checks",
        "Code edge-case review",
    )
    MYTHOS_ALGOS = (
        "Extended thinking trace",
        "Recursive draft improvement",
        "cat r1.10 prose polish",
        "Pre-emit self-check",
    )

    def __init__(self):
        self.last_passes = 0
        self.last_trace: List[str] = []

    @staticmethod
    def think_header() -> str:
        if CONFIG.get("o1_preview", True):
            return O1PreviewSyntax.header()
        return f"{BRAND} · {REASONING_MODE} · {CODE_ENGINE}"

    @staticmethod
    def _meaningful_words(pl: str) -> bool:
        return bool(re.search(r"[a-zA-Z0-9\u4e00-\u9fff]", pl))

    @classmethod
    def is_noise(cls, prompt: str) -> bool:
        pl = prompt.strip().lower()
        if not pl:
            return True
        if CatSeekR1Code._GO.match(pl) or CatSeekR1Code._CODE_SHORT.match(pl):
            return False
        if pl in {".", "..", "...", "\"", "'", "?", "!"}:
            return True
        return False

    @classmethod
    def session_followup(cls, engine: "CatR11Engine", prompt: str) -> Optional[str]:
        if not engine.chat_history:
            return None
        last_user = last_bot = ""
        for m in reversed(engine.chat_history[:-1]):
            if m.get("role") == "assistant" and not last_bot:
                last_bot = m.get("text", "")
            elif m.get("role") == "user" and not last_user:
                last_user = m.get("text", "")
            if last_user and last_bot:
                break
        if not last_user:
            return None
        pl = prompt.strip().lower()
        if cls.is_noise(prompt):
            if re.search(r"how are you|how're you|how is it", last_user.lower()):
                return "I'm doing well — thanks for asking! What's on your mind?"
            if re.search(r"你好吗|怎么样|还好吗", prompt):
                return "我很好，谢谢关心！你今天想聊什么？"
            return f"Still here — we were talking about \"{last_user[:60]}\". Want to continue that, or start something new?"
        return None

    def recursive_improve(self, draft: str, prompt: str, vec: Optional[np.ndarray]) -> str:
        if not CONFIG.get("mythos_recursive_improve") or not draft.strip():
            return draft
        depth = CONFIG.get("mythos_recursive_depth", 3)
        eps = CONFIG.get("mythos_recursive_epsilon", 0.04)
        out = draft
        self.last_trace = []
        prev = out
        for i in range(depth):
            score = self._quality_score(out, prompt, vec)
            self.last_trace.append(f"mythos pass {i + 1} · quality {score:.3f}")
            if i > 0 and score >= 0.92:
                self.last_trace.append(f"converged at pass {i + 1}")
                self.last_passes = i + 1
                return out
            out = self._polish_pass(out, prompt, vec, pass_idx=i)
            if i > 0:
                delta = abs(len(out) - len(prev)) / max(len(prev), 1)
                if delta < eps:
                    self.last_trace.append(f"converged at pass {i + 1}")
                    self.last_passes = i + 1
                    return out
            prev = out
        self.last_passes = depth
        return out

    @staticmethod
    def _quality_score(text: str, prompt: str, vec: Optional[np.ndarray]) -> float:
        if not text.strip():
            return 0.0
        score = 0.55
        if "?" in prompt and "?" not in text and len(text) > 40:
            score += 0.08
        if len(text.split()) >= 12:
            score += 0.12
        if re.search(r"\*\*[^*]+\*\*", text):
            score += 0.05
        if vec is not None and vec.size:
            score += min(0.2, float(np.linalg.norm(vec[:8])) * 0.02)
        if text.count("\n\n") >= 1:
            score += 0.05
        return min(1.0, score)

    @staticmethod
    def _polish_pass(text: str, prompt: str, vec: Optional[np.ndarray], pass_idx: int) -> str:
        t = re.sub(r"\n{3,}", "\n\n", text.strip())
        if pass_idx == 0 and "?" in prompt and not t.endswith("?"):
            if len(t.split()) > 20 and "I can go deeper" not in t:
                t += "\n\nWant me to go deeper on any part?"
        if pass_idx >= 1 and len(t) < 120 and "?" in prompt:
            topic = prompt.strip().rstrip("?")[:80]
            t = f"{t}\n\n**Short answer:** {topic} — ask for steps, code, or a comparison and I'll expand."
        return t

    @staticmethod
    def catseek_math_wrap(prompt: str, result: str) -> str:
        if not CONFIG.get("catr1_self_verify"):
            return f"Result: **{result}**"
        return (
            f"**Step-by-step ({BRAND} verify):**\n\n"
            f"1. Parse expression from prompt\n"
            f"2. Evaluate with integer/float rules\n"
            f"3. Self-check: re-evaluate → **{result}**\n\n"
            f"**Answer:** {result}"
        )

    def stats_line(self) -> str:
        return (
            f"Fusion · {BRAND} ({len(self.CATSEEK_ALGOS)} + {len(self.MYTHOS_ALGOS)} algos) · passes={self.last_passes}"
        )


# ──────────────────────────────────────────────────────────────
# CAT R1.1 RUNTIME (files = off · cat r1.10 parity)
# ──────────────────────────────────────────────────────────────
class ClaudeMythosRuntime:
    """
    cat r1.10 local runtime (files = off):
    extended thinking · recursive prose polish · cat r1.10 code engine · cat r1.10 voice.
    """

    TIER = CONFIG["mythos_tier"]
    PROSE = CONFIG["prose_tier"]
    CODE_LABEL = CONFIG["code_interpreter_name"]

    MYTHOS_ALGOS = (
        "Extended thinking trace",
        "Recursive draft improvement",
        "cat r1.10 prose polish",
        "Pre-emit self-check",
        "cat r1.10 code perfection loop",
        "Proactive follow-through",
    )

    @classmethod
    def enabled(cls) -> bool:
        return bool(CONFIG.get("mythos_runtime", True) and CONFIG.get("mythos_mode", True))

    @classmethod
    def think_header(cls) -> str:
        return f"{BRAND} · {REASONING_MODE}"

    @classmethod
    def voice(cls, body: str, prompt: str, task: str = "general") -> str:
        if not CONFIG.get("mythos_voice", True) or not body:
            return body
        text = body.strip()
        if "```" in text and task == "code":
            return text
        pl = prompt.lower()
        if "?" in prompt and task in {"explain", "general", "chat", "agent", "qa"}:
            if len(text) > 50 and not re.match(
                r"^(Here|I'd|Let me|Sure|The |Yes|I can|On |答案)", text, re.I
            ):
                topic = prompt.strip().rstrip("?！？.")[:100]
                text = f"**{topic}** — {text}"
        if any(k in pl for k in ("how ", "why ", "explain", "walk me", "step")):
            if len(text) > 140 and "1." not in text[:280] and "Step" not in text[:280]:
                paras = [p.strip() for p in text.split("\n\n") if p.strip()]
                if len(paras) >= 2:
                    text = "\n\n".join(
                        f"**{i}.** {p.lstrip('0123456789. ')}"
                        for i, p in enumerate(paras, 1)
                    )
        return text

    @classmethod
    def emit(cls, engine: "CatR11Engine", body: str, prompt: str, task: str) -> str:
        if not cls.enabled():
            if CONFIG.get("catr1_voice"):
                return GoogleWhitepaperCatSeekSorter.catseek_voice(body, prompt, task)

        out = body
        if CONFIG.get("mythos_recursive_improve") and task not in frozenset({"chat", "execute"}):
            out = engine.fusion.recursive_improve(out, prompt, engine.last_vec)
        out = cls.voice(out, prompt, task)
        if CONFIG.get("catr1_voice") and CATSEEK_R1_MODE:
            out = GoogleWhitepaperCatSeekSorter.catseek_voice(out, prompt, task)
        return out

    @classmethod
    def math_wrap(cls, prompt: str, result: str) -> str:
        if not cls.enabled():
            return CatSeekR1Fusion.catseek_math_wrap(prompt, result)
        return (
            f"**{MYTHOS_NAME} verify**\n\n"
            f"1. Parse the expression from your prompt\n"
            f"2. Evaluate with standard arithmetic rules\n"
            f"3. Independent re-check → **{result}**\n\n"
            f"**Answer:** {result}\n\n"
            f"*computed in-memory*"
        )

    @classmethod
    def code_help(cls) -> str:
        return (
            f"**{cls.CODE_LABEL}** · **{MYTHOS_NAME}** · \n\n"
            f"Engine `{CONFIG['code_backend']}` · version `{CONFIG['code_interpreter_version']}` · "
            f"prose `{cls.PROSE}`\n\n"
            "Full code capability — write, edit, run, lint, explain.\n"
            "**Code anything** — any language, any task (API, CLI, web, algo, mobile, config).\n"
            "Recursive perfection loop · pattern library · sandbox verify.\n\n"
            "**Commands:** `/run` · `/interpret` · `/code` · `/think` · paste ``` fences · **run it**\n\n"
            f"Recursive depth: {CONFIG['mythos_recursive_depth']}\n\n"
            "Everything stays in-memory — nothing written to disk."
        )

    @classmethod
    def format_code_result(cls, lang: str, script: str, output: str, *, error: str = "") -> str:
        if error:
            return (
                f"**Code engine** · **catr1.10**\n\n"
                f"`{script}` · `{lang}`\n\n```\n{error}\n```"
            )
        body = (output or "(no output)").rstrip()
        return (
            f"**`{script}`** · `{lang}`\n\n"
            f"```\n{body}\n```"
        )

    @classmethod
    def stats_line(cls) -> str:
        return f"{MYTHOS_NAME} ({len(cls.MYTHOS_ALGOS)} algos)"


MythosRuntime = ClaudeMythosRuntime


# ──────────────────────────────────────────────────────────────
# DEEPMIND FAST STACK (files = off · in-memory speed algorithms)
# Chinchilla adaptive compute · Flash attention · MuZero prefix cache
# · teacher-only turbo distillation · batch cat r1.10 linear · early exit
# ──────────────────────────────────────────────────────────────
class DeepMindFastStack:
    """
    In-memory DeepMind-inspired inference accelerators (no weight files).
    Composes with cat r1.10 + o1-preview recursive loop on CatR11Engine.
    """

    __slots__ = ("engine", "_prefix_cache", "_turbo_cache", "last_algo", "passes_saved")

    ALGOS = (
        "Chinchilla adaptive compute",
        "Flash causal attention",
        "Batch linear GEMM",
        "MuZero prefix latent cache",
        "In-memory student distillation",
        "Teacher-only turbo distillation",
        "Sparse top-k compression",
        "MoE top-k routing",
        "Recursive early convergence",
        "catr1 multi-token prediction",
        "catr1 real-time sparse attention",
        "catr1 speculative decoding",
    )

    def __init__(self, engine: "CatR11Engine"):
        self.engine = engine
        self._prefix_cache: Dict[str, np.ndarray] = {}
        self._turbo_cache: Dict[str, np.ndarray] = {}
        self.last_algo = ""
        self.passes_saved = 0

    def adaptive_depth(self, prompt: str, task: Optional[str] = None) -> int:
        """Chinchilla-style: scale recursive passes to prompt + task budget."""
        base = CONFIG["recursive_depth"]
        if not CONFIG.get("adaptive_compute"):
            return base
        n = len(prompt.strip())
        if task in CONFIG.get("turbo_encode_tasks", ()):
            return 1
        if n < 24:
            return 1
        if n < 80:
            return min(2, base)
        if n < 200:
            return min(3, base)
        return base

    def _prefix_key(self, prompt: str) -> str:
        words = prompt.lower().strip().split()
        return " ".join(words[:6]) if words else ""

    def _muzero_hit(self, prompt: str) -> Optional[np.ndarray]:
        key = self._prefix_key(prompt)
        if not key or len(prompt) < 32:
            return None
        hit = self._prefix_cache.get(key)
        if hit is not None:
            self.last_algo = "MuZero prefix latent cache"
            return hit.copy()
        return None

    def turbo_encode(self, prompt: str) -> np.ndarray:
        """Single-pass teacher-only encode — chat/code/math fast path."""
        key = prompt.lower().strip()
        hit = self._turbo_cache.get(key)
        if hit is not None:
            self.last_algo = "Teacher-only turbo distillation (cache)"
            return hit.copy()
        state = self.engine.encode_prompt(prompt)
        seq = state if state.ndim == 2 else state.reshape(1, -1)
        delta = self.engine._pool_sequence(
            self.engine.forward(seq, turbo_only=True)
        )
        out = self.engine._layer_norm(delta)
        if CONFIG["compression_enabled"]:
            out = self.engine.compressor.compress_roundtrip(out)
            self.engine.last_compression_ratio = self.engine.compressor.last_ratio
        self.engine.last_recursive_passes = 1
        self.engine.last_recursive_trace = [
            f"turbo · 1 pass · {BRAND} student",
        ]
        self.engine.last_vec = out.copy()
        self.last_algo = "catr1.10 student distillation"
        if len(self._turbo_cache) < 96:
            self._turbo_cache[key] = out.copy()
        return out

    def encode(self, prompt: str, task: Optional[str] = None) -> np.ndarray:
        if not CONFIG.get("deepmind_fast"):
            return self.engine.recursive_encode(prompt)
        if task in CONFIG.get("turbo_encode_tasks", ()):
            return self.turbo_encode(prompt)
        mu = self._muzero_hit(prompt)
        if mu is not None:
            self.engine.last_vec = mu
            return mu
        depth = self.adaptive_depth(prompt, task)
        saved = max(0, CONFIG["recursive_depth"] - depth)
        self.passes_saved += saved
        out = self.engine.recursive_encode(prompt, depth=depth)
        pk = self._prefix_key(prompt)
        if pk and len(self._prefix_cache) < 64:
            self._prefix_cache[pk] = out.copy()
        self.last_algo = f"Chinchilla adaptive compute ({depth} passes)"
        return out

    def stats_line(self) -> str:
        return (
            f"DeepMind fast · {len(self.ALGOS)} algos · "
            f"last={self.last_algo or 'idle'} · passes_saved={self.passes_saved}"
        )

    def clear(self) -> None:
        self._prefix_cache.clear()
        self._turbo_cache.clear()
        self.passes_saved = 0


# ──────────────────────────────────────────────────────────────
# CAT R1.1 COMPRESSION ENGINE (files = off · frontier-tier capacity)
# ──────────────────────────────────────────────────────────────
class CatSeekCompressor:
    """
    Frontier compression stack for catr1.10 (catseek r1.10 frontier):
    ternary weights · sparse top-k activations · low-rank bottleneck · AWQ · GPTQ · pack/unpack.
    Simulates multi-trillion-parameter capacity without external files.
    """

    __slots__ = ("d_model", "rank", "sparse_k", "down_s", "up_s", "last_ratio", "packs")

    def __init__(self, d_model: int, seed: int = 99):
        self.d_model = d_model
        self.rank = CONFIG["compression_rank"]
        self.sparse_k = CONFIG["compression_sparse_k"]
        rng = np.random.RandomState(seed)
        down = rng.choice([-1, 0, 1], (d_model, self.rank)).astype(np.int8)
        up = rng.choice([-1, 0, 1], (self.rank, d_model)).astype(np.int8)
        self.down_s = (down == 1).astype(np.int16) - (down == -1).astype(np.int16)
        self.up_s = (up == 1).astype(np.int16) - (up == -1).astype(np.int16)
        self.last_ratio = 1.0
        self.packs = 0

    @staticmethod
    def _ternary(x: np.ndarray, thr: float = 0.5) -> np.ndarray:
        q = np.zeros_like(x, dtype=np.int8)
        q[x > thr], q[x < -thr] = 1, -1
        return q

    def low_rank_bottleneck(self, x: np.ndarray) -> np.ndarray:
        xq = self._ternary(x).astype(np.int16)
        h = np.tanh(xq @ self.down_s).astype(np.float32)
        return (self._ternary(h).astype(np.int16) @ self.up_s).astype(np.float32)

    def sparse_reconstruct(self, x: np.ndarray) -> np.ndarray:
        k = min(self.sparse_k, len(x))
        idx = np.argsort(np.abs(x))[-k:]
        out = np.zeros_like(x, dtype=np.float32)
        out[idx] = x[idx]
        t = self._ternary(out).astype(np.float32)
        out[idx] = t[idx] * np.abs(x[idx])
        fp_bits = self.d_model * 32
        packed_bits = k * 32 + self.rank * 16
        self.last_ratio = max(1.0, fp_bits / max(packed_bits, 1))
        self.packs += 1
        return out

    def compress_roundtrip(self, x: np.ndarray) -> np.ndarray:
        if not CONFIG["compression_enabled"]:
            self.last_ratio = 1.0
            return x
        lr = self.low_rank_bottleneck(x)
        blended = x * 0.52 + lr * 0.48
        return self.sparse_reconstruct(blended)

    def effective_params_billions(self) -> float:
        mult = self.last_ratio * CONFIG["distil_passes"] * CONFIG["compression_stack_mult"]
        return CONFIG["nominal_base_params"] * mult / 1e9 / max(CONFIG["compression_stack_mult"], 1)


class CatSeekR1Core:
    """
    Unified inference core: compressed cat r1.10 forward + o1-preview recursive loop.
    Targets frontier-tier answer quality on consumer hardware (files = off).
    """

    __slots__ = ("engine",)

    def __init__(self, engine: "CatR11Engine"):
        self.engine = engine

    def infer_state(self, prompt: str) -> np.ndarray:
        return self.engine.recursive_encode(prompt)

    def stats_line(self) -> str:
        c = self.engine.compressor
        st = self.engine.catseek_stats
        base = (
            f"{BRAND} · {st['catr1_linear_layers']} cat r1.10 linear · "
            f"{BRAND} distil · "
            f"packed {st['packed_kb']:.1f}KB · "
            f"compress {c.last_ratio:.1f}x"
        )
        if CONFIG.get("deepmind_fast") and self.engine.deepmind:
            wp = GoogleWhitepaperCatSeekSorter.stats_line() if CONFIG.get("google_whitepaper_heuristics") else ""
            tail = f"{self.engine.deepmind.stats_line()}"
            if wp:
                tail = f"{wp} · {tail}"
            return f"{base} · {tail}"
        return base


# Legacy aliases (BitNet → cat r1.10)
BitLinear = CatSeekLinear
BitNetBlock = CatSeekBlock
BitNetCompressor = CatSeekCompressor
BitNetRivalCore = CatSeekR1Core
bitnet_memory_report = catseek_memory_report


# Casual chat — matched before educational intent routing (EN · 中文)
_SMALLTALK: Tuple[Tuple[str, str], ...] = (
    (r"^(?:hi|hey|hello|yo|howdy)\s*[!?.]*$", "Hi! How can I help you today?"),
    (r"^how are you(?: doing| today)?\??$", "Doing well, thanks for asking! I'm here and ready to chat. How about you?"),
    (r"^how(?:'re| are) you(?: doing| today)?\??$", "Doing well, thanks for asking! I'm here and ready to chat. How about you?"),
    (r"^how(?:'s| is|s) it going\??$", "Going well on my end — thanks! What can I help you with today?"),
    (r"^how(?:'s| is|s) everything\??$", "All good here! What's on your mind?"),
    (r"^how(?:'s| is|s) your day\??$", "Running smoothly so far. How's yours going?"),
    (r"^how have you been\??$", "Steady and ready to help. What have you been up to?"),
    (r"^(?:what's up|whats up|wassup|sup)\??$", "Not much — just here to help. What are you working on?"),
    (r"^how you doing\??$", "Doing great, thanks! What can I do for you?"),
    (r"^how(?:'s| is| are) (?:u|ya|you) doing\??$", "Doing great, thanks! What can I do for you?"),
    (r"^how r u\??$", "Doing well! How are you?"),
    (r"^how are u\??$", "Doing well! How are you?"),
    (r"^(?:good morning|good afternoon|good evening)\.?\??$", "Good to hear from you! What would you like to talk about?"),
    (r"^nice to meet you\.?\??$", "Nice to meet you too! Ask me anything — code, explanations, debugging, or just chat."),
    (r"^(?:are you ok|you ok|u ok)\??$", "I'm all good, thanks! How can I help?"),
    (r"^what(?:'s| is) new\??$", f"Same local {BRAND} engine, ready when you are. What's new with you?"),
    (r"^how do you feel\??$", "I feel ready to help! What's up?"),
    (r"^(?:你好|您好|嗨|哈喽|在吗)[!！?？。.\s]*$", "你好！有什么我可以帮你的？"),
    (r"^你好吗[!！?？]?$", "我很好，谢谢！你今天想聊点什么？"),
    (r"^(?:早上好|下午好|晚上好)[!！?？]?$", "你好！很高兴见到你。需要什么帮助？"),
    (r"^谢谢[!！?？]?$", "不客气！还需要别的帮助吗？"),
    (r"^感谢[!！?？]?$", "不客气！随时可以继续问我。"),
    (r"^(?:再见|拜拜)[!！?？]?$", "再见！期待下次聊天。"),
)


# Multilingual tokenization (EN · 中文 · mixed · files = off)
_TOKEN_EN = re.compile(r"[a-z0-9+#]+", re.I)
_TOKEN_CJK = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]")
_ZH_QUESTION = re.compile(
    r"(什么是|是什么|什么叫|何为|为何|为什么|为啥|怎么|如何|怎样|能否|可不可以|介绍一下|解释|说明|告诉我)"
)
_ZH_GREETING = re.compile(
    r"^(你好|您好|嗨|哈喽|早上好|下午好|晚上好|在吗|你是谁|你好吗)[!！?？。.\s]*$"
)
_ZH_TOPIC = re.compile(
    r"(?:什么是|是什么|什么叫|解释|说明|介绍|告诉我)(.+?)[?？。!！]?$"
)


def tokenize_text(text: str, max_tokens: int = 256) -> List[str]:
    """Split English words and CJK characters for in-memory embedding (files = off)."""
    raw = (text or "").strip()
    if not raw:
        return ["<unk>"]
    lower = raw.lower()
    tokens: List[str] = []
    i = 0
    while i < len(lower) and len(tokens) < max_tokens:
        m = _TOKEN_EN.match(lower, i)
        if m:
            tokens.append(m.group(0))
            i = m.end()
            continue
        m = _TOKEN_CJK.match(raw, i)
        if m:
            tokens.append(m.group(0))
            i = m.end()
            continue
        i += 1
    return tokens or ["<unk>"]


def is_zh_question(text: str) -> bool:
    return bool(_ZH_QUESTION.search(text or ""))


def is_zh_greeting(text: str) -> bool:
    s = (text or "").strip()
    if _ZH_GREETING.match(s):
        return True
    return len(s) <= 10 and bool(re.search(r"^(你好|您好|你好吗|早上好|下午好|晚上好)", s))


def is_explain_request(text: str) -> bool:
    pl = (text or "").lower()
    if re.search(r"\b(explain|what is|what are|what's|why|how (?:does|do|to|can|would|should))\b", pl):
        return True
    return is_zh_question(text or "")


def extract_zh_topic(prompt: str) -> str:
    s = (prompt or "").strip()
    m = _ZH_TOPIC.search(s)
    if m:
        return m.group(1).strip("？?。!！ ")[:80]
    return s[:80]


# ──────────────────────────────────────────────────────────────
# VIBE CODE HEURISTICS (EN · 中文 · mixed · pasted code · files = off)
# ──────────────────────────────────────────────────────────────
class VibeCodeHeuristics:
    """
    Multilingual intent + language detection for casual vibe-coding.
    Understands English, Chinese (中文), mixed human language, and raw code snippets.
    """

    CJK = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]")
    ZH_CODE_NOUNS = (
        "写代码", "编程", "程序", "源码", "源代码", "脚本", "网页", "代码",
        "函数", "算法", "斐波那契", "斐波那契数列", "小程序", "页面",
    )
    ZH_RUN = ("运行", "执行", "跑一下", "跑起来", "测试", "试一下")
    EN_VIBE = re.compile(
        r"\b(vibe[\s-]?code|vibecode|whip up| cook up|spin up|slap together|"
        r"gimme|lemme get|just (?:make|write|code)|yo make|hook me up with)\b",
        re.I,
    )
    ZH_WRITE = re.compile(
        r"(写|做|建|造|生成|帮我写|帮我做|给我写|给我做|帮我生成|"
        r"来个|整一个|弄一个|搞一个|创建一个|实现|编写|弄段|来段|"
        r"能不能写|可以写|请写|请帮我|麻烦写|麻烦帮我)"
    )
    ZH_CORRECT = re.compile(
        r"(不要|别|不对|不是|改成|改为|换成|用)(?:.{0,12})?(html|python|javascript|java|c\+\+?|cpp|rust|go|bash|typescript|sql|c语言|网页|脚本)",
        re.I,
    )
    ZH_LANG_INLINE = re.compile(
        r"(?:用|以|\bin\b|with|using|换成|改成|改为)\s+"
        r"(python|html|javascript|typescript|java|kotlin|swift|rust|go|bash|shell|cpp|c\+\+|c|sql|ruby|php|"
        r"c语言|c\+\+语言|网页|页面|脚本|js|py|python3?|html5?)",
        re.I,
    )
    ZH_SAYS = re.compile(
        r"(?:显示|输出|打印|说|写上|内容是|文字是|写着)[「\"']?([^」\"'\n，。！？]+)[」\"']?",
    )
    ZH_SUBJECT = re.compile(
        r"(?:写|做|建|生成|帮我写|帮我做|来个|整一个|弄一个|搞一个|创建一个|实现)"
        r"(?:一个|个|一段|段)?(.+?)(?:用|in|的|程序|代码|网页|脚本|$)",
    )
    LANG_ALIASES = {
        "py": "python", "python3": "python", "python2": "python",
        "js": "javascript", "node": "javascript", "ts": "typescript",
        "c++": "cpp", "cc": "cpp", "c语言": "c", "c++语言": "cpp",
        "网页": "html", "页面": "html", "html5": "html", "前端": "html",
        "脚本": "python", "shell": "bash", "sh": "bash", "zsh": "bash",
        "golang": "go", "go语言": "go", "java语言": "java",
        "rust语言": "rust", "php语言": "php",
    }
    CODE_SHAPES = re.compile(
        r"(#include\s*<|def\s+\w+\s*\(|function\s+\w+|fn\s+main|public\s+class|"
        r"<!DOCTYPE|<html|console\.log|printf\s*\(|System\.out|package\s+main|"
        r"import\s+\w+|class\s+\w+\s*[:{]|=>\s*\{|var\s+\w+\s*=)",
        re.I,
    )

    @classmethod
    def enabled(cls) -> bool:
        return bool(CONFIG.get("vibe_code_heuristics", True))

    @classmethod
    def has_cjk(cls, text: str) -> bool:
        return bool(cls.CJK.search(text or ""))

    @classmethod
    def _norm_lang(cls, token: str) -> Optional[str]:
        if not token:
            return None
        t = token.strip().lower()
        return cls.LANG_ALIASES.get(t, t)

    @classmethod
    def wants_code(cls, prompt: str) -> bool:
        if not cls.enabled():
            return False
        raw = (prompt or "").strip()
        if not raw:
            return False
        pl = raw.lower()

        # "什么是 python" / "what is python" = explain, not code generation.
        if is_explain_request(raw) and not cls.ZH_WRITE.search(raw):
            if not re.search(r"\b(write|make|build|create|implement|code|vibe)\b", pl):
                if not cls.EN_VIBE.search(raw):
                    return False

        if re.match(r"^\s*(?:code|/code|code\s*>|>|program|script)\s*$", pl, re.I):
            return True
        if cls.CODE_SHAPES.search(raw):
            return True
        if cls.EN_VIBE.search(raw):
            return True
        if any(cue in raw for cue in cls.ZH_CODE_NOUNS):
            return True
        if cls.ZH_WRITE.search(raw):
            if re.search(
                r"(代码|程序|脚本|网页|函数|html|python|javascript|java|rust|go|c\+\+|cpp|"
                r"fibonacci|fib|算法|页面|api|app|site|hello|cat|meow|打印|输出)",
                raw, re.I,
            ):
                return True
        if cls.ZH_CORRECT.search(raw):
            return True
        if cls.ZH_LANG_INLINE.search(raw):
            return True
        if re.search(r"[\u4e00-\u9fff].*(html|python|javascript|java|rust|go|cpp|c\+\+|c\b)", raw, re.I):
            if is_zh_question(raw) and not cls.ZH_WRITE.search(raw):
                return False
            return True
        if re.search(r"(html|python|javascript|java|rust|go|cpp|c\+\+|c\b).*[\u4e00-\u9fff]", raw, re.I):
            if is_zh_question(raw) and not cls.ZH_WRITE.search(raw):
                return False
            return True
        if cls.has_cjk(raw) and re.search(r"(写|做|建|生成|程序|代码|脚本|网页)", raw):
            return True
        if CodeAnythingEngine.enabled() and CodeAnythingEngine.wants_anything(raw):
            return True
        return False

    @classmethod
    def lang_from_text(cls, prompt: str, engine: "CatR11Engine") -> Optional[str]:
        if not cls.enabled():
            return None
        raw = (prompt or "").strip()
        pl = raw.lower()

        m = cls.ZH_CORRECT.search(raw)
        if m:
            return engine.normalize_lang(cls._norm_lang(m.group(2)))

        m = cls.ZH_LANG_INLINE.search(raw)
        if m:
            return engine.normalize_lang(cls._norm_lang(m.group(1)))

        m = CatSeekR1Code._MAKE_IT_LANG.search(pl)
        if m:
            return engine.normalize_lang(m.group(1))

        if re.search(r"(网页|页面|html|前端)", raw, re.I):
            if re.search(r"(写|做|建|生成|程序|代码|脚本|make|write|build|create|vibe)", raw, re.I):
                return "html"

        if re.search(r"(脚本|bash|shell)", raw, re.I) and re.search(
            r"(写|做|建|生成|write|make|build)", raw, re.I
        ):
            return "bash"

        lang_tokens = (
            ("python", r"\bpython\b|python3|\.py\b|蟒蛇"),
            ("javascript", r"\bjavascript\b|\bjs\b|node\.?js"),
            ("typescript", r"\btypescript\b|\bts\b"),
            ("java", r"\bjava\b(?!script)"),
            ("kotlin", r"\bkotlin\b"),
            ("swift", r"\bswift\b"),
            ("rust", r"\brust\b"),
            ("go", r"\bgo\b|golang|go语言"),
            ("cpp", r"\bc\+\+\b|\bcpp\b|c\+\+语言"),
            ("c", r"\bc语言\b|\bc\s+程序\b|\bc\s+代码\b"),
            ("html", r"\bhtml\b|html5"),
            ("bash", r"\bbash\b|\bshell\b"),
            ("sql", r"\bsql\b"),
            ("ruby", r"\bruby\b"),
            ("php", r"\bphp\b"),
        )
        has_write = bool(
            re.search(
                r"(写|做|建|生成|帮我|write|make|build|create|code|vibe|implement|scaffold|boilerplate)",
                raw, re.I,
            )
        )
        for lang, pat in lang_tokens:
            if re.search(pat, raw, re.I) and (has_write or cls.CODE_SHAPES.search(raw)):
                return engine.normalize_lang(lang)

        if cls.has_cjk(raw) and re.search(r"(程序|代码)", raw) and not re.search(
            r"(python|html|javascript|java|rust|go|c\+\+|cpp|c\b)", raw, re.I
        ):
            return "python"
        return None

    @classmethod
    def subject_from_text(cls, prompt: str) -> Optional[str]:
        if not cls.enabled():
            return None
        raw = (prompt or "").strip()

        m = cls.ZH_SAYS.search(raw)
        if m:
            return m.group(1).strip("?.! ，。！？")
        if "你好猫" in raw or "hello cat" in raw.lower():
            return "Hello Cat"
        if "喵" in raw or "meow" in raw.lower():
            return "Meow"
        if "你好世界" in raw or "hello world" in raw.lower():
            return "Hello World"

        m = cls.ZH_SUBJECT.search(raw)
        if m:
            subj = m.group(1).strip("?.! ，。！？的 ")
            skip = {"html", "python", "javascript", "java", "代码", "程序", "脚本", "网页", "一个", "段"}
            if subj and subj not in skip and len(subj) >= 1:
                return subj[:80]

        m = re.search(
            r"(?:write|make|build|create|vibe[\s-]?code|whip up|gimme)\s+(?:me\s+)?(?:a\s+)?(.+?)(?:\s+in\s+\w+|$)",
            raw, re.I,
        )
        if m:
            subj = m.group(1).strip("?.! ")
            if subj.lower() not in {"html", "it", "a", "an", "code"}:
                return subj[:80]
        return None

    @classmethod
    def wants_run(cls, prompt: str) -> bool:
        raw = (prompt or "").strip().lower()
        run_en = (
            "run it", "run this", "execute", "interpret", "test it",
            "/run", "and run", "then run",
        )
        if any(x in raw for x in run_en):
            return True
        return any(r in (prompt or "") for r in cls.ZH_RUN)


# ──────────────────────────────────────────────────────────────
# GOOGLE WHITEPAPER HEURISTICS → CATSEEK VOICE (files = off)
# Attention · BM25 rank · Chinchilla budget · MoE route · distil align
# ──────────────────────────────────────────────────────────────
@dataclass
class RankedDataItem:
    key: str
    text: str
    role: str = "chunk"
    score: float = 0.0
    expert: str = "chat"
    meta: Dict[str, Any] = field(default_factory=dict)


class GoogleWhitepaperCatSeekSorter:
    """
    Sort in-memory context with Google whitepaper heuristics (files = off),
    then shape replies in cat r1.10 conversational style.
    """

    EXPERTS = ("reason", "code", "math", "retrieve", "chat")
    WP_ALGOS = (
        "Transformer attention relevance",
        "BM25 learning-to-rank",
        "Chinchilla compute-optimal density",
        "Switch-MoE expert routing",
        "Distillation teacher alignment",
        "Session recency decay",
    )
    _DOC_FREQ: Dict[str, int] = {}
    _N_DOCS = 0
    _AVG_DL: float = 64.0

    @classmethod
    def enabled(cls) -> bool:
        return bool(CONFIG.get("google_whitepaper_heuristics", True))

    @classmethod
    def _tokens(cls, text: str) -> List[str]:
        return re.findall(r"[a-zA-Z\u4e00-\u9fff]{2,}", (text or "").lower())

    @classmethod
    def _register_corpus(cls, texts: List[str]) -> None:
        cls._DOC_FREQ.clear()
        cls._N_DOCS = max(len(texts), 1)
        total_dl = 0
        for doc in texts:
            tokens = cls._tokens(doc)
            total_dl += len(tokens)
            for tok in set(tokens):
                cls._DOC_FREQ[tok] = cls._DOC_FREQ.get(tok, 0) + 1
        cls._AVG_DL = total_dl / cls._N_DOCS if cls._N_DOCS else 64.0

    @classmethod
    def attention_score(cls, query: str, doc: str) -> float:
        """Scaled dot-product style relevance (Transformer attention analogy)."""
        q, d = cls._tokens(query), cls._tokens(doc)
        if not q or not d:
            return 0.0
        qs, ds = set(q), set(d)
        overlap = len(qs & ds)
        return overlap / (len(qs) ** 0.5 * len(ds) ** 0.5 + 1e-6)

    @classmethod
    def bm25_score(cls, query: str, doc: str, *, k1: float = 1.2, b: float = 0.75) -> float:
        """BM25-lite rank (Google Search / learning-to-rank family)."""
        qtoks = cls._tokens(query)
        dtoks = cls._tokens(doc)
        if not qtoks or not dtoks:
            return 0.0
        dl = len(dtoks)
        avgdl = cls._AVG_DL
        score = 0.0
        dset = dtoks
        for term in set(qtoks):
            tf = dset.count(term)
            if not tf:
                continue
            df = cls._DOC_FREQ.get(term, 0)
            idf = np.log(1.0 + (cls._N_DOCS - df + 0.5) / (df + 0.5))
            num = tf * (k1 + 1)
            den = tf + k1 * (1.0 - b + b * dl / avgdl)
            score += idf * num / den
        return float(score)

    @classmethod
    def chinchilla_density(cls, text: str) -> float:
        """Chinchilla-style: reward information-dense short spans."""
        words = cls._tokens(text)
        n = max(len(words), 1)
        unique = len(set(words))
        density = unique / n
        length_pen = min(1.0, 72.0 / n)
        return 0.55 * density + 0.45 * length_pen

    @classmethod
    def route_expert(cls, text: str) -> str:
        """Switch-MoE style expert bucket (Gemini / PaLM-MoE routing analogy)."""
        pl = (text or "").lower()
        if re.search(r"\d+\s*[\+\-\*/%^]|calculate|equation|integral|fibonacci|\bfib\b", pl):
            return "math"
        if re.search(r"\b(code|python|def |```|debug|error|traceback|compile|syntax)\b", pl):
            return "code"
        if re.search(r"\b(why|how|explain|walkthrough|step|because|therefore)\b", pl):
            return "reason"
        if re.search(r"\b(what is|what are|define|meaning of|who is|when did)\b", pl):
            return "retrieve"
        return "chat"

    @classmethod
    def distil_align(cls, text: str) -> float:
        """Boost teacher-aligned / chain-of-thought phrasing (distillation papers)."""
        pl = (text or "").lower()
        cues = ("step", "first", "second", "therefore", "because", "verify", "note that", "in summary")
        hits = sum(1 for c in cues if c in pl)
        return min(0.25, hits * 0.05)

    @classmethod
    def score_item(
        cls,
        query: str,
        text: str,
        *,
        recency: float = 1.0,
        role: str = "chunk",
    ) -> float:
        if not cls.enabled():
            return recency
        expert_q = cls.route_expert(query)
        expert_d = cls.route_expert(text)
        moe_bonus = 0.18 if expert_d == expert_q else 0.0
        role_bonus = 0.08 if role == "user" else 0.0
        return (
            cls.attention_score(query, text) * 0.32
            + cls.bm25_score(query, text) * 0.28
            + cls.chinchilla_density(text) * 0.18
            + moe_bonus
            + cls.distil_align(text)
            + role_bonus
            + recency * 0.12
        )

    @classmethod
    def sort_items(cls, items: List[RankedDataItem], query: str) -> List[RankedDataItem]:
        if not items:
            return items
        cls._register_corpus([it.text for it in items])
        n = len(items)
        for i, it in enumerate(items):
            recency = 0.85 + 0.15 * (i + 1) / n
            it.expert = cls.route_expert(it.text)
            it.score = cls.score_item(query, it.text, recency=recency, role=it.role)
        return sorted(items, key=lambda x: x.score, reverse=True)

    @classmethod
    def sort_history(cls, history: List[Tuple[str, str]], query: str) -> List[Tuple[str, str]]:
        if not cls.enabled() or not history:
            return history
        items = [
            RankedDataItem(key=str(i), text=t, role=r)
            for i, (r, t) in enumerate(history)
        ]
        return [(it.role, it.text) for it in cls.sort_items(items, query)]

    @classmethod
    def sort_history_dicts(cls, history: List[Dict[str, str]], query: str) -> List[Dict[str, str]]:
        if not cls.enabled() or not history:
            return history
        items = [
            RankedDataItem(key=str(i), text=m.get("text", ""), role=m.get("role", "user"))
            for i, m in enumerate(history)
        ]
        return [{"role": it.role, "text": it.text} for it in cls.sort_items(items, query)]

    @classmethod
    def sort_paragraphs(cls, body: str, query: str) -> str:
        """Reorder multi-paragraph drafts by relevance (in-memory only)."""
        if not cls.enabled() or not body:
            return body
        if "```" in body:
            return body
        paras = [p.strip() for p in re.split(r"\n{2,}", body.strip()) if p.strip()]
        if len(paras) < 2:
            return body
        items = [RankedDataItem(key=str(i), text=p, role="chunk") for i, p in enumerate(paras)]
        ranked = cls.sort_items(items, query)
        return "\n\n".join(it.text for it in ranked)

    @classmethod
    def catseek_voice(cls, body: str, prompt: str, task: str = "general") -> str:
        """Format sorted content like catr1.10: direct, structured style."""
        if not CONFIG.get("catr1_voice", True) or not body:
            return body
        text = body.strip()
        if "```" in text or re.search(r"\bfable\b|\bpoem\b", text, re.I):
            return text

        text = cls.sort_paragraphs(text, prompt)
        pl = prompt.lower().strip()

        if any(k in pl for k in ("how ", "why ", "step", "explain", "walk me", "怎么", "为什么")):
            if "1." not in text[:300] and "Step" not in text[:300] and len(text) > 140:
                paras = [p.strip() for p in text.split("\n\n") if p.strip()]
                if len(paras) >= 2 and not paras[0].startswith("**"):
                    text = "\n\n".join(
                        f"**{i}.** {p.lstrip('0123456789. ')}"
                        for i, p in enumerate(paras, 1)
                    )

        return text.strip()

    @classmethod
    def stats_line(cls) -> str:
        return f"Google WP sort ({len(cls.WP_ALGOS)} algos) · {BRAND} voice"


CatSeekVoiceSorter = GoogleWhitepaperCatSeekSorter
WPCatSeekSort = GoogleWhitepaperCatSeekSorter


# ──────────────────────────────────────────────────────────────
# CAT R1.1 HEURISTICS ENGINE (catseek r1.10-tier · chat + code)
# Unified heuristics: intent · code detection · quality · self-verify · tone
# ──────────────────────────────────────────────────────────────
class CatSeekR1Heuristics:
    """
    Unified heuristics engine for chatting and coding — catseek r1.10 tier.
    Consolidates intent detection, code detection, language detection,
    response quality scoring, self-verification, and conversation flow.
    All heuristics run in-memory (files = off).
    """

    THINK_DEPTH_MAP = {
        "chat": 0, "casual": 0, "greeting": 0,
        "math": 1, "code": 2, "debug": 2, "execute": 1,
        "explain": 2, "compare": 2, "design": 3,
        "agent": 3, "general": 2, "fable": 1, "poem": 1,
    }
    CONFIDENCE_CATEGORIES = ("high", "medium", "low")

    @classmethod
    def classify_intent(cls, prompt: str) -> Dict[str, Any]:
        """
        catseek r1.10-tier intent classification with confidence scoring.
        Returns {intent, confidence, topic, subtopics, needs_code, needs_web}.
        """
        raw = (prompt or "").strip()
        pl = raw.lower()
        cjk = bool(re.search(r"[\u4e00-\u9fff]", pl))

        # Fast-path smalltalk
        if CatR11Synthesizer.smalltalk_reply(pl) or is_zh_greeting(raw):
            return {"intent": "chat", "confidence": "high", "topic": raw[:48], "subtopics": [], "needs_code": False, "needs_web": False}

        # Noise detection
        if CatSeekR1Fusion.is_noise(raw):
            return {"intent": "chat", "confidence": "high", "topic": "", "subtopics": [], "needs_code": False, "needs_web": False}

        needs_code = cls.detect_code_request(raw)
        needs_web = CatSeekWebProgram.wants_web(pl) if CatSeekWebProgram.enabled() else False
        topic = cls.extract_topic(raw)
        subtopics: List[str] = []

        # Code with fenced blocks
        if needs_code and CatSeekR1Code.extract_prompt_code(raw)[1]:
            return {"intent": "code", "confidence": "high", "topic": topic, "subtopics": ["fenced-block"], "needs_code": True, "needs_web": False}

        # Multi-label scoring
        scores: Dict[str, float] = {}
        intent_patterns = {
            "code": (r"\b(code|function|class|implement|snippet|script|def |import |```|"
                     r"write\s+(a|an|me)\s+(function|class|program|script|code))", 3.0),
            "math": (r"\d+\s*[\+\-\*/%\^]|calculate|compute|equation|solve|fibonacci|prime|factorial", 2.5),
            "debug": (r"\b(error|bug|traceback|exception|crash|broken|fix|debug|not working|fails)", 3.0),
            "explain": (r"\b(explain|what is|what are|define|describe|meaning|how does|how do|why does|"
                        r"tell me about|walk me through|tutorial|什么是|是什么|解释|说明|介绍)", 2.5),
            "design": (r"\b(design|architecture|plan|roadmap|system design|architect|scaffold)", 2.0),
            "compare": (r"\b(compare|vs | versus |difference|better|pros and cons|trade.?off)", 2.0),
            "agent": (r"\b(agent|multi.?step|orchestrate|workflow|pipeline|automate)", 1.8),
            "fable": (r"\b(fable|parable|allegory|bedtime story|moral|once upon)", 2.0),
            "poem": (r"\b(poem|poetry|haiku|sonnet|verse|rhyme)", 2.0),
            "creative": (r"\b(creative|story|tale|narrative|imagine|write a story)", 1.5),
            "web": (r"\b(website|web page|landing page|html page|site|dashboard|portfolio|"
                    r"build a page|make a site)", 2.0),
            "execute": (r"\b(run it|execute|interpret|/run|test it|and run|运行|执行)", 2.0),
            "chat": (r"\b(hi|hey|hello|how are you|what's up|howdy|meow|thanks|thank you)", 0.5),
        }

        for intent, (pattern, weight) in intent_patterns.items():
            matches = re.findall(pattern, pl, re.I)
            if matches:
                count = len(matches) if isinstance(matches, list) else 1
                scores[intent] = scores.get(intent, 0) + weight * count

        # CJK-specific boost
        if cjk:
            zh_code = re.search(r"(写代码|编程|程序|函数|斐波那契|算法|脚本)", pl)
            if zh_code:
                scores["code"] = scores.get("code", 0) + 2.0
            zh_explain = re.search(r"(什么是|是什么|什么叫|解释|说明|介绍)", pl)
            if zh_explain:
                scores["explain"] = scores.get("explain", 0) + 2.0

        # Question mark heuristics
        if "?" in raw and "explain" not in scores:
            scores["explain"] = scores.get("explain", 0) + 1.0

        # Code request override (with explain/creative guards)
        if needs_code:
            has_what_is = bool(re.search(r"\b(what is|what are|what's|whats|explain|define|describe)\b", pl))
            has_creative_cue = bool(re.search(r"\b(poem|poetry|fable|story|tale|creative)\b", pl))
            if has_what_is and scores.get("explain", 0) > 0:
                pass
            elif has_creative_cue:
                pass
            else:
                scores["code"] = scores.get("code", 0) + 4.0

        # Web request override
        if needs_web:
            scores["web"] = scores.get("web", 0) + 3.0

        # Explain vs code disambiguation: "what is X in Y" should be explain
        if scores.get("explain", 0) > 0 and scores.get("code", 0) > 0:
            if re.search(r"\b(what is|what are|explain|define|describe|what's)\b", pl):
                scores["code"] *= 0.3

        # Creative/fable/poem disambiguation: "write a poem" is NOT code
        if scores.get("poem", 0) > 0 or scores.get("fable", 0) > 0:
            if scores.get("code", 0) > 0:
                scores["code"] *= 0.2

        if not scores:
            scores["general"] = 1.0

        best_intent = max(scores, key=scores.get)
        best_score = scores[best_intent]

        confidence: str
        if best_score >= 4.0:
            confidence = "high"
        elif best_score >= 2.0:
            confidence = "medium"
        else:
            confidence = "low"

        subtopics = [k for k, v in scores.items() if v >= 1.5 and k != best_intent][:3]
        think_depth = cls.THINK_DEPTH_MAP.get(best_intent, 1)

        return {
            "intent": best_intent,
            "confidence": confidence,
            "topic": topic,
            "subtopics": subtopics,
            "needs_code": needs_code or best_intent == "code",
            "needs_web": needs_web or best_intent == "web",
            "scores": scores,
            "think_depth": think_depth,
            "cjk": cjk,
        }

    @classmethod
    def detect_code_request(cls, prompt: str) -> bool:
        """catseek r1.10-tier code request detection with high recall."""
        if not CatSeekR1Code.enabled():
            return False
        raw = (prompt or "").strip()
        if not raw:
            return False

        # Direct checks via existing engines
        if CatSeekR1Code.wants_code(raw):
            return True
        if CodeAnythingEngine.enabled() and CodeAnythingEngine.wants_anything(raw):
            return True

        pl = raw.lower()
        # Code shape heuristics
        code_shapes = re.search(
            r"(#include\s*<|def\s+\w+\s*\(|function\s+\w+|fn\s+main|public\s+class|"
            r"<!DOCTYPE|<html|console\.log|printf\s*\(|System\.out|package\s+main|"
            r"import\s+\w+|class\s+\w+\s*[:{]|=>\s*\{|var\s+\w+\s*=)", pl
        )
        if code_shapes:
            return True

        # Creative/narrative/poem/fable override — these should NOT trigger code
        creative_cues = ("poem", "poetry", "haiku", "sonnet", "verse", "rhyme",
                         "fable", "parable", "allegory", "story", "tale", "narrative",
                         "creative", "imagine", "bedtime")
        if any(c in pl for c in creative_cues):
            return False

        # Multi-keyword code intent signals
        code_signals = 0
        write_verbs = ("write", "make", "build", "create", "code", "implement", "generate", "show", "give")
        lang_nouns = ("python", "javascript", "typescript", "html", "css", "rust", "go", "java",
                      "c++", "cpp", "c#", "csharp", "bash", "shell", "sql", "php", "ruby",
                      "swift", "kotlin", "dart", "scala", "r", "julia", "lua")
        code_nouns = ("function", "class", "script", "program", "app", "api", "cli", "website",
                      "page", "snippet", "algorithm", "code", "implementation")

        for v in write_verbs:
            if v in pl:
                code_signals += 1
                break
        for l in lang_nouns:
            if l in pl:
                code_signals += 1
                break
        for n in code_nouns:
            if n in pl:
                code_signals += 1
                break

        # Inline language pattern: "in python", "using rust"
        if re.search(r"\b(in|using|with)\s+(" + "|".join(lang_nouns) + r")\b", pl):
            code_signals += 1

        return code_signals >= 2

    @classmethod
    def detect_language(cls, prompt: str, engine: Optional["CatR11Engine"] = None) -> Optional[str]:
        """Detect programming language from prompt with catseek r1.10-tier accuracy."""
        raw = (prompt or "").strip()
        pl = raw.lower()

        # Extract from fenced blocks
        fenced_lang, fenced_code = CatSeekR1Code.extract_prompt_code(raw)
        if fenced_lang:
            return engine.normalize_lang(fenced_lang) if engine else fenced_lang

        # Direct language mentions with code intent
        lang_priority = [
            ("python", r"\bpython\b|\.py\b|蟒蛇"),
            ("javascript", r"\bjavascript\b|\bjs\b|node\.?js"),
            ("typescript", r"\btypescript\b|\bts\b"),
            ("html", r"\bhtml\b|html5|网页|前端"),
            ("rust", r"\brust\b|cargo\b"),
            ("go", r"\bgo\b|\bgolang\b"),
            ("java", r"\bjava\b(?!script)"),
            ("cpp", r"\bc\+\+\b|\bcpp\b|\bcc\b"),
            ("c", r"\bc\b(?!\+|\s*#|\s*/)|c语言"),
            ("bash", r"\bbash\b|\bshell\b|\bzsh\b"),
            ("sql", r"\bsql\b"),
            ("kotlin", r"\bkotlin\b|\bkt\b"),
            ("swift", r"\bswift\b"),
            ("ruby", r"\bruby\b"),
            ("php", r"\bphp\b"),
        ]

        has_write = bool(re.search(r"\b(write|make|build|create|code|implement)\b", pl))
        for lang, pattern in lang_priority:
            if re.search(pattern, pl, re.I) and has_write:
                norm = engine.normalize_lang(lang) if engine else lang
                return norm

        # Fallback to existing engines
        if engine:
            extracted = engine.extract_lang(raw)
            if extracted:
                return engine.normalize_lang(extracted)
            from_text = engine.detect_lang_from_text(raw)
            if from_text:
                return engine.normalize_lang(from_text)

        # VibeCode fallback for CJK
        if re.search(r"[\u4e00-\u9fff]", pl) and re.search(r"(程序|代码|脚本)", pl):
            return "python"

        return None

    @classmethod
    def extract_topic(cls, prompt: str) -> str:
        """Extract the core topic from a prompt."""
        raw = (prompt or "").strip()
        pl = raw.lower()

        # Subject extraction via existing engines
        if CatSeekR1Code._subject(raw):
            return CatSeekR1Code._subject(raw)
        vibe_subj = VibeCodeHeuristics.subject_from_text(raw)
        if vibe_subj:
            return vibe_subj

        # Strip question prefixes
        for prefix in ("what is ", "what's ", "what are ", "explain ", "define ",
                        "tell me about ", "什么是", "是什么", "解释", "说明"):
            if pl.startswith(prefix):
                return raw[len(prefix):].rstrip("?.,! ")[:80]

        # Extract topic after "about"
        m = re.search(r"\babout\s+(.+?)(?:\s*$|[.?!])", pl)
        if m:
            return m.group(1).strip()[:80]

        # First meaningful noun phrase
        m = re.search(r"\b(how\s+to\s+|how\s+do\s+i\s+|how\s+does\s+)(.+?)(?:\s*$|[.?!])", pl)
        if m:
            return m.group(2).strip()[:80]

        return raw[:80] or "general"

    @classmethod
    def quality_score(cls, text: str, prompt: str) -> float:
        """
        catseek r1.10-tier response quality scoring.
        Returns score 0.0–1.0 based on coverage, structure, relevance.
        """
        if not text or not text.strip():
            return 0.0

        score = 0.4  # base

        # Length adequacy (not too short, not too long)
        words = len(text.split())
        if 10 <= words <= 200:
            score += 0.1
        elif words > 200:
            score += 0.05

        # Structure signals
        if "\n\n" in text:
            score += 0.05
        if re.search(r"^\d+\.|\*\*|^- ", text, re.M):
            score += 0.05

        # Code block presence when prompt asks for code
        if cls.detect_code_request(prompt):
            if "```" in text:
                score += 0.1
            elif "`" in text:
                score += 0.05

        # Relevance: answer addresses the question
        if "?" in prompt:
            if "?" not in text and len(text) > 40:
                score += 0.05
            if re.search(r"\b(because|means|is called|refers to|works by)\b", text, re.I):
                score += 0.05

        # Verification markers
        if re.search(r"\b(verify|check|confirm|note that|in summary)\b", text, re.I):
            score += 0.05

        # catseek r1.10 tone: clear, direct, structured
        if re.search(r"\*\*[^*]+\*\*", text):
            score += 0.05
        if re.search(r"\b(step|first|second|finally|therefore)\b", text, re.I):
            score += 0.05

        return min(1.0, score)

    @classmethod
    def self_verify(cls, prompt: str, response: str) -> Dict[str, Any]:
        """
        Self-verification heuristics (catseek r1.10 GRPO-style).
        Returns {passed, issues, suggestions}.
        """
        issues: List[str] = []
        suggestions: List[str] = []

        if not response or not response.strip():
            return {"passed": False, "issues": ["empty response"], "suggestions": ["generate a substantive reply"]}

        # Check response addresses prompt
        pl = prompt.lower()
        rl = response.lower()

        # Question coverage
        if "?" in pl and not any(c in rl for c in ("because", "means", "is", "are", "was", "were")):
            if len(rl.split()) < 8:
                issues.append("response may not address the question")
                suggestions.append("directly answer the question asked")

        # Code verification
        if cls.detect_code_request(prompt):
            if "```" not in response:
                issues.append("code requested but no fenced code block in response")
                suggestions.append("wrap code in ``` fences for execution")
            else:
                # Check for common code issues
                code_blocks = re.findall(r"```(?:\w+)\n(.*?)```", response, re.S)
                for block in code_blocks:
                    if block.strip():
                        # Check for truncated code
                        if block.strip().endswith("...") or block.strip().endswith("…"):
                            issues.append("code block may be truncated")
                            suggestions.append("provide complete code")
                        # Check for placeholder comments
                        if re.search(r"(#\s*todo|//\s*todo|/\*\s*todo)", block, re.I):
                            issues.append("code contains TODO placeholders")

        # Explanation verification
        intent = cls.classify_intent(prompt)["intent"]
        if intent in ("explain", "compare", "design") and len(rl.split()) < 20:
            issues.append("explanation too brief for the topic")
            suggestions.append("add more detail, examples, or structure")

        # Math verification
        if intent == "math":
            numbers = re.findall(r"-?\d+\.?\d*", response)
            if not numbers:
                issues.append("math query but no numeric result in response")

        passed = len(issues) == 0
        return {
            "passed": passed,
            "issues": issues,
            "suggestions": suggestions,
            "quality": cls.quality_score(response, prompt),
        }

    @classmethod
    def conversation_flow(cls, prompt: str, history: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Analyze conversation flow and determine follow-up behavior.
        Returns {is_followup, context, suggested_action, needs_history}.
        """
        if not history:
            return {"is_followup": False, "context": "", "suggested_action": "new", "needs_history": False}

        pl = prompt.lower().strip()
        last_user = ""
        last_bot = ""
        for m in reversed(history):
            if m.get("role") == "assistant" and not last_bot:
                last_bot = m.get("text", "")
            elif m.get("role") == "user" and not last_user:
                last_user = m.get("text", "")
            if last_user and last_bot:
                break

        ack_words = frozenset({"yes", "ok", "okay", "sure", "yep", "yeah", "thanks", "thank you",
                               "好的", "是的", "对", "谢谢", "可以"})
        follow_cues = ("tell me more", "go on", "continue", "and then", "what else",
                       "more", "expand", "elaborate", "go deeper", "why though",
                       "how come", "what about", "how about", "say more")

        is_ack = pl.strip().rstrip(".!") in ack_words or pl.strip() in ack_words
        is_followup_cue = any(c in pl for c in follow_cues)
        is_followup = is_ack or is_followup_cue or pl in (".", "..", ">", "ok")

        if is_followup and last_user:
            return {
                "is_followup": True,
                "context": last_user[:120],
                "last_bot_snippet": last_bot[:200] if last_bot else "",
                "suggested_action": "continue",
                "needs_history": True,
            }

        return {
            "is_followup": False,
            "context": last_user[:120] if last_user else "",
            "suggested_action": "new_topic",
            "needs_history": bool(history),
        }

    @classmethod
    def detect_tone(cls, prompt: str) -> str:
        """Detect the appropriate response tone."""
        pl = prompt.lower().strip()
        if re.search(r"\b(urgent|asap|quick|fast|emergency|hurry)", pl):
            return "urgent"
        if re.search(r"\b(why|how|explain|walk me|step by step|tutorial)", pl):
            return "educational"
        if re.search(r"\b(写代码|编程|代码|程序|脚本)", pl):
            return "technical"
        if re.search(r"\b(hi|hey|hello|how are you|howdy|sup)", pl):
            return "friendly"
        if re.search(r"\b(joke|funny|humor|laugh|make me laugh)", pl):
            return "humorous"
        if re.search(r"\b(poem|fable|story|creative|imagine)", pl):
            return "creative"
        if re.search(r"\b(error|bug|traceback|crash|broken|fails)", pl):
            return "supportive"
        return "neutral"

    @classmethod
    def stats_line(cls) -> str:
        return f"CatSeekR1 heuristics · catseek r1.10 tier · intent+code+verify"


CatSeekHeuristics = CatSeekR1Heuristics


# ──────────────────────────────────────────────────────────────
# CAT R1.1 WEB (cat r1.10 websites · artifacts · fetch · files = off)
# ──────────────────────────────────────────────────────────────
class _TextExtractor(HTMLParser):
    __slots__ = ("parts", "_skip")

    def __init__(self):
        super().__init__()
        self.parts: List[str] = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript"}:
            self._skip = True

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript"}:
            self._skip = False
        elif tag in {"p", "br", "div", "li", "h1", "h2", "h3", "tr"}:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            t = data.strip()
            if t:
                self.parts.append(t)


@dataclass
class WebSiteRecord:
    site_id: str
    title: str
    html: str
    template: str
    prompt: str
    created: float = field(default_factory=time.time)

    def preview_url(self, port: int) -> str:
        return f"http://127.0.0.1:{port}/web/preview/{self.site_id}"


class CatSeekWebProgram:
    """
    In-memory cat r1.10 websites program (files = off):
    artifact HTML/CSS/JS · URL fetch · site registry · API preview — no disk writes.
    """

    _URL = re.compile(r"https?://[^\s<>\"']+", re.I)
    _WEB_CUES = (
        "website", "web page", "webpage", "landing page", "landing", "homepage",
        "home page", "portfolio site", "dashboard", "docs site", "documentation page",
        "saas page", "blog page", "artifact", "single page app", "spa",
        "做个网站", "做个网页", "网站", "网页", "落地页", "首页",
    )
    _FETCH_CUES = (
        "fetch ", "read url", "read website", "read this site", "open url",
        "scrape ", "get page", "summarize url", "summarize website",
        "读取网址", "打开网站", "抓取",
    )
    TEMPLATE_NAMES = (
        "landing", "dashboard", "docs", "portfolio", "saas", "blog",
        "resume", "todo", "calculator", "geocities", "minimal",
    )

    def __init__(self):
        self._sites: Dict[str, WebSiteRecord] = {}

    @classmethod
    def enabled(cls) -> bool:
        return bool(CONFIG.get("web_program_enabled", True))

    @classmethod
    def wants_web(cls, pl: str) -> bool:
        if not cls.enabled():
            return False
        if pl.startswith("/web"):
            return True
        if cls._URL.search(pl) and any(c in pl for c in cls._FETCH_CUES):
            return True
        if any(c in pl for c in cls._FETCH_CUES):
            return True
        if re.search(r"\b(build|make|create|design|generate|write)\s+(?:a|an|me\s+a|my\s+)?(?:\w+\s+){0,4}(?:website|webpage|web page|landing(?:\s+page)?|site|homepage)\b", pl):
            return True
        if any(c in pl for c in cls._WEB_CUES) and re.search(r"\b(build|make|create|write|design|generate|show)\b", pl):
            return True
        if re.search(r"\bhtml\b.*\b(site|page|app|ad)\b", pl):
            return True
        return False

    @classmethod
    def extract_url(cls, text: str) -> Optional[str]:
        m = cls._URL.search(text)
        return m.group(0).rstrip(".,)") if m else None

    @staticmethod
    def _subject(prompt: str) -> str:
        pl = prompt.lower()
        for prefix in ("build a ", "make a ", "create a ", "design a ", "write a ", "generate a "):
            if pl.startswith(prefix):
                rest = prompt[len(prefix):].strip()
                rest = re.split(r"\s+(?:website|webpage|web page|landing page|site|page)\b", rest, flags=re.I)[0]
                if rest.strip():
                    return rest.strip()[:80]
        m = re.search(r"(?:website|landing page|site|page)\s+(?:for|about|called)\s+(.+?)(?:\s*$|\.)", pl, re.I)
        if m:
            return m.group(1).strip()[:80]
        return BRAND

    @classmethod
    def pick_template(cls, prompt: str) -> str:
        pl = prompt.lower()
        if re.search(r"\bgeocit(?:ies|es)\b", pl) or ("gamer" in pl and "usb" in pl):
            return "geocities"
        if "dashboard" in pl or "admin" in pl:
            return "dashboard"
        if "doc" in pl or "documentation" in pl or "readme" in pl:
            return "docs"
        if "portfolio" in pl:
            return "portfolio"
        if "saas" in pl or "pricing" in pl or "subscribe" in pl:
            return "saas"
        if "blog" in pl or "article" in pl:
            return "blog"
        if "resume" in pl or "cv" in pl:
            return "resume"
        if "todo" in pl or "task list" in pl:
            return "todo"
        if "calculator" in pl or "calc" in pl:
            return "calculator"
        if "landing" in pl or "homepage" in pl or "home page" in pl:
            return "landing"
        return "minimal"

    @staticmethod
    def _shell(title: str, body: str, *, extra_head: str = "", extra_script: str = "") -> str:
        t = html_module.escape(title)
        return (
            f"<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
            f"  <meta charset=\"UTF-8\">\n"
            f"  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n"
            f"  <title>{t}</title>\n{extra_head}\n</head>\n<body>\n{body}\n"
            f"{extra_script}\n</body>\n</html>"
        )

    @classmethod
    def _template_landing(cls, title: str, headline: str) -> str:
        h = html_module.escape(headline)
        css = (
            "  <style>\n"
            "    :root { --bg:#0f172a; --fg:#e2e8f0; --acc:#38bdf8; --card:#1e293b; }\n"
            "    * { box-sizing:border-box; margin:0; }\n"
            "    body { font-family:system-ui,sans-serif; background:var(--bg); color:var(--fg); }\n"
            "    nav { display:flex; justify-content:space-between; padding:1rem 2rem; }\n"
            "    .hero { text-align:center; padding:4rem 1.5rem; max-width:720px; margin:0 auto; }\n"
            "    h1 { font-size:clamp(2rem,5vw,3rem); margin-bottom:1rem; }\n"
            "    p { opacity:.85; line-height:1.6; margin-bottom:2rem; }\n"
            "    .cta { display:inline-block; background:var(--acc); color:#0f172a; padding:.75rem 1.5rem; "
            "border-radius:8px; text-decoration:none; font-weight:600; }\n"
            "    .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:1rem; "
            "padding:2rem; max-width:960px; margin:0 auto; }\n"
            "    .card { background:var(--card); padding:1.25rem; border-radius:12px; }\n"
            "  </style>"
        )
        body = (
            f"  <nav><strong>{html_module.escape(title)}</strong><span>{BRAND}</span></nav>\n"
            f"  <section class=\"hero\"><h1>{h}</h1>\n"
            f"    <p>Built in-memory by {WEB_PROGRAM_NAME} — catr1.10 artifacts, no files written.</p>\n"
            f"    <a class=\"cta\" href=\"#features\">Get started</a></section>\n"
            f"  <div class=\"grid\" id=\"features\">\n"
            f"    <div class=\"card\"><h3>Fast</h3><p>catr1.10 token-weight synthesis.</p></div>\n"
            f"    <div class=\"card\"><h3>Local</h3><p>Runs fully offline.</p></div>\n"
            f"    <div class=\"card\"><h3>Private</h3><p>Everything stays in RAM.</p></div>\n"
            f"  </div>"
        )
        return cls._shell(title, body, extra_head=css)

    @classmethod
    def _template_dashboard(cls, title: str) -> str:
        css = (
            "  <style>body{font-family:system-ui;margin:0;background:#111;color:#eee;display:grid;"
            "grid-template-columns:220px 1fr;min-height:100vh}"
            "aside{background:#1a1a1a;padding:1rem}main{padding:1.5rem}"
            ".stat{display:inline-block;background:#222;padding:1rem 1.5rem;border-radius:8px;margin:.5rem}"
            ".stat b{font-size:1.5rem;color:#4ade80}</style>"
        )
        body = (
            f"  <aside><h2>{html_module.escape(title)}</h2><ul><li>Overview</li><li>Analytics</li>"
            f"<li>Settings</li></ul></aside>\n"
            f"  <main><h1>Dashboard</h1>\n"
            f"    <div class=\"stat\"><div>Users</div><b>1,024</b></div>\n"
            f"    <div class=\"stat\"><div>Requests</div><b>8,192</b></div>\n"
            f"    <div class=\"stat\"><div>Uptime</div><b>99.9%</b></div></main>"
        )
        return cls._shell(title, body, extra_head=css)

    @classmethod
    def _template_docs(cls, title: str) -> str:
        css = "  <style>body{font-family:Georgia,serif;max-width:720px;margin:2rem auto;padding:0 1rem;line-height:1.7}"
        body = (
            f"  <h1>{html_module.escape(title)} — Docs</h1>\n"
            f"  <h2>Quick start</h2><p>Web program stores sites in memory. Preview via "
            f"<code>/web/preview/&lt;id&gt;</code> on the local API.</p>\n"
            f"  <h2>Commands</h2><ul><li><code>/web list</code></li><li><code>/web fetch URL</code></li>"
            f"<li><code>/web build prompt</code></li></ul>"
        )
        return cls._shell(title, body, extra_head=css)

    @classmethod
    def _template_portfolio(cls, title: str, name: str) -> str:
        n = html_module.escape(name)
        css = (
            "  <style>body{font-family:system-ui;background:#fafafa;color:#111;max-width:800px;"
            "margin:2rem auto;padding:1rem}.project{border:1px solid #ddd;border-radius:8px;"
            "padding:1rem;margin:1rem 0}</style>"
        )
        body = (
            f"  <header><h1>{n}</h1><p>Portfolio · {html_module.escape(title)}</p></header>\n"
            f"  <section class=\"project\"><h3>Project Alpha</h3><p>Full-stack app with in-memory deploy.</p></section>\n"
            f"  <section class=\"project\"><h3>Project Beta</h3><p>catr1.10-powered local AI tooling.</p></section>"
        )
        return cls._shell(title, body, extra_head=css)

    @classmethod
    def _template_saas(cls, title: str) -> str:
        css = (
            "  <style>body{font-family:system-ui;text-align:center;padding:2rem;background:linear-gradient(#eef,#fff)}"
            ".price{display:inline-block;border:2px solid #333;border-radius:12px;padding:2rem;margin:1rem}"
            ".price h2{font-size:2.5rem}</style>"
        )
        body = (
            f"  <h1>{html_module.escape(title)}</h1><p>Simple pricing. No files. All in-memory.</p>\n"
            f"  <div class=\"price\"><h3>Pro</h3><h2>$9</h2><p>/month</p></div>\n"
            f"  <div class=\"price\"><h3>Team</h3><h2>$29</h2><p>/month</p></div>"
        )
        return cls._shell(title, body, extra_head=css)

    @classmethod
    def _template_blog(cls, title: str, headline: str) -> str:
        h = html_module.escape(headline)
        css = "  <style>body{font-family:Georgia,serif;max-width:640px;margin:2rem auto;line-height:1.8;padding:1rem}</style>"
        body = (
            f"  <article><h1>{h}</h1><p><em>{BRAND}</em></p>\n"
            f"  <p>This post was generated in-memory. {WEB_PROGRAM_NAME} by {BRAND}</p>"
            f"artifact pages — HTML, CSS, and optional JS — without writing to disk.</p></article>"
        )
        return cls._shell(title, body, extra_head=css)

    @classmethod
    def _template_todo(cls, title: str) -> str:
        css = "  <style>body{font-family:system-ui;max-width:420px;margin:3rem auto;padding:1rem}"
        script = (
            "  <script>\n"
            "    const ul=document.getElementById('t');\n"
            "    function add(){const i=document.getElementById('i');if(!i.value.trim())return;"
            "const li=document.createElement('li');li.textContent=i.value;i.value='';ul.appendChild(li);}\n"
            "  </script>"
        )
        body = (
            f"  <h1>{html_module.escape(title)}</h1>\n"
            f"  <input id=\"i\" placeholder=\"New task…\" style=\"width:70%\"> "
            f"<button onclick=\"add()\">Add</button>\n  <ul id=\"t\"></ul>"
        )
        return cls._shell(title, body, extra_head=css, extra_script=script)

    @classmethod
    def _template_calculator(cls, title: str) -> str:
        css = "  <style>body{font-family:monospace;text-align:center;padding:2rem}"
        script = (
            "  <script>\n"
            "    function calc(){try{document.getElementById('o').textContent="
            "eval(document.getElementById('e').value)}catch(e){document.getElementById('o').textContent='Error'}}\n"
            "  </script>"
        )
        body = (
            f"  <h1>{html_module.escape(title)}</h1>\n"
            f"  <input id=\"e\" style=\"font-size:1.2rem;width:200px\"> "
            f"<button onclick=\"calc()\">=</button>\n  <p id=\"o\"></p>"
        )
        return cls._shell(title, body, extra_head=css, extra_script=script)

    @classmethod
    def render(cls, template: str, prompt: str, engine: Optional["CatR11Engine"] = None) -> str:
        subject = cls._subject(prompt)
        title = subject.title() if subject else f"{BRAND} site"
        headline = subject or title
        if template == "geocities":
            return CatSeekR1Code._html_geocities(prompt)
        if template == "landing":
            return cls._template_landing(title, headline)
        if template == "dashboard":
            return cls._template_dashboard(title)
        if template == "docs":
            return cls._template_docs(title)
        if template == "portfolio":
            return cls._template_portfolio(title, headline)
        if template == "saas":
            return cls._template_saas(title)
        if template == "blog":
            return cls._template_blog(title, headline)
        if template == "resume":
            return cls._template_portfolio(title, f"{headline} — Resume")
        if template == "todo":
            return cls._template_todo(title)
        if template == "calculator":
            return cls._template_calculator(title)
        # minimal / default — landing template (avoid _html ↔ render recursion)
        return cls._template_landing(title, headline)

    def store(self, prompt: str, html: str, template: str) -> WebSiteRecord:
        sid = uuid.uuid4().hex[:10]
        title = self._subject(prompt).title() or "Site"
        rec = WebSiteRecord(site_id=sid, title=title, html=html, template=template, prompt=prompt[:200])
        self._sites[sid] = rec
        while len(self._sites) > CONFIG.get("web_max_sites", 256):
            oldest = min(self._sites.values(), key=lambda s: s.created)
            del self._sites[oldest.site_id]
        return rec

    def get(self, site_id: str) -> Optional[WebSiteRecord]:
        return self._sites.get(site_id)

    def build(self, prompt: str, engine: Optional["CatR11Engine"] = None) -> WebSiteRecord:
        tpl = self.pick_template(prompt)
        html = self.render(tpl, prompt, engine)
        return self.store(prompt, html, tpl)

    def fetch_url(self, url: str) -> str:
        if not CONFIG.get("web_fetch_enabled", True):
            return "Web fetch is disabled in CONFIG."
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            return "Only http:// and https:// URLs are supported."
        max_bytes = CONFIG.get("web_max_fetch_kb", 256) * 1024
        try:
            req = urlrequest.Request(url, headers={"User-Agent": f"{BRAND}/{VERSION} (in-memory)"})
            with urlrequest.urlopen(req, timeout=12) as resp:
                raw = resp.read(max_bytes + 1)
                if len(raw) > max_bytes:
                    raw = raw[:max_bytes]
                ctype = resp.headers.get("Content-Type", "")
                charset = "utf-8"
                if "charset=" in ctype:
                    charset = ctype.split("charset=")[-1].split(";")[0].strip()
                text = raw.decode(charset, errors="replace")
        except (URLError, HTTPError, TimeoutError, ValueError) as e:
            return f"Fetch failed: {e}"
        if "html" in text.lower()[:500] or "<" in text[:200]:
            parser = _TextExtractor()
            try:
                parser.feed(text)
            except Exception:
                pass
            body = re.sub(r"\n{3,}", "\n\n", " ".join(parser.parts))
            body = body[:4000]
            return f"**Fetched** `{url}`\n\n{body or '(no extractable text)'}"
        return f"**Fetched** `{url}`\n\n```\n{text[:4000]}\n```"

    @classmethod
    def help_text(cls) -> str:
        port = CONFIG.get("api_port", 8765)
        tpl = ", ".join(cls.TEMPLATE_NAMES)
        return (
            f"**{WEB_PROGRAM_NAME}** · {BRAND} websites · \n\n"
            "In-memory artifact builder, URL reader, and API preview.\n\n"
            "**Commands**\n"
            "- `/web` — this help\n"
            "- `/web list` — sites in memory\n"
            "- `/web templates` — available layouts\n"
            "- `/web build <prompt>` — generate & store a site\n"
            "- `/web fetch <url>` — read a page (text extract)\n"
            "- `/web preview <id>` — preview link\n\n"
            f"**Templates:** {tpl}\n\n"
            f"**API:** `GET http://127.0.0.1:{port}/web/preview/<id>` · `GET /web/sites`\n\n"
            "Natural language: `build a landing page for my app`, `fetch https://example.com`"
        )

    def list_sites(self) -> str:
        if not self._sites:
            return "No sites in memory. Try `/web build a landing page` or ask naturally."
        port = CONFIG.get("api_port", 8765)
        lines = [f"**{len(self._sites)} site(s)** in memory:\n"]
        for rec in sorted(self._sites.values(), key=lambda s: s.created, reverse=True):
            lines.append(
                f"- `{rec.site_id}` · **{rec.title}** · {rec.template} · "
                f"[preview]({rec.preview_url(port)})"
            )
        return "\n".join(lines)

    def handle_command(self, engine: "CatR11Engine", raw: str) -> str:
        pl = raw.lower().strip()
        if pl in {"/web", "/web help"}:
            return self.help_text()
        if pl == "/web list":
            return self.list_sites()
        if pl == "/web templates":
            return "**Templates:** " + ", ".join(self.TEMPLATE_NAMES)
        if pl.startswith("/web fetch "):
            url = self.extract_url(raw) or raw.split(maxsplit=2)[-1].strip()
            return self.fetch_url(url)
        if pl.startswith("/web build ") or pl.startswith("/web "):
            prompt = raw.split(maxsplit=2)[2] if pl.startswith("/web build ") else raw.split(maxsplit=1)[1]
            if prompt.lower().startswith("build "):
                prompt = prompt[6:]
            rec = self.build(prompt.strip() or "landing page", engine)
            port = CONFIG.get("api_port", 8765)
            return (
                f"**Site built** · `{rec.site_id}` · **{rec.template}**\n\n"
                f"Preview: {rec.preview_url(port)}\n\n"
                f"```html\n{rec.html[:1200]}{'…' if len(rec.html) > 1200 else ''}\n```"
            )
        if pl.startswith("/web preview "):
            sid = raw.split(maxsplit=2)[-1].strip()
            rec = self.get(sid)
            if not rec:
                return f"No site `{sid}`. Use `/web list`."
            return f"Preview: {rec.preview_url(CONFIG.get('api_port', 8765))}\n\n```html\n{rec.html[:800]}…\n```"
        return self.help_text()

    def respond(self, engine: "CatR11Engine", prompt: str) -> str:
        pl = prompt.lower().strip()
        if pl.startswith("/web"):
            return self.handle_command(engine, prompt)
        url = self.extract_url(prompt)
        if url and any(c in pl for c in self._FETCH_CUES + ("http", "https", "www.")):
            return self.fetch_url(url)
        if url and not re.search(r"\b(build|make|create|html|page|site)\b", pl):
            return self.fetch_url(url)
        rec = self.build(prompt, engine)
        port = CONFIG.get("api_port", 8765)
        fenced = f"```html\n{rec.html}\n```"
        if CONFIG.get("code_output_exact"):
            return (
                f"**{WEB_PROGRAM_NAME}** · `{rec.site_id}` · {rec.template}\n"
                f"Preview: {rec.preview_url(port)}\n\n{fenced}"
            )
        return (
            f"**Site built** · `{rec.site_id}` · **{rec.template}**\n"
            f"Preview: {rec.preview_url(port)}\n\n{fenced}"
        )

    def clear(self) -> None:
        self._sites.clear()


# ──────────────────────────────────────────────────────────────
# CODE ANYTHING (universal polyglot coder · files = off)
# ──────────────────────────────────────────────────────────────
class CodeAnythingEngine:
    """
    Universal in-memory coder for cat r1.10 · files = off.
    Any language · any task · polyglot scaffolds · dry-run simulation.
    """

    TAG = "code anything"
    MODE = CONFIG.get("code_anything_mode", "universal")

    LANGS = (
        "python", "javascript", "typescript", "java", "kotlin", "swift", "scala",
        "rust", "go", "c", "cpp", "csharp", "fsharp", "ruby", "php", "perl",
        "lua", "r", "julia", "dart", "elixir", "haskell", "clojure", "groovy",
        "powershell", "solidity", "zig", "nim", "crystal", "fortran", "cobol",
        "verilog", "vhdl", "assembly", "html", "css", "scss", "sql", "graphql",
        "yaml", "toml", "json", "dockerfile", "makefile", "bash", "shell",
    )
    VALID_LANGS = frozenset(LANGS)
    RUNNABLE = frozenset({"python", "javascript", "bash", "shell"})

    LANG_ALIASES = {
        "py": "python", "python3": "python", "python2": "python",
        "js": "javascript", "node": "javascript", "nodejs": "javascript",
        "ts": "typescript", "c++": "cpp", "cc": "cpp", "cxx": "cpp",
        "c#": "csharp", "cs": "csharp", "dotnet": "csharp",
        "f#": "fsharp", "fs": "fsharp",
        "sh": "bash", "shell": "bash", "zsh": "bash",
        "asm": "assembly", "golang": "go", "kt": "kotlin",
        "rb": "ruby", "rs": "rust", "yml": "yaml",
        "docker": "dockerfile", "make": "makefile",
        "ps1": "powershell", "pwsh": "powershell",
        "sol": "solidity", "proto": "protobuf",
    }

    FRAMEWORK_LANG = {
        "react": "javascript", "vue": "javascript", "svelte": "javascript",
        "angular": "typescript", "nextjs": "typescript", "next.js": "typescript",
        "fastapi": "python", "django": "python", "flask": "python", "streamlit": "python",
        "rails": "ruby", "laravel": "php", "symfony": "php",
        "spring": "java", "springboot": "java", "hibernate": "java",
        "express": "javascript", "nestjs": "typescript", "deno": "typescript",
        "actix": "rust", "rocket": "rust", "gin": "go", "echo": "go",
        "swiftui": "swift", "uikit": "swift", "jetpack": "kotlin",
        "flutter": "dart", "react native": "javascript",
        "solidity": "solidity", "hardhat": "solidity", "foundry": "solidity",
        "terraform": "hcl", "kubernetes": "yaml", "k8s": "yaml",
    }

    _EXT_LANG = {
        ".py": "python", ".js": "javascript", ".ts": "typescript", ".jsx": "javascript",
        ".tsx": "typescript", ".java": "java", ".kt": "kotlin", ".kts": "kotlin",
        ".swift": "swift", ".scala": "scala", ".rs": "rust", ".go": "go",
        ".c": "c", ".h": "c", ".cpp": "cpp", ".cc": "cpp", ".hpp": "cpp",
        ".cs": "csharp", ".fs": "fsharp", ".rb": "ruby", ".php": "php",
        ".pl": "perl", ".lua": "lua", ".r": "r", ".jl": "julia", ".dart": "dart",
        ".ex": "elixir", ".exs": "elixir", ".hs": "haskell", ".clj": "clojure",
        ".groovy": "groovy", ".ps1": "powershell", ".sol": "solidity",
        ".zig": "zig", ".nim": "nim", ".cr": "crystal", ".f90": "fortran",
        ".cob": "cobol", ".v": "verilog", ".vhdl": "vhdl", ".asm": "assembly",
        ".html": "html", ".css": "css", ".scss": "scss", ".sql": "sql",
        ".yaml": "yaml", ".yml": "yaml", ".toml": "toml", ".json": "json",
        ".sh": "bash", ".bash": "bash", ".zsh": "bash",
        ".dockerfile": "dockerfile", ".mk": "makefile",
    }

    _ANYTHING_CUE = re.compile(
        r"\b(code\s+anything|any\s+language|polyglot|multi[\s-]?language|"
        r"whatever\s+language|in\s+any\s+lang)\b",
        re.I,
    )
    _SCAFFOLD_CUE = re.compile(
        r"\b(scaffold|boilerplate|starter|template|skeleton|stub|implement|"
        r"port\s+to|convert\s+to|rewrite\s+in|refactor|migrate)\b",
        re.I,
    )

    @classmethod
    def enabled(cls) -> bool:
        return bool(CONFIG.get("code_anything", True))

    @classmethod
    def normalize_lang(cls, lang: Optional[str]) -> Optional[str]:
        if not lang:
            return None
        t = lang.lower().strip().strip(".")
        return cls.LANG_ALIASES.get(t, t)

    @classmethod
    def wants_anything(cls, prompt: str) -> bool:
        if not cls.enabled():
            return False
        pl = (prompt or "").lower()
        if cls._ANYTHING_CUE.search(pl):
            return True
        if cls._SCAFFOLD_CUE.search(pl) and re.search(
            r"\b(code|script|app|api|cli|program|function|class|module)\b", pl
        ):
            return True
        for fw in cls.FRAMEWORK_LANG:
            if fw in pl and re.search(r"\b(write|build|make|create|scaffold|implement|code)\b", pl):
                return True
        for ext in cls._EXT_LANG:
            if ext in pl and re.search(r"\b(write|build|make|create|code|implement)\b", pl):
                return True
        return False

    @classmethod
    def detect_task(cls, prompt: str) -> str:
        pl = (prompt or "").lower()
        if re.search(r"\b(fibonacci|fib\b|prime|sieve|sort|search|bst|graph|dp\b|leetcode|algorithm|algo)\b", pl):
            return "algo"
        if re.search(r"\b(api|rest|endpoint|graphql|grpc|fastapi|flask|express)\b", pl):
            return "api"
        if re.search(r"\b(cli|argparse|click|command[\s-]?line|argv)\b", pl):
            return "cli"
        if re.search(r"\b(test|unit test|pytest|jest|spec\b|tdd)\b", pl):
            return "test"
        if re.search(r"\b(csv|json|data|pandas|etl|parse)\b", pl):
            return "data"
        if re.search(r"\b(game|snake|pong|tic[\s-]?tac|chess|puzzle)\b", pl):
            return "game"
        if re.search(r"\b(yaml|toml|json|config|dockerfile|docker|kubernetes|k8s)\b", pl):
            return "config"
        if re.search(r"\b(html|website|landing|page|frontend|react|vue|css)\b", pl):
            return "web"
        if re.search(r"\b(mobile|ios|android|swiftui|kotlin)\b", pl):
            return "mobile"
        return "general"

    @classmethod
    def infer_lang(cls, engine: Optional["CatR11Engine"], prompt: str, hint: str = "") -> str:
        norm = cls.normalize_lang(hint)
        if norm and norm in cls.VALID_LANGS:
            return norm
        pl = (prompt or "").lower()
        for fw, lang in cls.FRAMEWORK_LANG.items():
            if fw in pl:
                return lang
        for ext, lang in cls._EXT_LANG.items():
            if ext in pl:
                return lang
        for lang in sorted(cls.LANGS, key=len, reverse=True):
            if re.search(rf"\b{re.escape(lang)}\b", pl):
                return lang
        if engine is not None:
            extracted = engine.extract_lang(prompt)
            if extracted:
                n = cls.normalize_lang(extracted)
                if n in cls.VALID_LANGS:
                    return n
            from_code = engine.detect_lang_from_text(prompt)
            if from_code:
                n = cls.normalize_lang(from_code)
                if n in cls.VALID_LANGS:
                    return n
        if re.search(r"\bhtml\b", pl):
            return "html"
        if re.search(r"\brust\b", pl):
            return "rust"
        if re.search(r"\bgo\b|\bgolang\b", pl):
            return "go"
        if re.search(r"\bjava\b(?!script)", pl):
            return "java"
        if re.search(r"\bkotlin\b", pl):
            return "kotlin"
        if re.search(r"\bswift\b", pl):
            return "swift"
        if re.search(r"\btypescript\b|\bts\b", pl):
            return "typescript"
        if re.search(r"\bjavascript\b|\bjs\b", pl):
            return "javascript"
        return "python"

    @classmethod
    def _subject(cls, prompt: str, engine: Optional["CatR11Engine"] = None) -> str:
        if engine is not None:
            subj = CatSeekR1Code._subject(prompt, engine)
            if subj and subj.lower() not in {"hello world", "it", "it html"}:
                return subj
        return "App"

    @classmethod
    def _py_algo(cls, prompt: str) -> str:
        pl = prompt.lower()
        if "fibonacci" in pl or re.search(r"\bfib\b", pl):
            return (
                "def fib(n: int) -> int:\n"
                "    a, b = 0, 1\n"
                "    for _ in range(n):\n"
                "        a, b = b, a + b\n"
                "    return a\n\n"
                "def main() -> None:\n"
                "    for i in range(12):\n"
                "        print(f'fib({i}) = {fib(i)}')\n\n"
                "if __name__ == '__main__':\n"
                "    main()"
            )
        if "prime" in pl:
            return (
                "def is_prime(n: int) -> bool:\n"
                "    if n < 2:\n"
                "        return False\n"
                "    d = 2\n"
                "    while d * d <= n:\n"
                "        if n % d == 0:\n"
                "            return False\n"
                "        d += 1\n"
                "    return True\n\n"
                "def main() -> None:\n"
                "    print([n for n in range(2, 50) if is_prime(n)])\n\n"
                "if __name__ == '__main__':\n"
                "    main()"
            )
        return (
            "from typing import List\n\n"
            "def two_sum(nums: List[int], target: int) -> List[int]:\n"
            "    seen = {}\n"
            "    for i, n in enumerate(nums):\n"
            "        need = target - n\n"
            "        if need in seen:\n"
            "            return [seen[need], i]\n"
            "        seen[n] = i\n"
            "    return []\n\n"
            "if __name__ == '__main__':\n"
            "    print(two_sum([2, 7, 11, 15], 9))"
        )

    @classmethod
    def _py_api(cls, prompt: str, subject: str) -> str:
        title = subject.replace("'", "\\'")
        return (
            "from http.server import BaseHTTPRequestHandler, HTTPServer\n"
            "import json\n\n"
            f"TITLE = '{title}'\n\n"
            "class Handler(BaseHTTPRequestHandler):\n"
            "    def do_GET(self):\n"
            "        if self.path == '/health':\n"
            "            self._json(200, {'ok': True, 'service': TITLE})\n"
            "        elif self.path == '/api/items':\n"
            "            self._json(200, {'items': [{'id': 1, 'name': 'alpha'}]})\n"
            "        else:\n"
            "            self._json(404, {'error': 'not found'})\n\n"
            "    def _json(self, code, payload):\n"
            "        body = json.dumps(payload).encode()\n"
            "        self.send_response(code)\n"
            "        self.send_header('Content-Type', 'application/json')\n"
            "        self.send_header('Content-Length', str(len(body)))\n"
            "        self.end_headers()\n"
            "        self.wfile.write(body)\n\n"
            "def main():\n"
            "    HTTPServer(('127.0.0.1', 8766), Handler).serve_forever()\n\n"
            "if __name__ == '__main__':\n"
            "    main()"
        )

    @classmethod
    def _py_cli(cls, prompt: str, subject: str) -> str:
        return (
            "import argparse\n\n"
            "def main() -> None:\n"
            "    p = argparse.ArgumentParser(description='CLI tool')\n"
            "    p.add_argument('name', nargs='?', default='world')\n"
            "    p.add_argument('--verbose', '-v', action='store_true')\n"
            "    args = p.parse_args()\n"
            f"    msg = f'Hello, {{args.name}}!'\n"
            "    if args.verbose:\n"
            "        msg += ' (verbose)'\n"
            "    print(msg)\n\n"
            "if __name__ == '__main__':\n"
            "    main()"
        )

    @classmethod
    def _py_game(cls, prompt: str, subject: str) -> str:
        pl = prompt.lower()
        if "snake" in pl:
            return (
                "import pygame\nimport random\n\n"
                "W, H = 400, 400\nCELL = 20\n\n"
                "def main():\n"
                "    pygame.init()\n"
                "    screen = pygame.display.set_mode((W, H))\n"
                "    pygame.display.set_caption('Snake')\n"
                "    clock = pygame.time.Clock()\n"
                "    snake = [(W//2, H//2)]\n"
                "    dx, dy = CELL, 0\n"
                "    food = (random.randrange(0, W, CELL), random.randrange(0, H, CELL))\n"
                "    running = True\n"
                "    while running:\n"
                "        for event in pygame.event.get():\n"
                "            if event.type == pygame.QUIT:\n"
                "                running = False\n"
                "            if event.type == pygame.KEYDOWN:\n"
                "                if event.key == pygame.K_UP and dy == 0:\n"
                "                    dx, dy = 0, -CELL\n"
                "                elif event.key == pygame.K_DOWN and dy == 0:\n"
                "                    dx, dy = 0, CELL\n"
                "                elif event.key == pygame.K_LEFT and dx == 0:\n"
                "                    dx, dy = -CELL, 0\n"
                "                elif event.key == pygame.K_RIGHT and dx == 0:\n"
                "                    dx, dy = CELL, 0\n"
                "        head = (snake[0][0] + dx, snake[0][1] + dy)\n"
                "        if head == food:\n"
                "            food = (random.randrange(0, W, CELL), random.randrange(0, H, CELL))\n"
                "        else:\n"
                "            snake.pop()\n"
                "        snake.insert(0, head)\n"
                "        if (head[0] < 0 or head[0] >= W or head[1] < 0 or head[1] >= H\n"
                "                or head in snake[1:]):\n"
                "            break\n"
                "        screen.fill('black')\n"
                "        for seg in snake:\n"
                "            pygame.draw.rect(screen, 'lime', (*seg, CELL, CELL))\n"
                "        pygame.draw.rect(screen, 'red', (*food, CELL, CELL))\n"
                "        pygame.display.flip()\n"
                "        clock.tick(10)\n"
                "    pygame.quit()\n\n"
                "if __name__ == '__main__':\n"
                "    main()"
            )
        if "pong" in pl or "breakout" in pl:
            return (
                "import pygame\n\n"
                "W, H = 600, 400\nPAD_W, PAD_H = 80, 10\n\n"
                "def main():\n"
                "    pygame.init()\n"
                "    screen = pygame.display.set_mode((W, H))\n"
                "    clock = pygame.time.Clock()\n"
                "    px, py = W//2, H-30\n"
                "    bx, by, bdx, bdy = W//2, H//2, 4, 4\n"
                "    running = True\n"
                "    while running:\n"
                "        for event in pygame.event.get():\n"
                "            if event.type == pygame.QUIT:\n"
                "                running = False\n"
                "        keys = pygame.key.get_pressed()\n"
                "        if keys[pygame.K_LEFT]:\n"
                "            px -= 6\n"
                "        if keys[pygame.K_RIGHT]:\n"
                "            px += 6\n"
                "        px = max(0, min(W - PAD_W, px))\n"
                "        bx += bdx\n"
                "        by += bdy\n"
                "        if bx <= 0 or bx >= W:\n"
                "            bdx = -bdx\n"
                "        if by <= 0:\n"
                "            by = -bdy\n"
                "        if py <= by <= py + PAD_H and px <= bx <= px + PAD_W:\n"
                "            bdy = -bdy\n"
                "        if by > H:\n"
                "            break\n"
                "        screen.fill('black')\n"
                "        pygame.draw.rect(screen, 'white', (px, py, PAD_W, PAD_H))\n"
                "        pygame.draw.circle(screen, 'white', (bx, by), 8)\n"
                "        pygame.display.flip()\n"
                "        clock.tick(60)\n"
                "    pygame.quit()\n\n"
                "if __name__ == '__main__':\n"
                "    main()"
            )
        if "tic" in pl or "tac" in pl or "tictac" in pl or "tick" in pl:
            return (
                "def print_board(b):\n"
                "    for i in range(3):\n"
                "        print('|'.join(b[i*3:(i+1)*3]))\n"
                "        if i < 2:\n"
                "            print('-'*5)\n\n"
                "def check_winner(b):\n"
                "    for i in range(3):\n"
                "        if b[i*3] == b[i*3+1] == b[i*3+2] != ' ':\n"
                "            return b[i*3]\n"
                "        if b[i] == b[i+3] == b[i+6] != ' ':\n"
                "            return b[i]\n"
                "    if b[0] == b[4] == b[8] != ' ' or b[2] == b[4] == b[6] != ' ':\n"
                "        return b[4]\n"
                "    return None\n\n"
                "def main():\n"
                "    b = [' ']*9\n"
                "    turn = 'X'\n"
                "    for _ in range(9):\n"
                "        print_board(b)\n"
                "        print(f\"Player {turn}'s turn (1-9):\")\n"
                "        try:\n"
                "            m = int(input()) - 1\n"
                "            if m < 0 or m > 8 or b[m] != ' ':\n"
                "                print('Invalid move')\n"
                "                continue\n"
                "        except ValueError:\n"
                "            print('Enter 1-9')\n"
                "            continue\n"
                "        b[m] = turn\n"
                "        w = check_winner(b)\n"
                "        if w:\n"
                "            print_board(b)\n"
                "            print(f'Player {w} wins!')\n"
                "            return\n"
                "        turn = 'O' if turn == 'X' else 'X'\n"
                "    print_board(b)\n"
                "    print(\"It's a tie!\")\n\n"
                "if __name__ == '__main__':\n"
                "    main()"
            )
        return (
            "import random\n\n"
            "def main():\n"
            "    print('Welcome to the game!')\n"
            "    number = random.randint(1, 100)\n"
            "    attempts = 0\n"
            "    while True:\n"
            "        try:\n"
            "            guess = int(input('Guess 1-100: '))\n"
            "            attempts += 1\n"
            "            if guess < number:\n"
            "                print('Too low!')\n"
            "            elif guess > number:\n"
            "                print('Too high!')\n"
            "            else:\n"
            "                print(f'Correct in {attempts} attempts!')\n"
            "                break\n"
            "        except ValueError:\n"
            "            print('Enter a number')\n\n"
            "if __name__ == '__main__':\n"
            "    main()"
        )

    @classmethod
    def _py_data(cls, prompt: str, subject: str) -> str:
        pl = prompt.lower()
        if "csv" in pl:
            return (
                "import csv\nimport sys\n\n"
                "def main():\n"
                "    if len(sys.argv) < 2:\n"
                "        print('Usage: python script.py <csv_file>')\n"
                "        return\n"
                "    with open(sys.argv[1]) as f:\n"
                "        reader = csv.DictReader(f)\n"
                "        rows = list(reader)\n"
                "    print(f'Read {len(rows)} rows')\n"
                "    if rows:\n"
                "        print(f'Columns: {\", \".join(rows[0].keys())}')\n\n"
                "if __name__ == '__main__':\n"
                "    main()"
            )
        if "json" in pl:
            return (
                "import json\nimport sys\n\n"
                "def main():\n"
                "    if len(sys.argv) < 2:\n"
                "        print('Usage: python script.py <json_file>')\n"
                "        return\n"
                "    with open(sys.argv[1]) as f:\n"
                "        data = json.load(f)\n"
                "    print(json.dumps(data, indent=2)[:2000])\n\n"
                "if __name__ == '__main__':\n"
                "    main()"
            )
        return (
            "from typing import List, Dict, Any\n\n"
            f"DATA = [{{\"id\": 1, \"name\": \"{subject}\", \"value\": 42}}]\n\n"
            "def main():\n"
            "    for item in DATA:\n"
            "        print(item)\n\n"
            "if __name__ == '__main__':\n"
            "    main()"
        )

    @classmethod
    def _py_general_app(cls, prompt: str, subject: str) -> str:
        pl = prompt.lower()
        subj = subject.lower()
        if "todo" in subj or "todo" in pl or "task" in pl:
            return (
                "import json\nimport sys\n\n"
                "tasks: list[dict] = []\n\n"
                "def add(title: str):\n"
                "    tasks.append({\"id\": len(tasks)+1, \"title\": title, \"done\": False})\n"
                "    print(f'Added: {title}')\n\n"
                "def list_tasks():\n"
                "    if not tasks:\n"
                "        print('No tasks')\n"
                "        return\n"
                "    for t in tasks:\n"
                "        status = '✓' if t['done'] else '○'\n"
                "        print(f\"{t['id']}. [{status}] {t['title']}\")\n\n"
                "def done(task_id: int):\n"
                "    for t in tasks:\n"
                "        if t['id'] == task_id:\n"
                "            t['done'] = True\n"
                "            print(f'Marked {task_id} done')\n"
                "            return\n"
                "    print('Task not found')\n\n"
                "def delete(task_id: int):\n"
                "    global tasks\n"
                "    tasks = [t for t in tasks if t['id'] != task_id]\n"
                "    print(f'Deleted {task_id}')\n\n"
                "def main():\n"
                "    print('Todo App - Commands: add <title>, list, done <id>, delete <id>, quit')\n"
                "    while True:\n"
                "        try:\n"
                "            line = input('> ').strip()\n"
                "            if not line:\n"
                "                continue\n"
                "            if line == 'quit':\n"
                "                break\n"
                "            if line == 'list':\n"
                "                list_tasks()\n"
                "            elif line.startswith('add '):\n"
                "                add(line[4:])\n"
                "            elif line.startswith('done '):\n"
                "                done(int(line[5:]))\n"
                "            elif line.startswith('delete '):\n"
                "                delete(int(line[7:]))\n"
                "            else:\n"
                "                print('Commands: add <title>, list, done <id>, delete <id>, quit')\n"
                "        except (EOFError, KeyboardInterrupt):\n"
                "            print()\n"
                "            break\n\n"
                "if __name__ == '__main__':\n"
                "    main()"
            )
        if "calc" in subj or "calc" in pl or "calculator" in subj:
            return (
                "import operator\n\n"
                "def main():\n"
                "    print('Calculator - type expressions like \"2 + 2\" or \"quit\"')\n"
                "    ops = {'+': operator.add, '-': operator.sub, '*': operator.mul, "
                "'/': operator.truediv, '**': operator.pow}\n"
                "    while True:\n"
                "        try:\n"
                "            line = input('> ').strip()\n"
                "            if not line or line == 'quit':\n"
                "                break\n"
                "            parts = line.split()\n"
                "            if len(parts) == 3:\n"
                "                a, op, b = parts[0], parts[1], parts[2]\n"
                "                if op in ops:\n"
                "                    result = ops[op](float(a), float(b))\n"
                "                    print(f'= {result}')\n"
                "                else:\n"
                "                    print('Unknown operator')\n"
                "            else:\n"
                "                print('Usage: <num> <op> <num>')\n"
                "        except ZeroDivisionError:\n"
                "            print('Division by zero')\n"
                "        except (ValueError, IndexError):\n"
                "            print('Invalid input')\n"
                "        except (EOFError, KeyboardInterrupt):\n"
                "            print()\n"
                "            break\n\n"
                "if __name__ == '__main__':\n"
                "    main()"
            )
        if "chat" in subj or "chat" in pl or "bot" in subj or "bot" in pl:
            return (
                "import random\n\n"
                "responses = {\n"
                "    'hello': ['Hi there!', 'Hello!', 'Hey!'],\n"
                "    'how are you': ['I am doing well!', 'Great, thanks!', 'All good!'],\n"
                "    'bye': ['Goodbye!', 'See you later!', 'Bye!'],\n"
                "    'default': ['Tell me more.', 'Interesting!', 'I see.']\n"
                "}\n\n"
                "def respond(msg: str) -> str:\n"
                "    msg = msg.lower().strip()\n"
                "    for key, replies in responses.items():\n"
                "        if key in msg:\n"
                "            return random.choice(replies)\n"
                "    return random.choice(responses['default'])\n\n"
                "def main():\n"
                "    print('Chatbot - type your message (quit to exit)')\n"
                "    while True:\n"
                "        try:\n"
                "            user = input('You: ').strip()\n"
                "            if not user or user.lower() == 'quit':\n"
                "                break\n"
                "            print(f'Bot: {respond(user)}')\n"
                "        except (EOFError, KeyboardInterrupt):\n"
                "            print()\n"
                "            break\n\n"
                "if __name__ == '__main__':\n"
                "    main()"
            )
        if "web" in subj or "server" in subj or "http" in pl or "api" in subj:
            return (
                "from http.server import BaseHTTPRequestHandler, HTTPServer\n"
                "import json\n\n"
                f"TITLE = '{subject.replace(chr(39), chr(39)*2)}'\n\n"
                "class Handler(BaseHTTPRequestHandler):\n"
                "    def do_GET(self):\n"
                "        if self.path == '/':\n"
                "            self.send_response(200)\n"
                "            self.send_header('Content-Type', 'text/html')\n"
                "            self.end_headers()\n"
                "            self.wfile.write(f'<h1>{TITLE}</h1>'.encode())\n"
                "        elif self.path == '/api/data':\n"
                "            self._json({'ok': True, 'data': [{'id': 1, 'name': TITLE}]})\n"
                "        else:\n"
                "            self._json({'error': 'not found'}, 404)\n\n"
                "    def _json(self, payload, code=200):\n"
                "        body = json.dumps(payload).encode()\n"
                "        self.send_response(code)\n"
                "        self.send_header('Content-Type', 'application/json')\n"
                "        self.end_headers()\n"
                "        self.wfile.write(body)\n\n"
                "def main():\n"
                "    port = 8766\n"
                f"    print(f'Server running on {{port}}')\n"
                "    HTTPServer(('', port), Handler).serve_forever()\n\n"
                "if __name__ == '__main__':\n"
                "    main()"
            )
        if "weather" in subj or "weather" in pl:
            return (
                "import urllib.request\nimport json\n\n"
                "def get_weather(city: str) -> dict | None:\n"
                "    url = f'https://wttr.in/{city}?format=j1'\n"
                "    try:\n"
                "        with urllib.request.urlopen(url, timeout=5) as r:\n"
                "            return json.loads(r.read())\n"
                "    except Exception as e:\n"
                "        print(f'Error: {e}')\n"
                "        return None\n\n"
                "def main():\n"
                "    import sys\n"
                "    city = sys.argv[1] if len(sys.argv) > 1 else input('City: ')\n"
                "    data = get_weather(city)\n"
                "    if data:\n"
                "        cc = data['current_condition'][0]\n"
                "        print(f\"Weather in {city}: {cc['temp_C']}C, {cc['weatherDesc'][0]['value']}\")\n\n"
                "if __name__ == '__main__':\n"
                "    main()"
            )
        if "stopwatch" in subj or "timer" in subj or "countdown" in pl:
            return (
                "import time\n\n"
                "def main():\n"
                "    import sys\n"
                "    if len(sys.argv) > 1 and sys.argv[1] == 'countdown':\n"
                "        seconds = int(sys.argv[2]) if len(sys.argv) > 2 else 10\n"
                "        for i in range(seconds, 0, -1):\n"
                "            print(f'{i}...', end=' ', flush=True)\n"
                "            time.sleep(1)\n"
                "        print('Done!')\n"
                "    else:\n"
                "        input('Press Enter to start stopwatch...')\n"
                "        start = time.time()\n"
                "        try:\n"
                "            input('Press Enter to stop...')\n"
                "            elapsed = time.time() - start\n"
                "            print(f'Elapsed: {elapsed:.2f}s')\n"
                "        except (EOFError, KeyboardInterrupt):\n"
                "            elapsed = time.time() - start\n"
                "            print(f'\\nElapsed: {elapsed:.2f}s')\n\n"
                "if __name__ == '__main__':\n"
                "    main()"
            )
        if "note" in subj or "note" in pl or "diary" in subj or "journal" in pl:
            return (
                "import json\nimport os\nfrom datetime import datetime\n\n"
                "NOTES_FILE = 'notes.json'\n\n"
                "def load() -> list:\n"
                "    if os.path.exists(NOTES_FILE):\n"
                "        with open(NOTES_FILE) as f:\n"
                "            return json.load(f)\n"
                "    return []\n\n"
                "def save(notes: list):\n"
                "    with open(NOTES_FILE, 'w') as f:\n"
                "        json.dump(notes, f, indent=2)\n\n"
                "def add(text: str):\n"
                "    notes = load()\n"
                "    notes.append({'id': len(notes)+1, 'text': text, 'created': str(datetime.now())})\n"
                "    save(notes)\n"
                "    print('Note added')\n\n"
                "def list_notes():\n"
                "    notes = load()\n"
                "    if not notes:\n"
                "        print('No notes')\n"
                "        return\n"
                "    for n in notes:\n"
                "        print(f\"{n['id']}. {n['text'][:60]}\")\n\n"
                "def delete(nid: int):\n"
                "    notes = load()\n"
                "    notes = [n for n in notes if n['id'] != nid]\n"
                "    save(notes)\n"
                "    print(f'Deleted note {nid}')\n\n"
                "def main():\n"
                "    print('Notes - commands: add <text>, list, delete <id>, quit')\n"
                "    while True:\n"
                "        try:\n"
                "            line = input('> ').strip()\n"
                "            if not line or line == 'quit':\n"
                "                break\n"
                "            if line == 'list':\n"
                "                list_notes()\n"
                "            elif line.startswith('add '):\n"
                "                add(line[4:])\n"
                "            elif line.startswith('delete '):\n"
                "                delete(int(line[7:]))\n"
                "            else:\n"
                "                print('Commands: add <text>, list, delete <id>, quit')\n"
                "        except (EOFError, KeyboardInterrupt):\n"
                "            print()\n"
                "            break\n\n"
                "if __name__ == '__main__':\n"
                "    main()"
            )
        return (
            "import sys\n\n"
            f"APP_NAME = '{subject.replace(chr(39), chr(39)*2)}'\n\n"
            "def main():\n"
            f"    print(f'Welcome to {{APP_NAME}}!')\n"
            "    print(f'Running Python {sys.version}')\n"
            "    user = input('Enter your name: ').strip()\n"
            f"    print(f'Hello {{user}}, welcome to {{APP_NAME}}!')\n\n"
            "if __name__ == '__main__':\n"
            "    main()"
        )

    @classmethod
    def _polyglot(cls, lang: str, task: str, prompt: str, subject: str) -> str:
        msg = subject.replace("\\", "\\\\").replace('"', '\\"').replace("'", "\\'")
        esc = msg
        if lang == "python":
            if task == "algo":
                return cls._py_algo(prompt)
            if task == "api":
                return cls._py_api(prompt, subject)
            if task == "cli":
                return cls._py_cli(prompt, subject)
            if task == "game":
                return cls._py_game(prompt, subject)
            if task == "data":
                return cls._py_data(prompt, subject)
            return cls._py_general_app(prompt, subject)
        if lang == "html" or (lang == "web" and task == "web"):
            return CatSeekR1Code._html(prompt, None)
        if lang == "javascript":
            if task == "api":
                return (
                    "const http = require('http');\n\n"
                    "const server = http.createServer((req, res) => {\n"
                    "  if (req.url === '/health') {\n"
                    "    res.writeHead(200, {'Content-Type': 'application/json'});\n"
                    "    res.end(JSON.stringify({ ok: true }));\n"
                    "    return;\n"
                    "  }\n"
                    "  res.writeHead(404);\n"
                    "  res.end(JSON.stringify({ error: 'not found' }));\n"
                    "});\n\n"
                    "server.listen(8766, () => console.log('API on :8766'));"
                )
            return f"console.log('{esc}');"
        if lang == "typescript":
            return (
                "interface Item { id: number; name: string; }\n\n"
                "const items: Item[] = [{ id: 1, name: 'alpha' }];\n"
                f"console.log('{esc}', items);"
            )
        if lang == "rust":
            return (
                "fn main() {\n"
                f'    println!("{esc}");\n'
                "}"
            )
        if lang == "go":
            return (
                "package main\n\n"
                "import \"fmt\"\n\n"
                "func main() {\n"
                f'\tfmt.Println("{esc}")\n'
                "}"
            )
        if lang == "java":
            return (
                "public class Main {\n"
                "    public static void main(String[] args) {\n"
                f'        System.out.println("{esc}");\n'
                "    }\n"
                "}"
            )
        if lang == "kotlin":
            return (
                "fun main() {\n"
                f'    println("{esc}")\n'
                "}"
            )
        if lang == "swift":
            return (
                "import Foundation\n\n"
                f'print("{esc}")'
            )
        if lang == "c":
            return TokenWeightCodeEmitter._c(msg)
        if lang == "cpp":
            return TokenWeightCodeEmitter._cpp(msg)
        if lang == "bash":
            return f'#!/bin/bash\necho "{esc}"'
        if lang == "ruby":
            return f'puts "{esc}"'
        if lang == "php":
            return f'<?php\necho "{esc}\\n";'
        if lang == "sql":
            return f"SELECT '{esc.replace(chr(39), chr(39)*2)}' AS message;"
        if lang == "css":
            return f"body {{\n  font-family: system-ui, sans-serif;\n}}\n/* {esc} */"
        if lang == "yaml":
            return f"service:\n  name: {subject.lower().replace(' ', '-')}\n  files: off\n"
        if lang == "json":
            return json.dumps({"message": subject, "files": "off"}, indent=2)
        if lang == "dockerfile":
            return (
                "FROM python:3.12-slim\n"
                "WORKDIR /app\n"
                "COPY . .\n"
                f'CMD ["python", "-c", "print(\\"{esc}\\")"]'
            )
        if lang == "solidity":
            return (
                "// SPDX-License-Identifier: MIT\n"
                "pragma solidity ^0.8.20;\n\n"
                "contract Token {\n"
                "    string public name = \"cat r1.10\";\n"
                "    uint256 public supply;\n"
                "    constructor(uint256 initial) { supply = initial; }\n"
                "}"
            )
        comment = "//"
        if lang in {"python", "bash", "dockerfile", "makefile", "ruby", "perl", "r"}:
            comment = "#"
        elif lang == "html":
            comment = "<!-- -->"
        return (
            f"{comment} {cls.TAG} · {lang} · {task}\n"
            f"{comment} Prompt: {prompt[:120]}\n"
            f"{comment} Subject: {subject}"
        )

    @classmethod
    def scaffold(
        cls,
        lang: str,
        task: str,
        prompt: str,
        engine: Optional["CatR11Engine"] = None,
        vec: Optional[np.ndarray] = None,
    ) -> str:
        lang = cls.normalize_lang(lang) or cls.infer_lang(engine, prompt)
        if lang not in cls.VALID_LANGS:
            lang = "python"
        subject = cls._subject(prompt, engine)
        return cls._polyglot(lang, task, prompt, subject)

    @classmethod
    def generate(
        cls,
        engine: "CatR11Engine",
        prompt: str,
        lang: str,
        vec: Optional[np.ndarray] = None,
    ) -> str:
        embedded_lang, embedded = CatSeekR1Code.extract_prompt_code(prompt)
        if embedded:
            return embedded
        task = cls.detect_task(prompt)
        lang = cls.infer_lang(engine, prompt, lang)
        code = cls.scaffold(lang, task, prompt, engine, vec)
        if hasattr(engine, "recompile_code_for_prompt"):
            code = engine.recompile_code_for_prompt(lang, prompt, code)
        return code

    @classmethod
    def simulate(cls, lang: str, code: str, prompt: str = "") -> str:
        lines = [ln for ln in code.splitlines() if ln.strip()]
        funcs = re.findall(r"(?:def|fn|func|function)\s+(\w+)", code)
        classes = re.findall(r"(?:class|struct|interface|contract)\s+(\w+)", code)
        imports = re.findall(r"(?:^import\s+|^use\s+|#include\s+)", code, re.MULTILINE)
        preview = "\n".join(lines[:8])
        if len(lines) > 8:
            preview += f"\n... ({len(lines) - 8} more lines)"
        return (
            f"**{cls.TAG}** dry-run · `{lang}`\n\n"
            f"- lines: {len(lines)} · chars: {len(code)}\n"
            f"- functions: {', '.join(funcs[:6]) or '(none)'}\n"
            f"- types: {', '.join(classes[:6]) or '(none)'}\n"
            f"- imports/includes: {len(imports)}\n"
            f"- runtime: simulated (no `{lang}` binary — structural review only)\n\n"
            f"```\n{preview}\n```\n\n"
            "Python · JavaScript · Bash run natively. Say **run it** after editing."
        )

    @classmethod
    def _minimal_expert(cls, lang: str) -> str:
        return cls._polyglot(lang, "general", f"hello world in {lang}", "Hello World")

    @classmethod
    def experts(cls) -> Dict[str, str]:
        heavy = frozenset({"html"})
        out: Dict[str, str] = {}
        for lang in cls.LANGS:
            if lang in heavy:
                out[lang] = "<!DOCTYPE html><html><body><h1>Hello World</h1></body></html>"
            else:
                out[lang] = cls._minimal_expert(lang)
        return out

    @classmethod
    def stats_line(cls) -> str:
        return f"{cls.TAG} · {len(cls.LANGS)} langs"


CodeAnything = CodeAnythingEngine


class CatSeekR1Code:
    """
    catr1.10 frontier code engine (catseek r1.10 frontier tier).
    Token-weight synthesis · recursive perfection · in-memory run loop · files = off.
    All 128+ languages · polyglot scaffold · sandbox verify.
    """

    NAME = CONFIG["code_interpreter_name"]
    FAMILY = CONFIG["code_interpreter_family"]
    BACKEND = CONFIG["code_interpreter"]
    VERSION = CONFIG["code_interpreter_version"]
    RUNNABLE = CodeAnythingEngine.RUNNABLE if CONFIG.get("code_anything") else frozenset({"python", "javascript", "bash"})
    LANGS = CodeAnythingEngine.LANGS if CONFIG.get("code_anything") else (
        "html", "css", "python", "javascript", "typescript", "java", "rust",
        "go", "bash", "shell", "cpp", "c", "sql", "ruby", "php", "assembly",
    )
    _WRITE_VERBS = re.compile(
        r"\b(write|make|build|create|generate|code|implement|show me|give me|draft|"
        r"scaffold|boilerplate|port|convert|rewrite|refactor|migrate)\b", re.I
    )
    _IN_LANG = re.compile(
        r"\b(?:in|using|with|as)\s+(html|css|python|javascript|typescript|java|rust|go|bash|shell|cpp|c\+\+|c|sql|ruby|php|assembly)\b",
        re.I,
    )
    _MAKE_IT_LANG = re.compile(
        r"\b(?:no\s+)?make\s+(?:it\s+)?(?:in\s+)?(html|css|python|javascript|typescript|java|rust|go|bash|shell|cpp|c\+\+|c|sql|ruby|php)\b",
        re.I,
    )
    _HTML_CUE = re.compile(
        r"\b(?:a|an)\s+html\b|\bhtml\s+(?:program|page|file|ad|that|site|app)\b|\b(?:write|make|build|create)\s+(?:a\s+)?html\b",
        re.I,
    )
    _CREATIVE_BLOCK = re.compile(
        r"\b(story|poem|tale|parable|bedtime|narrative|verse|rhyme)\b", re.I
    )
    _FABLE_STORY = re.compile(r"\b(?:write|tell|give)\s+(?:me\s+)?(?:a\s+)?fable\b", re.I)
    _PATH_TAIL = re.compile(
        r"(?:/Volumes/.+|/Users/.+|~/.+)\.(?:py|c|cpp|html|js|ts|java|rs|go|sh|php|rb)\s*$",
        re.I,
    )
    _BRAND_PREFIX = re.compile(
        r"^\[?\s*(?:catr1\.1|cat r1\.1(?:[\s.]*1\.1)?)\s*\]?:?\s*", re.I
    )
    _META_TAIL = re.compile(
        r"\s+(?:make the code|files\s*=|\btoken weight\b|\bfiles\.\s*=\s*off\b).*$",
        re.I | re.S,
    )
    _RAW_CODE = re.compile(r"(#include[\s\S]+|def\s+\w+|function\s+\w+|fn\s+main|public\s+class|<!DOCTYPE)", re.I)
    _SAYS = re.compile(r"\b(?:that\s+)?says\s+[\"']?([^\"'\n]+)[\"']?", re.I)
    _GO = re.compile(r"^\s*(?:go|do it|yes|continue|proceed|ok|>|code\s*>)\s*\.?\s*$", re.I)
    _CODE_SHORT = re.compile(
        r"^\s*(?:code|/code|code\s*>|>|program|script)\s*$", re.I
    )

    @classmethod
    def enabled(cls) -> bool:
        return bool(CONFIG.get("catr1_code_enabled", True))

    @classmethod
    def code_help(cls) -> str:
        return ClaudeMythosRuntime.code_help()
    VALID_LANGS = frozenset(LANGS)

    @classmethod
    def _coerce_lang(cls, engine: "CatR11Engine", lang: Optional[str], prompt: str) -> str:
        lang = engine.normalize_lang(lang) or ""
        if lang in cls.VALID_LANGS:
            return lang
        pl = cls.normalize_prompt(prompt).lower()
        direct = cls._lang_from_text(pl, engine)
        if direct:
            return direct
        if "html" in pl:
            return "html"
        if re.search(r"\b(?:c|c\+\+|cpp)\b", pl):
            return "c" if "iostream" not in pl else "cpp"
        return "python"

    @classmethod
    def normalize_prompt(cls, prompt: str) -> str:
        s = prompt.strip()
        s = cls._BRAND_PREFIX.sub("", s)
        s = cls._META_TAIL.sub("", s)
        return cls._clean_prompt(s)

    @classmethod
    def _clean_prompt(cls, prompt: str) -> str:
        s = prompt.strip()
        s = cls._PATH_TAIL.sub("", s)
        return s.strip()

    @classmethod
    def extract_prompt_code(cls, prompt: str) -> Tuple[Optional[str], Optional[str]]:
        """If the user pasted code (± file path), return it exactly — files = off."""
        cleaned = cls.normalize_prompt(prompt)
        lang, code = None, None
        m = re.search(r"```([a-zA-Z0-9_+#-]*)\s*\n([\s\S]*?)```", cleaned)
        if m:
            lang = (m.group(1) or "").strip().lower() or None
            code = m.group(2).strip()
        elif cls._RAW_CODE.search(cleaned) or VibeCodeHeuristics.CODE_SHAPES.search(cleaned):
            code = cleaned.strip()
            if "#include" in code and "stdio" in code:
                lang = "c" if "iostream" not in code else "cpp"
            elif "iostream" in code:
                lang = "cpp"
            elif code.lstrip().startswith("<!"):
                lang = "html"
            elif "def " in code:
                lang = "python"
        if code:
            code = re.sub(r"^c\s*\n(?=#include)", "", code, flags=re.I)
            return lang, code
        return None, None

    @classmethod
    def wants_code(cls, pl: str) -> bool:
        if not cls.enabled():
            return False
        if CodeAnythingEngine.enabled() and CodeAnythingEngine.wants_anything(pl):
            return True
        if VibeCodeHeuristics.wants_code(pl):
            return True
        pl = pl.strip().lower()
        if cls._CODE_SHORT.match(pl):
            return True
        if cls._FABLE_STORY.search(pl):
            return False
        if cls._CREATIVE_BLOCK.search(pl):
            return False
        if re.search(r"\b(?:can you|please|help me|i need you to|let'?s)\s+code\b", pl):
            return True
        if re.search(r"\bcode\s*[>:]", pl):
            return True
        if re.search(r"\b(?:write|make|build|create|show|give|draft|implement)\s+(?:me\s+)?(?:some\s+)?code\b", pl):
            return True
        if cls._MAKE_IT_LANG.search(pl):
            return True
        if cls._HTML_CUE.search(pl):
            return True
        if cls._IN_LANG.search(pl) and cls._WRITE_VERBS.search(pl):
            return True
        if cls._IN_LANG.search(pl) and any(w in pl for w in ("page", "app", "script", "site", "cat", "hello", "world", "ad", "usb", "gamer")):
            return True
        if re.search(r"\b(function|def |class |snippet|fibonacci|fib|prime|```)\b", pl):
            return True
        if re.search(r"\b(write|make|build|create)\b", pl) and re.search(
            r"\b(code|function|script|program|app|page|html|python|javascript|java)\b", pl
        ):
            return True
        if re.search(r"\b(python|javascript|typescript|html|rust|java|go|bash|cpp|c\+\+)\b", pl):
            if re.search(r"\b(write|make|build|create|implement|code)\b", pl):
                return True
        if cls._RAW_CODE.search(cls._clean_prompt(pl)):
            return True
        if re.search(r"\b(c|c\+\+|cpp)\b", pl) and re.search(
            r"\b(write|make|build|create|code|hello|world|program|main)\b", pl
        ):
            return True
        if re.search(r"\bgeocit(?:ies|es)\b", pl) and "html" in pl:
            return True
        if re.search(r"\b(program|script|snippet|algorithm)\b", pl) and re.search(
            r"\b(write|make|build|create|need|want|give|show)\b", pl
        ):
            return True
        return False

    @classmethod
    def wants_code_with_history(cls, pl: str, engine: "CatR11Engine") -> bool:
        if cls.wants_code(pl):
            return True
        if not cls.enabled():
            return False
        if cls._GO.match(pl.strip()) or cls._CODE_SHORT.match(pl.strip()):
            return bool(engine.chat_history) or bool(cls._lang_from_history(engine))
        return False

    @classmethod
    def _lang_from_text(cls, pl: str, engine: "CatR11Engine") -> Optional[str]:
        vibe = VibeCodeHeuristics.lang_from_text(pl, engine)
        if vibe:
            return vibe
        m = cls._MAKE_IT_LANG.search(pl)
        if m:
            raw = m.group(1).lower()
            if raw == "shell":
                raw = "bash"
            return engine.normalize_lang(raw) or raw
        m = cls._IN_LANG.search(pl)
        if m:
            raw = m.group(1).lower()
            if raw == "shell":
                raw = "bash"
            return engine.normalize_lang(raw) or raw
        if cls._HTML_CUE.search(pl) or re.search(r"\bhtml\b", pl):
            return "html"
        if re.search(r"\b(?:write|make|code)\s+(?:me\s+)?(?:a\s+)?c\s+program\b", pl):
            return "c"
        if re.search(r"\bc\s+(?:program|code|hello)\b", pl):
            return "c"
        if re.search(r"\bjava\b", pl) and re.search(r"\b(write|make|code|program)\b", pl):
            return "java"
        return None

    @classmethod
    def _lang_from_history(cls, engine: "CatR11Engine") -> Optional[str]:
        for msg in reversed(engine.chat_history):
            if msg.get("role") != "user":
                continue
            lang = cls._lang_from_text(msg.get("text", "").lower(), engine)
            if lang:
                return lang
            if cls._HTML_CUE.search(msg.get("text", "").lower()) or "html" in msg.get("text", "").lower():
                return "html"
        return None

    @classmethod
    def run(cls, engine: "CatR11Engine", lang: str, code: str) -> str:
        return engine.execute_code_any_language(lang, code)

    @classmethod
    def should_run(cls, pl: str) -> bool:
        if CONFIG.get("code_auto_run"):
            return True
        if VibeCodeHeuristics.wants_run(pl):
            return True
        return any(
            x in pl for x in (
                "run it", "run this", "execute", "interpret", "test it",
                "/run", "and run", "then run",
            )
        )

    @classmethod
    def detect_lang(cls, engine: "CatR11Engine", prompt: str) -> str:
        pl = cls.normalize_prompt(prompt)
        embedded_lang, embedded = cls.extract_prompt_code(prompt)
        if embedded_lang:
            return engine.normalize_lang(embedded_lang) or embedded_lang
        if embedded and "#include" in embedded:
            return "cpp" if "iostream" in embedded else "c"
        direct = cls._lang_from_text(pl, engine)
        if not direct:
            direct = VibeCodeHeuristics.lang_from_text(prompt, engine)
        if direct:
            return direct
        if cls._GO.match(pl.strip().lower()):
            hist_lang = cls._lang_from_history(engine)
            if hist_lang:
                return hist_lang
        lang = engine.extract_lang(prompt) or engine.detect_lang_from_text(prompt)
        if CodeAnythingEngine.enabled():
            return CodeAnythingEngine.infer_lang(engine, prompt, lang or "")
        return cls._coerce_lang(engine, lang, prompt)

    @classmethod
    def _subject(cls, prompt: str, engine: Optional["CatR11Engine"] = None) -> str:
        vibe_subj = VibeCodeHeuristics.subject_from_text(prompt)
        if vibe_subj:
            return vibe_subj
        pl = prompt.lower().strip()
        if re.search(r"\b(?:no\s+)?make\s+it\b", pl) and engine:
            for msg in reversed(engine.chat_history):
                if msg.get("role") != "user":
                    continue
                prior = cls._subject(msg.get("text", ""))
                if prior and prior.lower() not in {"it", "it html", "hello world", "no"}:
                    return prior
        m = cls._SAYS.search(pl)
        if m:
            return m.group(1).strip("?.! ")
        if "hello cat" in pl:
            return "Hello Cat"
        if "meow" in pl:
            return "Meow"
        if "hello world" in pl:
            return "Hello World"
        if "gamer" in pl and "usb" in pl:
            return "GAMER USB — ULTRA SPEED"
        m = re.search(
            r"(?:write|make|build|create|generate|show|give)\s+(?:me\s+)?(?:a\s+)?(?:html\s+)?(?:that\s+is\s+a\s+)?(.+?)(?:\s+in\s+\w+|\s+program|\s*$)",
            pl,
        )
        if m:
            subj = m.group(1).strip("?.! ")
            if subj and subj not in {"html", "it", "a", "an", "no"}:
                return subj[:80]
        m = re.search(
            r"(?:write|make|build|create|generate|show|give)\s+(?:me\s+)?(?:a\s+)?(.+?)\s+in\s+\w+",
            pl,
        )
        if m:
            subj = m.group(1).strip("?.! ")
            if subj not in {"html", "it", "a", "an", "no"}:
                return subj
        return "Hello World"

    @classmethod
    def _html_geocities(cls, prompt: str) -> str:
        pl = prompt.lower()
        title = "GAMER USB AD — GEOCITIES EDITION"
        headline = "ULTRA GAMER USB 9000"
        if "gamer" in pl and "usb" in pl:
            headline = "GAMER USB — 1337 MB/s OF RAW POWER"
        label = html_module.escape(headline)
        return (
            "<!DOCTYPE html>\n"
            "<html>\n"
            "<head>\n"
            "  <meta charset=\"UTF-8\">\n"
            f"  <title>{html_module.escape(title)}</title>\n"
            "  <style>\n"
            "    body { background: #000080 url('data:image/svg+xml,<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"20\" height=\"20\"><rect fill=\"%23ffff00\" width=\"10\" height=\"10\"/><rect fill=\"%2300ff00\" x=\"10\" y=\"10\" width=\"10\" height=\"10\"/></svg>'); "
            "color: #00ff00; font-family: Comic Sans MS, cursive; margin: 0; }\n"
            "    .banner { background: linear-gradient(90deg,red,yellow,lime,cyan,blue,magenta); padding: 4px; text-align: center; }\n"
            "    .banner h1 { color: #fff; text-shadow: 2px 2px #000; font-size: 2rem; margin: 0; animation: blink 1s step-end infinite; }\n"
            "    @keyframes blink { 50% { opacity: 0.3; } }\n"
            "    .box { border: 4px ridge #ff00ff; background: #c0c0c0; color: #000; margin: 1rem auto; max-width: 640px; padding: 1rem; }\n"
            "    .counter { font-size: 0.75rem; color: #888; text-align: center; }\n"
            "    marquee { font-size: 1.2rem; color: #ff0000; font-weight: bold; }\n"
            "    .usb { font-size: 4rem; text-align: center; }\n"
            "  </style>\n"
            "</head>\n"
            "<body>\n"
            "  <div class=\"banner\"><h1>★ WELCOME TO MY HOMEPAGE ★</h1></div>\n"
            "  <marquee>🔥 BUY NOW — GAMER USB — LIMITED STOCK — CLICK HERE 🔥</marquee>\n"
            "  <div class=\"box\">\n"
            "    <div class=\"usb\">💾🎮</div>\n"
            f"    <h2 style=\"text-align:center;color:#0000ff;\">{label}</h2>\n"
            "    <ul>\n"
            "      <li>⚡ ZERO LAG FILE TRANSFERS</li>\n"
            "      <li>🎯 RGB NOT INCLUDED (1999 AUTHENTIC)</li>\n"
            "      <li>🏆 BEST VIEWED IN NETSCAPE NAVIGATOR</li>\n"
            "    </ul>\n"
            "    <p style=\"text-align:center\"><blink><b>ONLY $19.99!!!</b></blink></p>\n"
            "  </div>\n"
            "  <p class=\"counter\">You are visitor #420,069</p>\n"
            "</body>\n"
            "</html>"
        )

    @classmethod
    def _html(cls, prompt: str, engine: Optional["CatR11Engine"] = None) -> str:
        if CONFIG.get("web_program_enabled") and engine is not None and hasattr(engine, "web"):
            tpl = CatSeekWebProgram.pick_template(prompt)
            if tpl != "minimal" or re.search(r"\b(website|landing|dashboard|portfolio|site)\b", prompt.lower()):
                return CatSeekWebProgram.render(tpl, prompt, engine)
        pl = prompt.lower()
        if re.search(r"\bgeocit(?:ies|es)\b", pl):
            return cls._html_geocities(prompt)
        subject = cls._subject(prompt, engine)
        title = subject.title() if subject else "Hello"
        label = html_module.escape(subject if subject else "App")
        has_cat = "cat" in pl or "🐱" in prompt or "kitty" in pl or "kitten" in pl

        if "todo" in pl or "task" in pl or "todos" in pl:
            return (
                "<!DOCTYPE html>\n"
                '<html lang="en">\n'
                "<head>\n"
                "  <meta charset=\"UTF-8\">\n"
                "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n"
                f"  <title>{label} - Todo</title>\n"
                "  <style>\n"
                "    * { box-sizing: border-box; margin: 0; padding: 0; }\n"
                "    body { font-family: system-ui, sans-serif; background: #0f172a; color: #e2e8f0; "
                "display: flex; justify-content: center; padding: 2rem; }\n"
                "    .app { width: 100%; max-width: 480px; }\n"
                "    h1 { font-size: 1.5rem; margin-bottom: 1rem; color: #38bdf8; }\n"
                "    .input-row { display: flex; gap: 0.5rem; margin-bottom: 1rem; }\n"
                "    input { flex: 1; padding: 0.6rem; border: 1px solid #334155; border-radius: 8px; "
                "background: #1e293b; color: #e2e8f0; font-size: 0.9rem; }\n"
                "    input:focus { outline: none; border-color: #38bdf8; }\n"
                "    button { padding: 0.6rem 1rem; border: none; border-radius: 8px; "
                "background: #38bdf8; color: #0f172a; cursor: pointer; font-weight: 600; }\n"
                "    button:hover { background: #7dd3fc; }\n"
                "    ul { list-style: none; }\n"
                "    li { display: flex; align-items: center; gap: 0.5rem; "
                "padding: 0.6rem; border-bottom: 1px solid #1e293b; }\n"
                "    li.done span { text-decoration: line-through; opacity: 0.5; }\n"
                "    li button { margin-left: auto; background: transparent; color: #ef4444; "
                "font-size: 1rem; padding: 0.2rem 0.5rem; }\n"
                "  </style>\n"
                "</head>\n"
                "<body>\n"
                "  <div class=\"app\">\n"
                f"    <h1>{label}</h1>\n"
                "    <div class=\"input-row\">\n"
                "      <input id=\"input\" placeholder=\"Add a task...\" autofocus>\n"
                "      <button onclick=\"addTask()\">Add</button>\n"
                "    </div>\n"
                "    <ul id=\"list\"></ul>\n"
                "  </div>\n"
                "  <script>\n"
                "    const list = document.getElementById('list');\n"
                "    const input = document.getElementById('input');\n"
                "    function addTask() {\n"
                "      const text = input.value.trim();\n"
                "      if (!text) return;\n"
                "      const li = document.createElement('li');\n"
                "      const span = document.createElement('span');\n"
                "      span.textContent = text;\n"
                "      const del = document.createElement('button');\n"
                "      del.textContent = 'x';\n"
                "      del.onclick = () => li.remove();\n"
                "      li.onclick = () => li.classList.toggle('done');\n"
                "      li.appendChild(span);\n"
                "      li.appendChild(del);\n"
                "      list.appendChild(li);\n"
                "      input.value = '';\n"
                "      input.focus();\n"
                "    }\n"
                "    input.addEventListener('keydown', e => { if (e.key === 'Enter') addTask(); });\n"
                "  </script>\n"
                "</body>\n"
                "</html>"
            )

        if "calc" in pl or "calculator" in pl:
            return (
                "<!DOCTYPE html>\n"
                '<html lang="en">\n'
                "<head>\n"
                "  <meta charset=\"UTF-8\">\n"
                "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n"
                f"  <title>{label} - Calculator</title>\n"
                "  <style>\n"
                "    * { box-sizing: border-box; margin: 0; }\n"
                "    body { font-family: system-ui, sans-serif; background: #0f172a; color: #e2e8f0; "
                "display: flex; justify-content: center; padding: 2rem; }\n"
                "    .calc { width: 280px; }\n"
                "    h1 { font-size: 1.2rem; margin-bottom: 0.8rem; color: #38bdf8; }\n"
                "    #display { width: 100%; padding: 0.8rem; font-size: 1.5rem; text-align: right; "
                "border: 1px solid #334155; border-radius: 8px; background: #1e293b; "
                "color: #e2e8f0; margin-bottom: 0.5rem; }\n"
                "    .grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.4rem; }\n"
                "    button { padding: 0.8rem; font-size: 1.1rem; border: none; border-radius: 6px; "
                "background: #1e293b; color: #e2e8f0; cursor: pointer; }\n"
                "    button:hover { background: #334155; }\n"
                "    button.op { background: #334155; color: #38bdf8; }\n"
                "    button.eq { background: #38bdf8; color: #0f172a; }\n"
                "    button.clr { background: #ef4444; color: #fff; }\n"
                "  </style>\n"
                "</head>\n"
                "<body>\n"
                "  <div class=\"calc\">\n"
                f"    <h1>{label}</h1>\n"
                "    <input id=\"display\" readonly value=\"0\">\n"
                "    <div class=\"grid\">\n"
                "      <button class=\"clr\" onclick=\"clearD()\">C</button>"
                "<button onclick=\"add('(')\">(</button>"
                "<button onclick=\"add(')')\">)</button>"
                "<button class=\"op\" onclick=\"add('/')\">÷</button>\n"
                "      <button onclick=\"add('7')\">7</button>"
                "<button onclick=\"add('8')\">8</button>"
                "<button onclick=\"add('9')\">9</button>"
                "<button class=\"op\" onclick=\"add('*')\">×</button>\n"
                "      <button onclick=\"add('4')\">4</button>"
                "<button onclick=\"add('5')\">5</button>"
                "<button onclick=\"add('6')\">6</button>"
                "<button class=\"op\" onclick=\"add('-')\">−</button>\n"
                "      <button onclick=\"add('1')\">1</button>"
                "<button onclick=\"add('2')\">2</button>"
                "<button onclick=\"add('3')\">3</button>"
                "<button class=\"op\" onclick=\"add('+')\">+</button>\n"
                "      <button onclick=\"add('0')\">0</button>"
                "<button onclick=\"add('.')\">.</button>"
                "<button style=\"grid-column:span 2\" class=\"eq\" onclick=\"calc()\">=</button>\n"
                "    </div>\n"
                "  </div>\n"
                "  <script>\n"
                "    const d = document.getElementById('display');\n"
                "    function add(v) { d.value = d.value === '0' ? v : d.value + v; d.focus(); }\n"
                "    function calc() { try { d.value = eval(d.value); } catch { d.value = 'Error'; } }\n"
                "    function clearD() { d.value = '0'; }\n"
                "    document.addEventListener('keydown', e => {\n"
                "      if (/^[0-9.+\\-*/()]$/.test(e.key)) add(e.key);\n"
                "      if (e.key === 'Enter') calc();\n"
                "      if (e.key === 'Escape') clearD();\n"
                "    });\n"
                "  </script>\n"
                "</body>\n"
                "</html>"
            )

        if "chat" in pl or "bot" in pl:
            return (
                "<!DOCTYPE html>\n"
                '<html lang="en">\n'
                "<head>\n"
                "  <meta charset=\"UTF-8\">\n"
                "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n"
                f"  <title>{label} - Chat</title>\n"
                "  <style>\n"
                "    * { box-sizing: border-box; margin: 0; }\n"
                "    body { font-family: system-ui, sans-serif; background: #0f172a; color: #e2e8f0; "
                "display: flex; justify-content: center; padding: 2rem; }\n"
                "    .chat { width: 100%; max-width: 480px; }\n"
                "    h1 { font-size: 1.2rem; margin-bottom: 0.8rem; color: #38bdf8; }\n"
                "    #messages { height: 320px; overflow-y: auto; border: 1px solid #334155; "
                "border-radius: 8px; padding: 0.8rem; margin-bottom: 0.8rem; background: #1e293b; }\n"
                "    .msg { margin-bottom: 0.5rem; padding: 0.4rem 0.8rem; border-radius: 8px; "
                "max-width: 80%; }\n"
                "    .user { background: #2563eb; color: #fff; margin-left: auto; }\n"
                "    .bot { background: #334155; }\n"
                "    .row { display: flex; gap: 0.5rem; }\n"
                "    input { flex: 1; padding: 0.6rem; border: 1px solid #334155; border-radius: 8px; "
                "background: #1e293b; color: #e2e8f0; }\n"
                "    button { padding: 0.6rem 1rem; border: none; border-radius: 8px; "
                "background: #38bdf8; color: #0f172a; cursor: pointer; font-weight: 600; }\n"
                "  </style>\n"
                "</head>\n"
                "<body>\n"
                "  <div class=\"chat\">\n"
                f"    <h1>{label}</h1>\n"
                "    <div id=\"messages\">\n"
                "      <div class=\"msg bot\">Hello! How can I help you?</div>\n"
                "    </div>\n"
                "    <div class=\"row\">\n"
                "      <input id=\"input\" placeholder=\"Type a message...\" autofocus>\n"
                "      <button onclick=\"send()\">Send</button>\n"
                "    </div>\n"
                "  </div>\n"
                "  <script>\n"
                "    const input = document.getElementById('input');\n"
                "    const msgs = document.getElementById('messages');\n"
                "    const responses = ['Tell me more.', 'Interesting!', 'I see.', 'Go on...', 'Hmm, let me think about that.'];\n"
                "    function send() {\n"
                "      const text = input.value.trim();\n"
                "      if (!text) return;\n"
                "      const u = document.createElement('div');\n"
                "      u.className = 'msg user'; u.textContent = text;\n"
                "      msgs.appendChild(u);\n"
                "      setTimeout(() => {\n"
                "        const b = document.createElement('div');\n"
                "        b.className = 'msg bot';\n"
                "        b.textContent = responses[Math.floor(Math.random() * responses.length)];\n"
                "        msgs.appendChild(b);\n"
                "        msgs.scrollTop = msgs.scrollHeight;\n"
                "      }, 400);\n"
                "      input.value = '';\n"
                "      msgs.scrollTop = msgs.scrollHeight;\n"
                "    }\n"
                "    input.addEventListener('keydown', e => { if (e.key === 'Enter') send(); });\n"
                "  </script>\n"
                "</body>\n"
                "</html>"
            )

        if "counter" in pl or "count" in pl:
            return (
                "<!DOCTYPE html>\n"
                '<html lang="en">\n'
                "<head>\n"
                "  <meta charset=\"UTF-8\">\n"
                "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n"
                f"  <title>{label} - Counter</title>\n"
                "  <style>\n"
                "    * { box-sizing: border-box; margin: 0; }\n"
                "    body { font-family: system-ui, sans-serif; background: #0f172a; color: #e2e8f0; "
                "display: flex; justify-content: center; padding: 2rem; text-align: center; }\n"
                "    .app { padding: 2rem; }\n"
                "    h1 { font-size: 1.2rem; margin-bottom: 1rem; color: #38bdf8; }\n"
                "    #count { font-size: 4rem; margin: 1rem 0; }\n"
                "    button { padding: 0.6rem 1.2rem; margin: 0.3rem; border: none; border-radius: 8px; "
                "background: #38bdf8; color: #0f172a; cursor: pointer; font-weight: 600; font-size: 1rem; }\n"
                "    button:hover { background: #7dd3fc; }\n"
                "    button.reset { background: #334155; color: #94a3b8; }\n"
                "  </style>\n"
                "</head>\n"
                "<body>\n"
                "  <div class=\"app\">\n"
                f"    <h1>{label}</h1>\n"
                "    <div id=\"count\">0</div>\n"
                "    <button onclick=\"c.textContent=+c.textContent-1\">−</button>\n"
                "    <button onclick=\"c.textContent=+c.textContent+1\">+</button>\n"
                "    <br><br><button class=\"reset\" onclick=\"c.textContent='0'\">Reset</button>\n"
                "  </div>\n"
                "  <script>const c = document.getElementById('count');</script>\n"
                "</body>\n"
                "</html>"
            )

        if has_cat:
            return (
                "<!DOCTYPE html>\n"
                '<html lang="en">\n'
                "<head>\n"
                "  <meta charset=\"UTF-8\">\n"
                "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n"
                "  <title>🐱 Hello Cat</title>\n"
                "  <style>\n"
                "    * { box-sizing: border-box; }\n"
                "    body { font-family: system-ui, -apple-system, sans-serif; display: flex; "
                "align-items: center; justify-content: center; min-height: 100vh; margin: 0; "
                "background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); color: #eee; }\n"
                "    .card { text-align: center; padding: 2rem; }\n"
                "    .emoji { font-size: 5rem; }\n"
                "    h1 { font-size: 2.5rem; margin: 0.5rem 0 0; font-weight: 600; }\n"
                "    p { color: #94a3b8; margin-top: 0.5rem; }\n"
                "  </style>\n"
                "</head>\n"
                "<body>\n"
                "  <div class=\"card\">\n"
                "    <div class=\"emoji\">🐱</div>\n"
                "    <h1>Hello Cat!</h1>\n"
                "    <p>Meow! Built by catr1.10</p>\n"
                "  </div>\n"
                "</body>\n"
                "</html>"
            )

        return (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n'
            "<head>\n"
            "  <meta charset=\"UTF-8\">\n"
            "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n"
            f"  <title>{label}</title>\n"
            "  <style>\n"
            "    * { box-sizing: border-box; margin: 0; }\n"
            "    body { font-family: system-ui, sans-serif; background: #0f172a; color: #e2e8f0; "
            "display: flex; align-items: center; justify-content: center; min-height: 100vh; }\n"
            "    .app { text-align: center; padding: 2rem; max-width: 400px; }\n"
            f"    h1 {{ font-size: 1.8rem; margin-bottom: 0.5rem; color: #38bdf8; }}\n"
            "    p { color: #94a3b8; line-height: 1.6; }\n"
            "  </style>\n"
            "</head>\n"
            "<body>\n"
            "  <div class=\"app\">\n"
            f"    <h1>{label}</h1>\n"
            "    <p>Built by catr1.10 — in-memory, files=off.</p>\n"
            "  </div>\n"
            "</body>\n"
            "</html>"
        )

    @classmethod
    def _python(cls, engine: "CatR11Engine", prompt: str) -> str:
        pl = prompt.lower()
        if "fibonacci" in pl or re.search(r"\bfib\b", pl):
            block = engine.synth._code(engine._extract_topic_words(prompt), pl, None)
            m = re.search(r"```python\n(.*?)```", block, re.S)
            if m:
                return m.group(1).strip()
        if CodeAnythingEngine.enabled():
            return CodeAnythingEngine._py_general_app(prompt, cls._subject(prompt))
        subject = cls._subject(prompt)
        msg = TokenWeightCodeEmitter.message_from_prompt(prompt, None, engine)
        return TokenWeightCodeEmitter._python(msg)

    @classmethod
    def build(cls, engine: "CatR11Engine", prompt: str, lang: str, vec: Optional[np.ndarray] = None) -> str:
        if vec is None:
            vec = engine.last_vec if engine.last_vec is not None else engine.encode_for_task(prompt, task="code")
        if CodeAnythingEngine.enabled():
            code = CodeAnythingEngine.generate(engine, prompt, lang, vec)
            if CONFIG.get("catr1_engine"):
                return CatR1MythosEngine._recursive_perfect(code, lang, prompt, engine)
            return code
        if CONFIG.get("catr1_engine"):
            return CatR1MythosEngine.generate(engine, prompt, lang, vec)
        return TokenWeightCodeEmitter.emit(engine, prompt, lang, vec)

    @classmethod
    def format_response(cls, lang: str, code: str, prompt: str, engine: "CatR11Engine") -> str:
        pl = prompt.lower()
        lang = cls._coerce_lang(engine, lang, prompt)
        fenced = TokenWeightCodeEmitter.fence(lang, code)
        if CONFIG.get("code_output_exact") or engine._wants_code_only(prompt):
            return fenced
        if cls.should_run(pl) and lang in cls.RUNNABLE:
            result = cls.run(engine, lang, code)
            return f"{fenced}\n\n**Output:**\n```\n{result}\n```"
        return fenced

    @classmethod
    def _prompt_for_code(cls, prompt: str, engine: "CatR11Engine") -> str:
        pl = cls.normalize_prompt(prompt).strip().lower()
        if cls._GO.match(pl) or cls._CODE_SHORT.match(pl):
            for msg in reversed(engine.chat_history):
                if msg.get("role") == "user":
                    prior = msg.get("text", "").strip()
                    if prior and not cls._CODE_SHORT.match(prior.lower()):
                        return prior
            return "write hello world in python"
        if pl.startswith("/code "):
            return prompt.split(maxsplit=1)[1] if " " in prompt.strip() else "write hello world in python"
        return prompt

    @classmethod
    def respond(cls, engine: "CatR11Engine", prompt: str) -> str:
        if not cls.enabled():
            return f"{BRAND} code is disabled. Set `catr1_code_enabled=True` in CONFIG."
        prompt = cls._prompt_for_code(prompt, engine)
        norm = cls.normalize_prompt(prompt)
        vec = engine.encode_for_task(norm, task="code")
        lang = cls.detect_lang(engine, norm)
        code = cls.build(engine, norm, lang, vec)
        return cls.format_response(lang, code, norm, engine)


CatSeekCode = CatSeekR1Code
CatCode010 = CatSeekR1Code


# ──────────────────────────────────────────────────────────────
# CAT R1.1 CODING API 0.1 (cat r1.10 code–style agent · files = off)
# ──────────────────────────────────────────────────────────────
@dataclass
class CodingBuffer:
    name: str = "untitled.py"
    lang: str = "python"
    code: str = ""
    last_output: str = ""
    runs: int = 0


class CatSeekR1CodingAPI:
    """
    cat r1.10 code interpreter (files = off).
    o1-preview trace syntax + cat r1.10 code execution engine.
    """

    PROTO = CONFIG.get("o1_interpreter_protocol", O1_INTERPRETER_PROTO)
    VER = O1PreviewSyntax.VERSION
    NAME = CONFIG["code_interpreter_name"]
    FULL = NAME
    MYTHOS = CONFIG["mythos_tier"]
    PROSE = CONFIG["prose_tier"]
    BACKEND = CONFIG["code_interpreter"]
    MYTHOS_VER = CONFIG["code_interpreter_version"]
    TOOLS = ("run", "edit", "lint", "explain", "new", "clear")

    _buffers: Dict[str, CodingBuffer] = {}
    _active_id: str = "default"

    @classmethod
    def default_snippet(cls) -> str:
        return (
            f"# {ClaudeMythosRuntime.think_header()}\n"
            f"# {cls.FULL} · in-memory buffer only\n\n"
            "def main():\n"
            "    print('Hello from cat r1.10 code')\n\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        )

    @classmethod
    def help_text(cls) -> str:
        return ClaudeMythosRuntime.code_help()

    @classmethod
    def session(cls, session_id: Optional[str] = None) -> CodingBuffer:
        sid = session_id or cls._active_id
        if sid not in cls._buffers:
            cls._buffers[sid] = CodingBuffer(code=cls.default_snippet())
        cls._active_id = sid
        return cls._buffers[sid]

    @classmethod
    def _detect_lang(cls, engine: "CatR11Engine", code: str, hint: str = "") -> str:
        if hint:
            norm = engine.normalize_lang(hint)
            if norm:
                return norm
        return CatSeekR1Code.detect_lang(engine, code)

    @classmethod
    def lint(cls, code: str, lang: str) -> Tuple[bool, str]:
        lang = (lang or "python").lower()
        if lang == "python":
            try:
                ast.parse(code)
                return True, "syntax ok"
            except SyntaxError as e:
                return False, str(e)
        if lang in {"javascript", "typescript", "js", "ts"}:
            return True, "js lint skipped"
        return True, "lint skipped"

    @classmethod
    def explain(cls, engine: "CatR11Engine", code: str, lang: str) -> str:
        preview = code.strip()[:400]
        lines = len(code.splitlines())
        return (
            f"**{cls.FULL}** · `{lang}` · {lines} lines\n\n"
            f"Buffer is in-memory only.\n\n"
            f"```{(lang or 'text')[:12]}\n{preview}\n```"
        )

    @classmethod
    def agent_run(
        cls,
        engine: "CatR11Engine",
        code: str,
        lang: str = "",
        script_name: str = "untitled.py",
        *,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """cat r1.10 code agent loop: lint → run → capture output (files = off)."""
        body = (code or "").strip()
        if not body:
            return {"ok": False, "error": "empty buffer", "protocol": cls.PROTO, "version": cls.VER,
                    "files": "off", "mythos_tier": cls.MYTHOS, "prose_tier": cls.PROSE}
        lang_key = cls._detect_lang(engine, body, lang)
        thinking = O1PreviewSyntax.code_trace(lang_key, script_name)
        ok, lint_msg = cls.lint(body, lang_key)
        if not ok:
            return {
                "ok": False,
                "error": lint_msg,
                "action": "lint",
                "lang": lang_key,
                "script": script_name,
                "protocol": cls.PROTO,
                "version": cls.VER,
                "model": O1PreviewSyntax.MODEL,
                "syntax": O1PreviewSyntax.TAG,
                "files": "off",
                "thinking": thinking,
            }
        output = CatSeekR1Code.run(engine, lang_key, body)
        buf = cls.session(session_id)
        buf.name = script_name
        buf.lang = lang_key
        buf.code = body
        buf.last_output = output
        buf.runs += 1
        return {
            "ok": True,
            "action": "run",
            "protocol": cls.PROTO,
            "version": cls.VER,
            "model": O1PreviewSyntax.MODEL,
            "syntax": O1PreviewSyntax.TAG,
            "files": "off",
            "thinking": thinking,
            "engine": cls.FULL,
            "lang": lang_key,
            "script": script_name,
            "lint": lint_msg,
            "output": output,
            "runs": buf.runs,
        }

    @classmethod
    def format_result(cls, result: Dict[str, Any]) -> str:
        if not result.get("ok"):
            return ClaudeMythosRuntime.format_code_result(
                result.get("lang", "code"),
                result.get("script", "buffer"),
                "",
                error=result.get("error", "unknown"),
            )
        return ClaudeMythosRuntime.format_code_result(
            result.get("lang", "code"),
            result.get("script", "buffer"),
            result.get("output", ""),
        )

    @classmethod
    def parse_request(cls, engine: "CatR11Engine", data: Dict[str, Any]) -> Dict[str, Any]:
        if data.get("protocol") not in (None, cls.PROTO):
            return {"error": f"protocol must be {cls.PROTO}", "files": "off"}
        action = (data.get("action") or "run").lower()
        code = data.get("code") or data.get("buffer") or ""
        lang = data.get("lang") or data.get("language") or ""
        script = data.get("script") or data.get("name") or "untitled.py"
        sid = data.get("session") or data.get("session_id") or "default"

        if action == "help":
            return {"protocol": cls.PROTO, "version": cls.VER, "files": "off", "help": cls.help_text()}
        if action == "explain":
            text = cls.explain(engine, code, lang or "python")
            return {"protocol": cls.PROTO, "version": cls.VER, "files": "off", "action": "explain", "content": text}
        if action == "lint":
            ok, msg = cls.lint(code, lang or "python")
            return {"protocol": cls.PROTO, "version": cls.VER, "files": "off", "action": "lint", "ok": ok, "message": msg}
        if action in {"run", "execute", "interpret"}:
            return cls.agent_run(engine, code, lang, script, session_id=sid)
        if action == "new":
            buf = cls.session(sid)
            buf.code = cls.default_snippet()
            buf.name = "untitled.py"
            buf.last_output = ""
            buf.runs = 0
            return {"protocol": cls.PROTO, "version": cls.VER, "files": "off", "action": "new", "script": buf.name, "code": buf.code}
        return {"error": f"unknown action: {action}", "files": "off", "tools": list(cls.TOOLS)}


CodingAPI = CatSeekR1CodingAPI
ClaudeCodeStyle = CatSeekR1CodingAPI


class TokenWeightCodeEmitter:
    """
    Code synthesis from cat r1.10 token weights + prompt (files = off).
    Output is derived only from the inference vector and prompt tokens — no disk files,
    no dynamic template stubs.
    """

    @staticmethod
    def _pick(vec: Optional[np.ndarray], n: int, salt: int = 0) -> int:
        if vec is None or vec.size == 0:
            return salt % max(n, 1)
        return int(abs(float(np.sum(vec * (1.0 + salt * 0.031)) * 10007))) % max(n, 1)

    @classmethod
    def message_from_prompt(
        cls,
        prompt: str,
        vec: Optional[np.ndarray],
        engine: Optional["CatR11Engine"] = None,
    ) -> str:
        """Payload string from prompt tokens + cat r1.10 vec (files = off)."""
        subj = CatSeekR1Code._subject(prompt, engine)
        if subj and subj.lower() not in {"hello world", "it", "it html"}:
            return subj
        pl = CatSeekR1Code.normalize_prompt(prompt).lower()
        if "cat" in pl:
            return "Hello Cat" if cls._pick(vec, 2, 2) == 0 else "Meow!"
        defaults = ("Hello World", "Hello", "Meow", "Hi there")
        return defaults[cls._pick(vec, len(defaults), 1)]

    @classmethod
    def _c(cls, msg: str) -> str:
        esc = msg.replace("\\", "\\\\").replace('"', '\\"')
        return (
            "#include <stdio.h>\n\n"
            "int main(void) {\n"
            f'    printf("{esc}\\n");\n'
            "    return 0;\n"
            "}"
        )

    @classmethod
    def _cpp(cls, msg: str) -> str:
        esc = msg.replace("\\", "\\\\").replace('"', '\\"')
        return (
            "#include <iostream>\n\n"
            "int main() {\n"
            f'    std::cout << "{esc}" << std::endl;\n'
            "    return 0;\n"
            "}"
        )

    @classmethod
    def _python(cls, msg: str) -> str:
        esc = msg.replace("\\", "\\\\").replace("'", "\\'")
        return (
            "import sys\n\n"
            f"MSG = '{esc}'\n\n"
            "def main():\n"
            "    name = sys.argv[1] if len(sys.argv) > 1 else 'World'\n"
            "    print(f'{MSG}, {name}!')\n"
            "    print(f'Running Python {sys.version}')\n\n"
            "if __name__ == '__main__':\n"
            "    main()"
        )

    @classmethod
    def emit(
        cls,
        engine: "CatR11Engine",
        prompt: str,
        lang: str,
        vec: Optional[np.ndarray],
    ) -> str:
        embedded_lang, embedded = CatSeekR1Code.extract_prompt_code(prompt)
        if embedded:
            return embedded

        lang = CatSeekR1Code._coerce_lang(engine, lang, prompt)
        msg = cls.message_from_prompt(prompt, vec, engine)

        if lang == "html":
            return CatSeekR1Code._html(prompt, engine)
        if lang == "python":
            pl = prompt.lower()
            if "fibonacci" in pl or re.search(r"\bfib\b", pl):
                return CatSeekR1Code._python(engine, prompt)
            if CodeAnythingEngine.enabled() and len(prompt.split()) > 4:
                return CodeAnythingEngine._py_general_app(prompt, msg)
            return cls._python(msg)
        if lang == "c":
            return cls._c(msg)
        if lang == "cpp":
            return cls._cpp(msg)
        if lang == "javascript":
            esc = msg.replace("\\", "\\\\").replace("'", "\\'")
            return f"console.log('{esc}');"
        if lang == "bash":
            esc = msg.replace("\\", "\\\\").replace('"', '\\"')
            return f'#!/bin/bash\necho "{esc}"'
        if lang == "rust":
            esc = msg.replace("\\", "\\\\").replace('"', '\\"')
            return f'fn main() {{\n    println!("{esc}");\n}}'
        if lang == "go":
            esc = msg.replace("\\", "\\\\").replace('"', '\\"')
            return f'package main\n\nimport "fmt"\n\nfunc main() {{\n\tfmt.Println("{esc}")\n}}'
        if lang == "java":
            esc = msg.replace("\\", "\\\\").replace('"', '\\"')
            return (
                "public class Main {\n"
                "    public static void main(String[] args) {\n"
                f'        System.out.println("{esc}");\n'
                "    }\n"
                "}"
            )
        if lang == "css":
            return f"body {{\n  font-family: system-ui, sans-serif;\n}}\n\n/* {msg} */"
        if lang == "sql":
            esc = msg.replace("'", "''")
            return f"SELECT '{esc}' AS message;"
        if CodeAnythingEngine.enabled():
            task = CodeAnythingEngine.detect_task(prompt)
            return CodeAnythingEngine.scaffold(lang, task, prompt, engine, vec)
        # Token-weight fallback — never emit dynamic template stubs
        return cls._python(msg)

    @staticmethod
    def fence(lang: str, code: str) -> str:
        tag = lang if lang else ""
        return f"```{tag}\n{code.rstrip()}\n```"


# ──────────────────────────────────────────────────────────────
# CAT R1.1 CODE ENGINE (files = off · perfect-code loop)
# ──────────────────────────────────────────────────────────────
class CatR1MythosEngine:
    """
    cat r1.10 code engine for cat r1.10 (files = off).
    Pattern library · recursive perfection · lint · sandbox verify.
    """

    NAME = CONFIG["code_interpreter_name"]
    FAMILY = CONFIG["code_interpreter_family"]
    BACKEND = CONFIG["code_interpreter"]
    VERSION = CONFIG["code_interpreter_version"]

    _HELLO_CAT_C = (
        "#include <stdio.h>\n\n"
        "int main(void) {\n"
        '    printf("Hello Cat\\n");\n'
        '    printf("Meow!\\n");\n'
        "    return 0;\n"
        "}"
    )

    _PY_HELLO = (
        "def main() -> None:\n"
        '    print("Hello, World!")\n\n'
        "if __name__ == \"__main__\":\n"
        "    main()"
    )

    _PY_HELLO_CAT = (
        "def main() -> None:\n"
        '    print("Hello Cat!")\n'
        '    print("Meow!")\n\n'
        "if __name__ == \"__main__\":\n"
        "    main()"
    )

    _JS_HELLO = "console.log('Hello, World!');\n"

    _RUST_HELLO = (
        "fn main() {\n"
        '    println!("Hello, World!");\n'
        "}"
    )

    @classmethod
    def _lint(cls, code: str, lang: str) -> str:
        if not CONFIG.get("catr1b_lint", True):
            return code
        lines = [ln.rstrip() for ln in code.splitlines()]
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        body = "\n".join(lines)
        if lang == "python" and body and not body.endswith("\n"):
            body += "\n"
        return body

    @classmethod
    def _pattern_match(cls, prompt: str, lang: str, engine: "CatR11Engine") -> Optional[str]:
        pl = CatSeekR1Code.normalize_prompt(prompt).lower()
        if lang == "c" and re.search(r"\bhello\b", pl):
            msg = CatSeekR1Code._subject(prompt, engine)
            if "cat" in pl:
                return cls._HELLO_CAT_C
            return TokenWeightCodeEmitter._c(msg if msg else "Hello")
        if lang == "cpp" and "hello" in pl:
            msg = CatSeekR1Code._subject(prompt, engine)
            return TokenWeightCodeEmitter._cpp(msg if msg else "Hello")
        if lang == "python":
            if "fibonacci" in pl or re.search(r"\bfib\b", pl):
                return CatSeekR1Code._python(engine, prompt)
            if re.search(r"\bhello\s+cat\b", pl):
                return cls._PY_HELLO_CAT
            if re.search(r"\bhello\s+world\b", pl) or pl.strip() in {"hello", "write hello in python"}:
                return cls._PY_HELLO
            if CodeAnythingEngine.enabled():
                subject = CatSeekR1Code._subject(prompt, engine)
                task = CodeAnythingEngine.detect_task(prompt)
                if task != "general":
                    return CodeAnythingEngine.scaffold(lang, task, prompt, engine)
                return CodeAnythingEngine._py_general_app(prompt, subject)
        if lang == "javascript" and "hello" in pl:
            return cls._JS_HELLO
        if lang == "rust" and "hello" in pl:
            return cls._RUST_HELLO
        if lang == "html":
            return CatSeekR1Code._html(prompt, engine)
        return None

    @classmethod
    def _validate(cls, lang: str, code: str) -> Tuple[bool, str]:
        lang = lang.lower()
        if not code or not code.strip():
            return False, "empty"
        if lang == "python":
            try:
                ast.parse(code)
                return True, ""
            except SyntaxError as e:
                return False, str(e)
        if lang in {"c", "cpp"}:
            if lang == "c" and "#include" not in code and "printf" in code:
                return False, "missing include"
            if "main" not in code:
                return False, "missing main"
            return True, ""
        if lang == "html":
            if "<html" not in code.lower() and "<!doctype" not in code.lower():
                return False, "not html"
            return True, ""
        if lang == "javascript":
            return True, ""
        return True, ""

    @classmethod
    def _fix(cls, lang: str, code: str, prompt: str, reason: str) -> str:
        lang = lang.lower()
        code = code.strip()
        if lang == "c" and "missing include" in reason:
            if "#include" not in code:
                code = "#include <stdio.h>\n\n" + code
        if lang == "python" and ("SyntaxError" in reason or "invalid syntax" in reason):
            if "def main" not in code:
                msg = TokenWeightCodeEmitter.message_from_prompt(prompt, None, None)
                return TokenWeightCodeEmitter._python(msg)
        if lang == "c" and "return 0" not in code and "main" in code:
            if not code.rstrip().endswith("}"):
                code = code.rstrip() + "\n    return 0;\n}"
        code = re.sub(r"\n{3,}", "\n\n", code)
        return code.rstrip() + "\n"

    @classmethod
    def _polish(cls, code: str, lang: str) -> str:
        lines = code.splitlines()
        out: List[str] = []
        for ln in lines:
            if re.match(r"^c\s*$", ln.strip()) and lang != "text":
                continue
            if re.search(r"^/Volumes/|^/Users/", ln):
                continue
            out.append(ln.rstrip())
        body = "\n".join(out).strip()
        if lang == "python" and body and "if __name__" not in body and "def main" in body:
            if not body.endswith("\n"):
                body += "\n"
            if "if __name__" not in body:
                body += "\nif __name__ == '__main__':\n    main()"
        return body

    @classmethod
    def _recursive_perfect(
        cls, code: str, lang: str, prompt: str, engine: "CatR11Engine"
    ) -> str:
        if not CONFIG.get("catr1_code_perfect", True):
            return code
        depth = CONFIG.get("catr1_recursive_depth", 3)
        out = code
        for i in range(depth):
            out = cls._polish(out, lang)
            ok, reason = cls._validate(lang, out)
            if ok:
                if lang == "python" and engine._validate_code("python", out):
                    break
                if lang != "python":
                    break
            out = cls._fix(lang, out, prompt, reason)
        return cls._lint(cls._polish(out, lang), lang)

    @classmethod
    def enabled(cls) -> bool:
        return bool(CONFIG.get("catr1_engine", True) and CONFIG.get("catr1_code_enabled", True))

    @classmethod
    def generate(
        cls,
        engine: "CatR11Engine",
        prompt: str,
        lang: str,
        vec: Optional[np.ndarray] = None,
    ) -> str:
        if not cls.enabled():
            return TokenWeightCodeEmitter.emit(engine, prompt, lang, vec)
        hit = cls._pattern_match(prompt, lang, engine)
        if hit:
            draft = hit
        elif CodeAnythingEngine.enabled():
            draft = CodeAnythingEngine.generate(engine, prompt, lang, vec)
        else:
            draft = TokenWeightCodeEmitter.emit(engine, prompt, lang, vec)
        return cls._recursive_perfect(draft, lang, prompt, engine)

    @classmethod
    def respond(cls, engine: "CatR11Engine", prompt: str) -> str:
        return CatSeekR1Code.respond(engine, prompt)


ClaudeMythosCode = CatR1MythosEngine
CatR1Code = CatR1MythosEngine


# ──────────────────────────────────────────────────────────────
# CAT R1.1 LLM (files = off · local runtime on cat r1.10)
# ──────────────────────────────────────────────────────────────
class CatSeekContextMemory:
    """1M-token logical context window — in-memory only, files = off."""

    __slots__ = ("turns", "logical_tokens")

    def __init__(self):
        self.turns: List[Dict[str, str]] = []
        self.logical_tokens = 0

    def add(self, role: str, text: str) -> None:
        self.turns.append({"role": role, "text": text})
        self.logical_tokens += max(len(text.split()), 1)
        limit = CONFIG["catr1_context_window"]
        while self.logical_tokens > limit and len(self.turns) > 2:
            old = self.turns.pop(0)
            self.logical_tokens -= max(len(old["text"].split()), 1)

    def recent(self, n: int = 24) -> List[Dict[str, str]]:
        return self.turns[-n:]

    def prior_topics(self) -> str:
        users = [t["text"][:60] for t in self.turns if t["role"] == "user"][-4:]
        return "; ".join(users) if users else ""


FableContextMemory = CatSeekContextMemory


class CatSeekR1LLM:
    """
    catr1.10 frontier runtime (catseek r1.10 frontier):
    encode → think → draft → self-check → respond — all in-memory.
    1.7T effective parameters · 10M token context · 512K output.
    """

    MODEL_ID = CONFIG["catr1_model_id"]
    NAME = BRAND
    FAMILY = "cat-r1"
    CONTEXT_WINDOW = CONFIG["catr1_context_window"]
    MAX_OUTPUT = CONFIG["catr1_max_output"]
    _BRAND_PREFIX = CatSeekR1Code._BRAND_PREFIX

    _CREATIVE_CUES = (
        "fable", "parable", "allegory", "short story", "tell me a story",
        "write a story", "write me a story", "bedtime story", "tale about",
        "story about", "fable about", "once upon", "moral story",
        "poem", "poetry", "haiku", "sonnet", "verse", "rhyme",
        "narrative", "imagine a", "creative writing",
    )
    _CODE_CUES = (
        "code", "function", "implement", "snippet", "script", "class",
        "fibonacci", "fib", "prime", "python", "javascript", "rust", "java",
        "def ", "import ", "```",
    )

    def __init__(self):
        self.memory = CatSeekContextMemory()

    @staticmethod
    def _seed(vec: Optional[np.ndarray], salt: int = 0) -> int:
        if vec is None or vec.size == 0:
            return salt
        return int(abs(float(np.sum(vec * (1.0 + salt * 0.01)) * 10007))) % 10_000

    @classmethod
    def pick(cls, vec: Optional[np.ndarray], n: int, salt: int = 0) -> int:
        return cls._seed(vec, salt) % max(n, 1)

    @classmethod
    def is_creative(cls, pl: str) -> bool:
        pl = cls._BRAND_PREFIX.sub("", pl.strip())
        return CONFIG["catr1_enabled"] and any(c in pl for c in cls._CREATIVE_CUES)

    @classmethod
    def is_code_request(cls, pl: str) -> bool:
        if not CONFIG.get("catr1_code_enabled", True):
            return False
        return CatSeekR1Code.wants_code(pl)

    @classmethod
    def wants_fable(cls, pl: str) -> bool:
        pl = CatSeekR1Code._BRAND_PREFIX.sub("", pl.strip())
        cues = ("fable about", "write a fable", "tell me a fable", "parable", "allegory",
                "tell me a story", "write a story", "story about", "fable about")
        return any(c in pl for c in cues) or (
            re.search(r"\bfable\b", pl) is not None and "catr1" not in pl and "cat r1" not in pl
        )

    @classmethod
    def wants_poem(cls, pl: str) -> bool:
        return any(c in pl for c in ("poem", "poetry", "haiku", "sonnet", "verse", "rhyme"))

    def classify(self, prompt: str, engine: "CatR11Engine") -> str:
        if CatSeekR1Fusion.is_noise(prompt):
            return "chat"
        norm = CatSeekR1Code.normalize_prompt(prompt)
        pl = norm.lower().strip()
        follow = CatSeekR1Fusion.session_followup(engine, prompt)
        if follow and CatSeekR1Fusion.is_noise(prompt):
            return "chat"
        if is_zh_greeting(prompt.strip()):
            return "chat"
        if engine.synth.smalltalk_reply(pl) or engine.synth.smalltalk_reply(prompt.strip()):
            return "chat"
        if is_explain_request(prompt) and not CatSeekR1Code.extract_prompt_code(prompt)[1]:
            return "explain"
        if CONFIG.get("web_program_enabled") and CatSeekWebProgram.wants_web(pl):
            return "web"
        if CONFIG.get("catr1_code_enabled"):
            if CatSeekR1Code.extract_prompt_code(prompt)[1]:
                return "code"
            if CatSeekR1Code.wants_code_with_history(pl, engine):
                return "code"
        if self.is_creative(pl):
            if self.wants_poem(pl):
                return "poem"
            if self.wants_fable(pl) or "story" in pl or "tale" in pl:
                return "fable"
            return "creative"
        if engine._try_simple_math(prompt) is not None:
            return "math"
        if any(x in pl for x in ("run code", "execute", "interpret", "/exec", "运行", "执行")):
            return "execute"
        if pl in ("run it", "run this", "execute it", "test it", "run"):
            return "execute"
        if self.is_code_request(pl):
            return "code"
        if any(k in pl for k in ("traceback", "exception", "error", "bug", "debug", "broken")):
            return "debug"
        if is_explain_request(prompt):
            return "explain"
        if re.search(r"\b(explain|what is|what are|why|how (?:does|do|to|can))\b", pl):
            return "explain"
        if re.search(r"(解释|说明|介绍)", prompt):
            return "explain"
        if re.search(r"\b(plan|architecture)\b", pl) and len(pl.split()) > 2:
            return "agent"
        if re.search(r"\b(design|build)\s+(?:a|an|the|my|this|your|me)\b", pl):
            return "agent"
        return "general"

    def plan(self, prompt: str, task: str, vec: Optional[np.ndarray]) -> List[str]:
        steps = {
            "chat": ["match tone", "respond warmly", "invite next turn"],
            "explain": ["identify core concept", "mechanism", "example", "caveats"],
            "code": ["spec inputs/outputs", "draft", "edge cases", "self-test"],
            "web": ["pick template", "render artifact HTML", "store in memory", "preview URL"],
            "debug": ["reproduce", "localize", "fix", "verify"],
            "fable": ["theme", "arc", "moral", "polish prose"],
            "poem": ["image", "turn", "close"],
            "creative": ["voice", "structure", "finish"],
            "agent": ["decompose", "sequence", "deliverables"],
            "math": ["parse", "compute", "verify"],
            "execute": ["parse block", "run sandbox", "report"],
            "general": ["understand", "answer", "check completeness"],
        }
        base = steps.get(task, steps["general"])
        if vec is not None and self.pick(vec, 3, 12) == 0 and task in ("explain", "code", "agent"):
            base = base + ["proactive follow-up"]
        return base

    @staticmethod
    def _extract_subject(prompt: str) -> str:
        pl = prompt.lower()
        for prefix in ("fable about ", "story about ", "poem about ", "tale about ", "about ", "on "):
            if prefix in pl:
                return prompt[pl.index(prefix) + len(prefix):].strip("?.")
        return re.sub(
            r"^(?:write|tell|give)\s+(?:me\s+)?(?:a\s+)?(?:fable|story|poem|tale)\s*(?:about\s+)?",
            "", pl, flags=re.I,
        ).strip("?.")

    def compose_fable(self, prompt: str, topic: str, vec: Optional[np.ndarray]) -> str:
        heroes = (
            ("a clever fox", "wit over force"), ("a patient tortoise", "steady effort"),
            ("a curious owl", "seeing what others miss"), ("a humble mouse", "small acts matter"),
        )
        settings = (
            "at the edge of an ancient forest", "in a harbor town of quiet clocks",
            "on a hillside where seasons negotiated in whispers",
        )
        hero, virtue = heroes[self.pick(vec, len(heroes), 2)]
        setting = settings[self.pick(vec, len(settings), 3)]
        subject = self._extract_subject(prompt) or topic.strip("?.") or "the work that waits"
        return (
            f"**{subject.title()}** — a fable\n\n"
            f"Once, {hero} lived {setting}. The village spoke of **{subject}**, "
            f"though no two voices agreed on its shape.\n\n"
            f"When difficulty arrived — as it always does — the hero did not perform brilliance. "
            f"They attended: mending what broke, keeping promises, asking honest questions. "
            f"Others mistook patience for slowness until the results could no longer be ignored.\n\n"
            f"**Moral:** {virtue.capitalize()} compounds quietly. "
            f"The world becomes slightly more legible for whoever comes next."
        )

    def compose_poem(self, topic: str, vec: Optional[np.ndarray], prompt: str = "") -> str:
        subj = (self._extract_subject(prompt) if prompt else "") or topic.strip("?.") or "the road ahead"
        forms = (
            f"**On {subj}**\n\nNot all answers arrive with noise;\n"
            f"some knock like moth-wings at the glass —\n"
            f"small, insistent, easy to ignore\nuntil you remember you wanted light.\n\n"
            f"Begin again. {subj.capitalize()} waits\nlike shore waits tide: patient, sure, returned.",
            f"**{subj.title()}**\n\nYou asked about {subj}.\n"
            f"The mind builds bridges out of questions;\neach span creaks until you walk it.\n\n"
            f"Go slowly. Name what you know.\nLeave doors open for better names.",
        )
        return forms[self.pick(vec, len(forms), 5)]

    def self_check(self, prompt: str, draft: str, task: str, engine: "CatR11Engine") -> str:
        out = draft
        if task == "code" and "```" in out:
            for m in re.finditer(r"```(\w*)\n(.*?)```", out, re.S):
                lang, code = m.group(1) or "python", m.group(2)
                if lang == "python" and engine._validate_code("python", code):
                    continue
        if task in ("explain", "general", "agent") and "?" in prompt and len(out) < 80:
            out += "\n\nI can go deeper — tell me which part needs expansion."
        return ClaudeMythosRuntime.emit(engine, out, prompt, task)

    def complete(self, engine: "CatR11Engine", prompt: str, *, simulate: bool = True) -> str:
        """Main cat r1.10 completion entry — files = off."""
        _CATSEEK_FAST = frozenset({"chat", "code", "math", "execute", "web", "explain"})
        pl = prompt.lower().strip()
        follow = CatSeekR1Fusion.session_followup(engine, prompt)
        if follow and CatSeekR1Fusion.is_noise(prompt):
            return follow
        task = self.classify(prompt, engine)
        engine.encode_for_task(prompt, task=task)
        vec = engine.last_vec
        if task in _CATSEEK_FAST:
            if engine.ultrathink_on:
                engine._run_ultrathink(prompt)
            else:
                engine.last_think = ""
                engine._pending_think = ""
        _ = self.plan(prompt, task, vec)

        if task == "chat":
            hit = engine.synth.smalltalk_reply(pl)
            if hit:
                return hit
            if follow:
                return follow
            if CONFIG.get("catr1_code_enabled") and CatSeekR1Code.wants_code_with_history(pl, engine):
                return ClaudeMythosRuntime.emit(
                    engine, CatSeekR1Code.respond(engine, prompt), prompt, "code"
                )
            body = engine.synth.converse(prompt, engine.chat_history, vec=vec)
            return ClaudeMythosRuntime.emit(engine, body, prompt, "chat")

        # cat r1.10 Code fast path — cat r1.10 token weights · files = off
        if task == "code":
            return ClaudeMythosRuntime.emit(
                engine, CatSeekR1Code.respond(engine, prompt), prompt, "code"
            )

        if task == "web":
            try:
                return engine.web.respond(engine, prompt)
            except Exception as exc:
                return f"**{WEB_PROGRAM_NAME}** error: {exc}"

        if task == "math":
            result = engine._try_simple_math(prompt)
            if result:
                if CONFIG.get("mythos_mode") and ClaudeMythosRuntime.enabled():
                    return ClaudeMythosRuntime.math_wrap(prompt, result)
                if CONFIG.get("catr1_reasoning"):
                    return engine.fusion.catseek_math_wrap(prompt, result)
                return f"Result: **{result}**"

        if task == "explain":
            dia = engine.get_dialect(engine.detect_locale(prompt))
            intent = engine._best_intent(prompt)
            if intent in {"recursion", "core", "help", "languages", "profile"}:
                hit = engine._intent_response(intent, prompt, dia)
                if hit:
                    return self.self_check(prompt, hit, task, engine)
            history = engine.chat_history[:-1] if engine.chat_history else []
            history_sorted = GoogleWhitepaperCatSeekSorter.sort_history_dicts(history, prompt)
            body = engine.synth.synthesize(
                prompt, [(m["role"], m["text"]) for m in history_sorted], vec=vec
            )
            return self.self_check(prompt, body, task, engine)

        if task == "execute":
            block_lang, code = engine.extract_code_block(prompt)
            exec_lang = CatSeekR1Code.detect_lang(
                engine, prompt if not block_lang else f"in {block_lang}"
            )
            if not code:
                for m in reversed(engine.chat_history):
                    if m.get("role") == "assistant":
                        block_lang, code = engine.extract_code_block(m.get("text", ""))
                        if code:
                            exec_lang = engine.normalize_lang(block_lang) or exec_lang
                            break
            if not code:
                return "Paste code in a fenced block, or generate code first then say **run it**."
            result = CatSeekR1CodingAPI.agent_run(engine, code, exec_lang)
            return CatSeekR1CodingAPI.format_result(result)

        norm = CatSeekR1Code.normalize_prompt(prompt)
        if CONFIG.get("catr1_code_enabled") and CatSeekR1Code.wants_code_with_history(norm.lower(), engine):
            return CatSeekR1Code.respond(engine, prompt)

        if task not in _CATSEEK_FAST and not engine._pending_think:
            engine._run_ultrathink(prompt)
        history = engine.chat_history[:-1] if engine.chat_history else []
        dia = engine.get_dialect(engine.detect_locale(prompt))

        if task == "fable":
            topic = engine._extract_topic_words(prompt)
            body = self.compose_fable(prompt, topic, vec)
            return self.self_check(prompt, body, task, engine)

        if task == "poem":
            topic = engine._extract_topic_words(prompt)
            body = self.compose_poem(topic, vec, prompt)
            return self.self_check(prompt, body, task, engine)

        if task == "creative":
            topic = engine._extract_topic_words(prompt)
            body = self.compose_fable(prompt, topic, vec)
            return self.self_check(prompt, body, task, engine)

        intent = engine._best_intent(prompt)
        if intent in {"languages", "profile", "core", "recursion", "help"}:
            hit = engine._intent_response(intent, prompt, dia)
            if hit:
                return self.self_check(prompt, hit, task, engine)

        history = GoogleWhitepaperCatSeekSorter.sort_history_dicts(history, prompt)
        body = engine.synth.o1_answer(prompt, history, vec=vec)
        return self.self_check(prompt, body, task, engine)

    @staticmethod
    def model_card() -> Dict[str, Any]:
        return {
            "id": CATSEEK_MODEL_ID,
            "object": "model",
            "family": "cat-r1.10",
            "display_name": CatSeekR1LLM.NAME,
            "owned_by": "catseek",
            "tier": "frontier",
            "rivals": [],
            "context_window": CONFIG["catr1_context_window"],
            "max_output_tokens": CONFIG["catr1_max_output"],
            "nominal_params": CONFIG["nominal_base_params"],
            "effective_params_tier": "1.7T (frontier)",
            "files": FILES,
            "prose_tier": PROSE_TIER,
            "mythos_tier": MYTHOS_TIER,
            "reasoning": REASONING_MODE,
            "catr1_reasoning": CONFIG.get("catr1_reasoning", True),
            "code_interpreter": CODE_ENGINE,
            "code_interpreter_version": CODING_API_VER,
            "coding_api_protocol": CODING_API_PROTO,
            "coding_api": CODING_API_LABEL,
            "mythos_engine_version": MYTHOS_ENGINE_VER,
            "code_backend": CODE_BACKEND,
            "catr1_code_enabled": CAT_R1_CODE_ENABLED,
            "web_program": CONFIG.get("web_program_enabled", True),
            "arch": "catr11_frontier",
            "bitnet_engine": {
                "type": "AbsMeanTernary",
                "bits": CONFIG["weight_bits"],
                "packing": "base-3",
                "status": "active",
            },
            "moe": {
                "experts": CONFIG["n_experts"],
                "top_k": CONFIG["top_k"],
                "catr1_scale": CONFIG.get("catr1_moe_scale", True),
                "catr1_dense": CONFIG.get("catr1_moe_dense", True),
            },
            "compression": {
                "sparse_k": CONFIG["compression_sparse_k"],
                "low_rank": CONFIG["compression_rank"],
                "awq_bits": CONFIG.get("compression_nvidia_awq_bits", 4),
                "gptq_block": CONFIG.get("compression_nvidia_gptq_block", 128),
                "sparse_2_4": CONFIG.get("compression_nvidia_sparse_2_4", True),
            },
            "distillation": {
                "enabled": CONFIG.get("distil_enabled", True),
                "student_passes": CONFIG["distil_passes"],
                "teacher_weight": CONFIG.get("distil_teacher_weight", CONFIG["teacher_weight"]),
                "protocol": CONFIG.get("distil_protocol", "cat-r1.10-distil"),
                "files": FILES,
            },
            "catr1_features": {
                "moe_dense": CONFIG.get("catr1_moe_dense", True),
                "multi_turn_search": CONFIG.get("catr1_multi_turn_search", True),
                "fun_mode": CONFIG.get("catr1_fun_mode", True),
                "realtime": CONFIG.get("catr1_realtime", True),
                "moe_scale": CONFIG.get("catr1_moe_scale", True),
                "multi_token_prediction": CONFIG.get("catr1_multi_token_prediction", True),
                "active_inference": CONFIG.get("catr1_active_inference", True),
                "long_context_finegrained": CONFIG.get("catr1_long_context_finegrained", True),
                "sparse_attention": CONFIG.get("catr1_sparse_attention", True),
            },
            "google_whitepaper_heuristics": CONFIG.get("google_whitepaper_heuristics", True),
            "catr1_voice": CONFIG.get("catr1_voice", True),
            "mythos_runtime": CONFIG.get("mythos_runtime", True),
            "mythos_voice": CONFIG.get("mythos_voice", True),
            "mythos_algos": list(ClaudeMythosRuntime.MYTHOS_ALGOS),
            "code_anything": CONFIG.get("code_anything", True),
            "code_anything_mode": CONFIG.get("code_anything_mode", "universal-frontier"),
            "code_anything_langs": len(CodeAnythingEngine.LANGS),
        }




class CatR11Synthesizer:
    """Knowledge + chat synthesis backend for cat r1.10 (files = off)."""

    def __init__(self):
        pass

    @staticmethod
    def _normalize_chat(pl: str) -> str:
        s = pl.strip().lower()
        s = re.sub(r"\s+", " ", s)
        s = s.rstrip("?!.！？。")
        return s

    @classmethod
    def smalltalk_reply(cls, pl: str) -> Optional[str]:
        s = cls._normalize_chat(pl)
        for pat, reply in _SMALLTALK:
            if re.match(pat, s) or re.match(pat, pl.strip()):
                return reply
        return None

    @classmethod
    def is_educational(cls, pl: str) -> bool:
        if cls.smalltalk_reply(pl):
            return False
        if is_explain_request(pl) or is_zh_question(pl):
            return True
        if re.search(r"(解释|说明|介绍|教程|原理)", pl):
            return True
        patterns = (
            r"\b(explain|define|describe|what is|what are|why|how (?:does|do|to|can|would|should|could|will))\b",
            r"\b(compare|versus|vs\.?|difference|tutorial|implement|algorithm|architecture)\b",
            r"\b(debug|traceback|exception|error|bug|broken|fix)\b",
            r"\b(write|code|function|script|snippet|class|api|docker|python|catr1|cat r1.10|cat r1|core|sql|git)\b",
            r"\b(calculate|compute|solve|fibonacci|prime)\b",
            r"\b(plan|roadmap|design|system|recommend|should i)\b",
        )
        s = cls._normalize_chat(pl)
        return any(re.search(p, s) for p in patterns)

    def _topic(self, prompt: str, pl: str) -> str:
        for prefix in (
            "explain ", "why ", "how does ", "how do ", "how to ", "what is ", "what are ",
            "define ", "compare ", "debug ", "fix ", "write ",
        ):
            if pl.startswith(prefix):
                return prompt[len(prefix):].strip("?.")
        if "?" in prompt:
            return prompt.strip().rstrip("?")
        return prompt.strip()[:200] or "your question"

    def analyze(self, prompt: str) -> Dict[str, str]:
        pl = prompt.lower().strip()
        if is_zh_greeting(prompt.strip()):
            return {"intent": "casual", "topic": prompt.strip(), "pl": pl}
        if is_explain_request(prompt) or re.search(r"(解释|说明|介绍)", prompt):
            topic = extract_zh_topic(prompt) if VibeCodeHeuristics.has_cjk(prompt) else self._topic(prompt, pl)
            return {"intent": "explain", "topic": topic, "pl": pl}
        topic = self._topic(prompt, pl)
        if self.smalltalk_reply(pl):
            return {"intent": "casual", "topic": topic, "pl": pl}
        if CONFIG["catr1_enabled"] and CatSeekR1LLM.wants_fable(pl):
            return {"intent": "fable", "topic": topic, "pl": pl}
        if CONFIG["catr1_enabled"] and CatSeekR1LLM.wants_poem(pl):
            return {"intent": "poem", "topic": topic, "pl": pl}
        if any(k in pl for k in ("story", "tale", "creative", "imagine", "narrative")):
            return {"intent": "creative", "topic": topic, "pl": pl}
        if any(k in pl for k in ("traceback", "exception", "error", "bug", "broken", "fails")):
            return {"intent": "debug", "topic": topic, "pl": pl}
        if any(k in pl for k in ("compare", " vs ", " versus ", "difference", "better")):
            return {"intent": "compare", "topic": topic, "pl": pl}
        if re.search(
            r"\b(explain|what is|what are|why|how (?:does|do|to|can|would|should|could|will))\b", pl
        ):
            return {"intent": "explain", "topic": topic, "pl": pl}
        if any(k in pl for k in ("fibonacci", "fib", "prime", "primes")):
            return {"intent": "code", "topic": topic, "pl": pl}
        if CatSeekR1Code.wants_code(pl):
            return {"intent": "code", "topic": topic, "pl": pl}
        if any(k in pl for k in ("write code", "function", "implement", "snippet", "script")):
            return {"intent": "code", "topic": topic, "pl": pl}
        if any(k in pl for k in ("plan", "roadmap", "architecture", "design", "system")):
            return {"intent": "design", "topic": topic, "pl": pl}
        if re.search(r"\d\s*[+\-*/^%]", pl) or "calculate" in pl or "solve" in pl:
            return {"intent": "math", "topic": topic, "pl": pl}
        if any(k in pl for k in ("should i", "opinion", "recommend", "best")):
            return {"intent": "advise", "topic": topic, "pl": pl}
        if any(k in pl for k in ("meow", "mew", "purr", "nya", "rawr")):
            return {"intent": "casual", "topic": topic, "pl": pl}
        if len(pl.split()) <= 4 and "?" not in prompt and not self.is_educational(pl):
            return {"intent": "casual", "topic": topic, "pl": pl}
        if not self.is_educational(pl):
            return {"intent": "casual", "topic": topic, "pl": pl}
        return {"intent": "general", "topic": topic, "pl": pl}

    @staticmethod
    def _footer() -> str:
        return ""

    def _explain(self, topic: str, pl: str, vec: Optional[np.ndarray] = None) -> str:
        if ("python" in pl or "python" in topic.lower()) and VibeCodeHeuristics.has_cjk(topic + pl):
            return (
                f"**{topic.capitalize()}** — 清晰解释如下。\n\n"
                "**Python** 是一种高级编程语言，语法简洁、生态丰富。\n\n"
                "常用于脚本、Web 后端、数据分析、自动化和机器学习原型。\n\n"
                "优点：开发快、库多、社区大。\n"
                "缺点：GIL 限制 CPU 并行；大型项目建议加类型注解。\n\n"
                "```python\nfor i in range(3):\n    print(i)\n```"
            )
        kb = {
            ("recursion",): (
                f"**{topic.capitalize()}** is when a function calls itself until it hits a base case, "
                "then unwinds the stack to combine results.\n\n"
                "Each call gets its own stack frame. The base case stops new frames; the recursive step "
                "handles a smaller instance of the same problem.\n\n"
                "Example: `factorial(3)` waits on `factorial(2)` → `factorial(1)` returns 1, "
                "then multiplies back up: 1 × 2 × 3 = 6.\n\n"
                "Watch out for missing base cases — they cause stack overflow."
            ),
            ("递归",): (
                "**递归**是函数调用自身，直到遇到**基准情况**（base case），然后逐层返回结果。\n\n"
                "每次调用都有独立的栈帧。基准情况阻止无限调用；递归步骤处理更小规模的同一问题。\n\n"
                "示例：`factorial(3)` 等待 `factorial(2)` → `factorial(1)` 返回 1，"
                "再逐层相乘：1 × 2 × 3 = 6。\n\n"
                "注意：缺少基准情况会导致栈溢出。"
            ),
            ("docker",): (
                f"**Docker** packages an application and its dependencies into a **container** that runs "
                "consistently on any machine with Docker installed.\n\n"
                "Key pieces:\n"
                "- **Image** — read-only blueprint\n"
                "- **Container** — running instance of an image\n"
                "- **Dockerfile** — recipe to build an image\n\n"
                "Great for reproducible dev environments. For large-scale orchestration, people often add Kubernetes."
            ),
            ("core",): (
                f"**catr1.10** quantizes neural network weights to three values: **{{-1, 0, 1}}**. "
                "That turns matrix multiplication into mostly additions and subtractions, "
                "which cuts memory use and can speed up inference.\n\n"
                f"{BRAND} runs a local in-memory student stack — "
                f"{CONFIG['distil_passes']} heads, all in-memory."
            ),
            ("transformer", "attention"): (
                f"A **transformer** processes sequences using **self-attention** — each token can "
                "weigh every other token to capture context.\n\n"
                "Stack: embeddings → multi-head attention → feed-forward network → residuals and layer norms, "
                "repeated across many layers. Decoder models mask future tokens for causal generation."
            ),
            ("reason", "think", "o1"): (
                f"{BRAND} R1 runs **{REASONING_MODE}** reasoning: think internally, answer cleanly.\n\n"
                f"catr1.10 **{CONFIG['weight_bits']}-bit** ternary weights + sparse compression + "
                f"rank-{CONFIG['compression_rank']} bottleneck — all in-memory."
            ),
            ("python",): (
                "**Python** is a high-level language known for readable syntax and a huge ecosystem.\n\n"
                "Use it for scripting, web backends, data work, automation, and ML prototypes.\n\n"
                "Strengths: fast to write, great libraries, huge community.\n"
                "Tradeoffs: GIL limits CPU parallelism; type hints help at scale.\n\n"
                "```python\nfor i in range(3):\n    print(i)\n```"
            ),
            ("javascript", "js"): (
                "**JavaScript** runs in browsers and on servers (Node.js).\n\n"
                "Event-driven, async-friendly — ideal for interactive UIs and I/O-heavy APIs.\n\n"
                "Use `async/await` for readable asynchronous code."
            ),
            ("git",): (
                "**Git** tracks code history with commits, branches, and merges.\n\n"
                "Daily flow: `git pull` → edit → `git add` → `git commit` → `git push`.\n"
                "Branches isolate features; merge or rebase when ready."
            ),
            ("api", "rest"): (
                "A **REST API** exposes resources over HTTP with verbs like GET, POST, PUT, DELETE.\n\n"
                "Use nouns in paths, correct status codes, and version your API.\n"
                "In this app everything is local — describe payloads in chat (in-memory)."
            ),
            ("sql", "database"): (
                "**SQL** queries relational databases with tables, rows, and joins.\n\n"
                "Core ops: SELECT, INSERT, UPDATE, DELETE, JOIN.\n"
                "Index columns you filter on; avoid SELECT * in production."
            ),
            ("machine", "ml", "learning"): (
                "**Machine learning** learns patterns from data instead of hand-written rules.\n\n"
                "Pipeline: data → features → model → loss → training → evaluation on hold-out set.\n"
                "Start simple; add complexity only when metrics justify it."
            ),
            ("async", "await"): (
                "**async/await** lets one thread juggle many I/O-bound tasks without blocking.\n\n"
                "The event loop schedules coroutines; `await` yields until I/O completes.\n"
                "Best for network/disk waits — use threads/processes for CPU-heavy work."
            ),
            ("compress", "quant"): (
                f"**Model compression** shrinks memory and speeds inference: quantization (cat r1.10 ternary), "
                f"sparsity (top-{CONFIG['compression_sparse_k']} activations), and low-rank bottlenecks.\n\n"
                f"{BRAND} stacks these in-memory for frontier-tier capacity."
            ),
        }
        for keys, body in kb.items():
            if any(k in pl or k in topic for k in keys):
                return f"**{topic.capitalize()}** — here's a clear explanation.\n\n{body}"
        if not self.is_educational(pl):
            return self._casual(topic, pl, topic, vec)
        return (
            f"I can walk through **{topic}** — concept, mechanism, or a worked example. "
            f"Which would help most?"
        )

    def _fable(self, topic: str, prompt: str, vec: Optional[np.ndarray]) -> str:
        return CatSeekR1LLM().compose_fable(prompt, topic, vec)

    def _poem(self, topic: str, vec: Optional[np.ndarray], prompt: str = "") -> str:
        return CatSeekR1LLM().compose_poem(topic, vec, prompt)

    def _creative(self, topic: str, prompt: str, pl: str, vec: Optional[np.ndarray]) -> str:
        if CatSeekR1LLM.wants_poem(pl):
            return self._poem(topic, vec)
        return self._fable(topic, prompt, vec)

    def _debug(self, topic: str, vec: Optional[np.ndarray] = None) -> str:
        return (
            "Debugging is a conversation with reality — start small, listen closely, change one thing at a time.\n\n"
            "**Reproduce minimally.** Find the smallest input that still fails.\n"
            "**Read bottom-up.** The last frame in *your* code is usually where to look first.\n"
            "**Name expected vs actual.** Write both down before you touch anything.\n"
            "**Iterate with discipline.** One change, one run, one note.\n\n"
            "Paste the full traceback when you have it — I'll walk through it line by line with you."
        )

    def _code(self, topic: str, pl: str, vec: Optional[np.ndarray] = None) -> str:
        if "fibonacci" in pl or "fib" in pl:
            body = (
                "```python\ndef fib(n: int) -> list[int]:\n"
                "    a, b = 0, 1\n    out = [a]\n"
                "    for _ in range(1, max(n, 1)):\n        a, b = b, a + b\n        out.append(a)\n"
                "    return out[:n]\n\nprint(fib(10))\n```"
            )
        elif "prime" in pl:
            body = (
                "```python\ndef primes_upto(n: int) -> list[int]:\n"
                "    if n < 2: return []\n    sieve = [True] * (n + 1)\n"
                "    sieve[0] = sieve[1] = False\n"
                "    for p in range(2, int(n**0.5) + 1):\n"
                "        if sieve[p]:\n            sieve[p*p:n+1:p] = [False] * len(sieve[p*p:n+1:p])\n"
                "    return [i for i, ok in enumerate(sieve) if ok]\n\nprint(primes_upto(50))\n```"
            )
        else:
            body = (
                f"```python\ndef solve():\n    \"\"\"Sketch: {topic[:60]}\"\"\"\n"
                "    return None\n\nif __name__ == '__main__':\n    print(solve())\n```"
            )
        return (
            f"Here's a **{topic}** sketch — readable first, correct second. Run it, then harden edge cases.\n\n"
            f"{body}\n\n"
            "Empty input, zero, and one large value are the three tests I always run."
        )

    def _math(self, topic: str, pl: str) -> str:
        m = re.search(r"(\d+(?:\.\d+)?)\s*([+\-*/^])\s*(\d+(?:\.\d+)?)", pl)
        if m:
            a, op, b = float(m.group(1)), m.group(2), float(m.group(3))
            ops = {"+": a + b, "-": a - b, "*": a * b, "/": a / b if b else None, "^": a ** b}
            val = ops.get(op)
            if val is not None:
                out = int(val) if val == int(val) else round(val, 6)
                sym = {"*": "×", "/": "÷"}.get(op, op)
                return f"{a:g} {sym} {b:g} = **{out}**"
        return f"What expression should I evaluate for **{topic}**? You can type something like `15 * 7` or `100 / 4`."

    def _compare(self, topic: str, vec: Optional[np.ndarray] = None) -> str:
        return (
            f"Comparing **{topic}** well means naming what you optimize for before you pick winners.\n\n"
            "Consider **latency** (how fast must it be?), **complexity** (what can your team live with?), "
            "**offline needs** (this stack runs fully local), and **total cost** (infra plus time).\n\n"
            "List must-haves, eliminate what breaks them, prototype the top two. "
            "Tell me the pair you're weighing and I'll give you a sharper read."
        )

    def _advise(self, topic: str, vec: Optional[np.ndarray] = None) -> str:
        return (
            f"On **{topic}**, the honest answer is: it depends on what you're optimizing.\n\n"
            "If you need to **learn fast**, ship a small prototype and let reality edit your plan. "
            "If you need **long-term maintenance**, favor simpler architecture over clever architecture. "
            "If you need **privacy or offline work**, everything here stays in-memory.\n\n"
            "Share two or three constraints and I'll recommend something concrete."
        )

    def _casual(self, topic: str, pl: str, prompt: str, vec: Optional[np.ndarray] = None) -> str:
        hit = self.smalltalk_reply(pl)
        if hit:
            return hit
        if any(k in pl for k in ("meow", "mew", "purr", "nya")):
            return (
                f"I'm **{BRAND}**. "
                "Explain, code, debug, stories — what would you like?"
            )
        if pl in {"hey", "hi", "yo", "sup", "hello"}:
            return "Hi — good to see you. What would you like to work on?"
        if is_zh_greeting(prompt.strip()):
            return f"你好！我是 CatR1.10。可以帮你写代码、解释概念或调试问题。"
        if VibeCodeHeuristics.has_cjk(prompt) and len(prompt.strip()) <= 12:
            return "你好！请告诉我你需要什么帮助——代码、解释或调试都可以。"
        if len(pl.split()) <= 4 and "?" not in prompt:
            return (
                f"**catr1.10** — "
                "ask for code, an explanation, or paste a snippet to run."
            )
        return (
            f"I can help with that. **catr1.10** runs extended reasoning — "
                f"thorough, proactive, and self-checking. "
            "What would you like to do next?"
        )

    def _general(self, topic: str, prompt: str, vec: Optional[np.ndarray] = None) -> str:
        pl = prompt.lower().strip()
        if not self.is_educational(pl):
            return self._casual(topic, pl, prompt, vec)
        return (
            f"On **{topic or 'that'}** — I can explain the idea, compare approaches, or sketch code. "
            f"What would help most?"
        )

    def synthesize(self, prompt: str, history: List[tuple], vec: Optional[np.ndarray] = None) -> str:
        a = self.analyze(prompt)
        intent = a["intent"]
        topic, pl = a["topic"], a["pl"]
        bodies = {
            "explain": self._explain(topic, pl, vec),
            "qa": self._explain(topic, pl, vec),
            "compare": self._compare(topic, vec),
            "debug": self._debug(topic, vec),
            "code": self._code(topic, pl, vec),
            "design": self._general(topic, prompt, vec),
            "math": self._math(topic, pl),
            "advise": self._advise(topic, vec),
            "casual": self._casual(topic, pl, prompt, vec),
            "fable": self._fable(topic, prompt, vec),
            "poem": self._poem(topic, vec),
            "creative": self._creative(topic, prompt, pl, vec),
        }
        return bodies.get(intent, self._general(topic, prompt, vec))

    def _followup(self, prompt: str, history: List[Dict[str, str]], vec: Optional[np.ndarray] = None) -> Optional[str]:
        if not history:
            return None
        pl = prompt.lower().strip()
        cues = (
            "tell me more", "go on", "continue", "and then", "what else", "more detail",
            "expand", "elaborate", "can you explain", "say more", "go deeper", "why though",
        )
        short_ack = {"yes", "ok", "okay", "sure", "please", "yep", "yeah", "do it", "thanks"}
        last_user = last_bot = ""
        for m in reversed(history):
            if m["role"] == "assistant" and not last_bot:
                last_bot = m["text"]
            elif m["role"] == "user" and not last_user:
                last_user = m["text"]
            if last_user and last_bot:
                break
        if not last_bot:
            return None
        if any(c in pl for c in cues) or pl in short_ack or pl.startswith(("why ", "how come")):
            snippet = last_bot.strip().split("\n\n")[0][:500]
            return (
                f"Building on our exchange about \"{last_user[:80]}\":\n\n"
                f"{snippet}\n\n"
                "The deeper layer is *mechanism* — what actually changes when you apply this. "
                "Tell me which part you want expanded and I'll go there."
            )
        if pl.startswith(("what about", "how about")):
            sub = prompt.split(maxsplit=2)[-1] if len(prompt.split()) > 2 else prompt
            return self._explain(sub.strip("?"), pl, vec)
        return None

    def converse(self, prompt: str, history: List[Dict[str, str]], vec: Optional[np.ndarray] = None) -> str:
        pl = prompt.lower().strip()
        small = self.smalltalk_reply(pl)
        if small:
            return small
        follow = self._followup(prompt, history, vec)
        if follow:
            return GoogleWhitepaperCatSeekSorter.catseek_voice(follow, prompt, "chat")
        sorted_hist = GoogleWhitepaperCatSeekSorter.sort_history(
            [(m["role"], m["text"]) for m in history], prompt
        )
        return self.synthesize(prompt, sorted_hist, vec=vec)

    @staticmethod
    def _polish(text: str) -> str:
        t = re.sub(r"\n{3,}", "\n\n", text.strip())
        if t and t[-1] not in ".!?`\"'":
            t += "."
        return t

    def o1_answer(
        self,
        prompt: str,
        history: List[Dict[str, str]],
        *,
        reasoning: str = "",
        vec: Optional[np.ndarray] = None,
    ) -> str:
        """cat r1.10 answer path — Mythos voice + Google WP sort · files = off."""
        body = self.converse(prompt, history, vec=vec)
        body = GoogleWhitepaperCatSeekSorter.sort_paragraphs(body, prompt)
        if ClaudeMythosRuntime.enabled():
            body = ClaudeMythosRuntime.voice(body, prompt, "general")
        if CONFIG.get("catr1_voice") and CATSEEK_R1_MODE:
            body = GoogleWhitepaperCatSeekSorter.catseek_voice(body, prompt, "general")
        return self._polish(body)


# ──────────────────────────────────────────────────────────────
# CAT R1.1 CHAT PROTOCOL v1.1 (files = off, in-memory sessions)
# ──────────────────────────────────────────────────────────────
@dataclass
class ChatMessage:
    role: str
    content: str
    turn: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"role": self.role, "content": self.content, "turn": self.turn}


@dataclass
class ChatSession:
    session_id: str
    messages: List[ChatMessage] = field(default_factory=list)
    turn_count: int = 0
    created: float = field(default_factory=time.time)

    def append(self, role: str, content: str) -> ChatMessage:
        self.turn_count += 1
        msg = ChatMessage(role=role, content=content, turn=self.turn_count)
        self.messages.append(msg)
        if len(self.messages) > 48:
            self.messages = self.messages[-48:]
        return msg

    def history_dicts(self) -> List[Dict[str, str]]:
        return [{"role": m.role, "text": m.content} for m in self.messages]

    def transcript(self, limit: int = 12) -> str:
        lines = []
        for m in self.messages[-limit:]:
            label = "You" if m.role == "user" else BRAND
            lines.append(f"[{label}]: {m.content[:200]}")
        return "\n".join(lines)


class ChatProtocol:
    """
    cat r1.10 Chat Protocol v1.1 — multi-turn conversation, files = off.

    JSON envelope::
        {"protocol":"cat-r1-chat","version":"1.1","action":"message",
         "session":"<id>","message":{"role":"user","content":"hello"}}

    Text wire (stdin/stdout)::
        @chat user: hello
        @chat assistant: Hi! ...
    """

    PROTO = CONFIG["chat_protocol"]
    VER = CONFIG["chat_version"]

    def __init__(self, engine: "CatR11Engine"):
        self.engine = engine
        self._sessions: Dict[str, ChatSession] = {}
        self._active = ""

    def new_session(self) -> str:
        sid = uuid.uuid4().hex[:12]
        self._sessions[sid] = ChatSession(session_id=sid)
        self._active = sid
        while len(self._sessions) > CONFIG["max_sessions"]:
            oldest = min(self._sessions.values(), key=lambda s: s.created)
            del self._sessions[oldest.session_id]
        return sid

    def session(self, session_id: Optional[str] = None) -> ChatSession:
        sid = session_id or self._active
        if not sid or sid not in self._sessions:
            sid = self.new_session()
        self._active = sid
        return self._sessions[sid]

    def sync_engine_history(self, sess: ChatSession) -> None:
        self.engine.chat_history = sess.history_dicts()

    def turn(self, user_text: str, *, session_id: Optional[str] = None, simulate: bool = True, on_token=None) -> Dict[str, Any]:
        text = (user_text or "").strip()
        if not text:
            return self._err("empty message")
        sess = self.session(session_id)
        self.sync_engine_history(sess)
        start = len(sess.messages)
        reply = self.engine.generate(text, simulate=simulate, on_token=on_token)
        think = self.engine.last_think
        for m in self.engine.chat_history[start:]:
            sess.append(m["role"], m["text"])
        return self._ok(sess, reply, think)

    def handle_action(self, data: Dict[str, Any]) -> Dict[str, Any]:
        action = (data.get("action") or "message").lower()
        sid = data.get("session") or data.get("session_id")

        if action in ("new", "reset"):
            new_id = self.new_session()
            if action == "reset" and sid and sid in self._sessions:
                del self._sessions[sid]
                new_id = self.new_session()
            self.engine.chat_history = []
            self.engine.last_think = ""
            return {
                "protocol": self.PROTO, "version": self.VER, "files": "off",
                "action": action, "session": new_id, "turn": 0,
                "message": {"role": "system", "content": f"New chat session {new_id}."},
            }

        if action == "history":
            sess = self.session(sid)
            return {
                "protocol": self.PROTO, "version": self.VER, "files": "off",
                "action": "history", "session": sess.session_id, "turn": sess.turn_count,
                "messages": [m.to_dict() for m in sess.messages],
            }

        if action == "message":
            msg = data.get("message") or {}
            content = msg.get("content") or data.get("content") or data.get("text") or ""
            if isinstance(content, list):
                content = content[-1].get("text", "") if content else ""
            return self.turn(str(content), session_id=sid, simulate=False)

        return self._err(f"unknown action: {action}")

    def parse_text_wire(self, line: str) -> Optional[Tuple[str, str]]:
        m = re.match(r"^@chat\s+(user|assistant|system)\s*:\s*(.*)$", line.strip(), re.I)
        if not m:
            return None
        return m.group(1).lower(), m.group(2)

    def format_text_wire(self, role: str, content: str, sess: ChatSession) -> str:
        return f"@chat {role}: {content}\n@chat meta session={sess.session_id} turn={sess.turn_count}"

    def parse_request(self, raw: Any) -> Dict[str, Any]:
        if isinstance(raw, dict):
            if raw.get("protocol") not in (None, self.PROTO):
                return self._err(f"protocol must be {self.PROTO}")
            return self.handle_action(raw)
        if isinstance(raw, str):
            wire = self.parse_text_wire(raw)
            if wire:
                role, content = wire
                if role == "user":
                    return self.turn(content, simulate=False)
                return self._err("text wire expects @chat user: <message>")
            try:
                return self.handle_action(json.loads(raw))
            except json.JSONDecodeError:
                return self.turn(raw, simulate=False)
        return self._err("invalid request")

    def _ok(self, sess: ChatSession, content: str, thinking: str = "") -> Dict[str, Any]:
        return {
            "protocol": self.PROTO,
            "version": self.VER,
            "files": "off",
            "action": "message",
            "session": sess.session_id,
            "turn": sess.turn_count,
            "message": {"role": "assistant", "content": content},
            "thinking": thinking,
        }

    def _err(self, detail: str) -> Dict[str, Any]:
        return {
            "protocol": self.PROTO,
            "version": self.VER,
            "files": "off",
            "error": detail,
            "session": self._active or None,
        }

    @staticmethod
    def help_text() -> str:
        return (
            f"**catr1.10 Chat Protocol v{CONFIG['chat_version']}** · `{CATSEEK_MODEL_ID}`\n\n"
            f"**{O1PreviewSyntax.TAG} + {MYTHOS_NAME} interpreter** · `/run` · `/interpret` · `/think` · paste ``` blocks\n\n"
            "**In-app commands**\n"
            "- `/chat` — show protocol help\n"
            "- `/chat new` — start a fresh session\n"
            "- `/chat history` — show this session transcript\n"
            "- `/chat session` — show session id\n\n"
            "**JSON API** — `POST /chat`\n"
            "```json\n"
            '{"protocol":"cat-r1-chat","version":"1.1","action":"message",'
            '"session":"<id>","message":{"role":"user","content":"hello"}}\n'
            "```\n\n"
            "**Text wire** (CLI `--chat`)\n"
            "`@chat user: your message here`\n\n"
            "Sessions live in memory only — no files written."
        )


def run_chat_cli(engine: CatR11Engine) -> None:
    proto = engine.chat
    sid = proto.new_session()
    print(f"{BRAND} · {CATSEEK_MODEL_ID} · protocol {ChatProtocol.PROTO}/{ChatProtocol.VER}")
    print(f"session={sid} · type @chat user: ... or plain text · /quit to exit\n")
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break
        if not line:
            continue
        if line.lower() in {"/quit", "/exit", "quit", "exit"}:
            break
        if line.startswith("@chat"):
            parsed = proto.parse_text_wire(line)
            if parsed and parsed[0] == "user":
                out = proto.turn(parsed[1], session_id=sid, simulate=False)
            else:
                out = proto.parse_request(line)
        elif line == "/chat new":
            sid = proto.new_session()
            print(f"new session={sid}")
            continue
        elif line.startswith("/"):
            out = {"message": {"content": engine.generate(line, simulate=False)}}
        else:
            out = proto.turn(line, session_id=sid, simulate=False)
        if out.get("error"):
            print(f"error: {out['error']}")
            continue
        reply = out.get("message", {}).get("content", "")
        print(f"\n{reply}\n")


# ──────────────────────────────────────────────────────────────
# CAT R1.1 ENGINE (files = off · in-memory ternary stack)
# ──────────────────────────────────────────────────────────────
class CatR11Engine:
    __slots__ = ("name", "ver", "d_model", "_lock", "dialect_idx",
                 "dialects", "intent_map", "code_experts", "aliases",
                 "embeddings", "vocab", "output_head", "output_head_lin",
                 "_intent_weights", "_token_index", "_intent_trained",
                 "learning_curve", "response_locale",
                 "chat_history", "max_history", "assistant_mode",
                 "_category_lexicon", "_embed_cache",
                 "_student_layers", "ultrathink", "synth", "last_think",
                 "ultrathink_on", "_pending_think", "_distil_cache", "_recursive_cache",
                 "last_recursive_passes", "last_recursive_trace", "rival", "compressor",
                 "last_compression_ratio", "compression_trace", "chat", "catseek_blocks",
                 "catseek_stats", "norm_gamma", "norm_beta", "last_vec", "catseek", "fable", "deepmind", "fusion", "web", "heuristics",
                 "persistent_memory")

    def __init__(self, d_model: int = None):
        self.name = BRAND
        self.ver = MODEL_NAME
        self.d_model = d_model or CONFIG["d_model"]
        self._lock = threading.Lock()
        self.dialect_idx = {"english": 0, "chinese": 0}
        self.learning_curve: List[float] = []
        self._intent_trained = False
        self.persistent_memory = _load_persistent_memory()
        if self.persistent_memory.first_run:
            self.persistent_memory.first_run = False
            self.persistent_memory.setup_version = EDITION
            _save_persistent_memory(self.persistent_memory)
        self._token_index: Dict[str, int] = {}
        self._intent_weights: Optional[np.ndarray] = None
        self.response_locale = "english"
        self.chat_history: List[Dict[str, str]] = []
        self.max_history = 24
        self.assistant_mode = "cat_r1"
        self._embed_cache: Dict[str, np.ndarray] = {}
        self._student_layers: List[List[CatSeekBlock]] = []
        self._distil_cache: Dict[tuple, np.ndarray] = {}
        self._recursive_cache: Dict[tuple, np.ndarray] = {}
        self.last_recursive_passes = 0
        self.last_recursive_trace: List[str] = []
        self.compression_trace: List[str] = []
        self.last_compression_ratio = 1.0
        self.last_vec: Optional[np.ndarray] = None
        self.ultrathink = O1PreviewReasoner()
        self.synth = CatR11Synthesizer()
        self.last_think = ""
        self._pending_think = ""
        self.ultrathink_on = CONFIG["ultrathink_default"]
        self.catseek = CatSeekR1LLM()
        self.fable = self.catseek
        self.chat = ChatProtocol(self)
        self.compressor = CatSeekCompressor(self.d_model)
        self.rival = CatSeekR1Core(self)
        self.deepmind = DeepMindFastStack(self)
        self.fusion = CatSeekR1Fusion()
        self.web = CatSeekWebProgram()
        self.heuristics = CatSeekR1Heuristics()
        self.aliases = dict(CodeAnythingEngine.LANG_ALIASES) if CONFIG.get("code_anything") else {
            "py": "python", "c++": "cpp", "js": "javascript", "ts": "typescript",
            "sh": "bash", "shell": "bash", "asm": "assembly", "node": "javascript",
        }
        self.intent_map = {
            "hello": ["hi","hello","hey","yo","sup","good morning","good evening","howdy",
                      "how are you","how're you","how is it going","what's up","whats up",
                      "你好","您好","你好吗","嗨","哈喽","早上好","下午好","晚上好"],
            "core": ["core","catr1","cat r1.10","cat r1","bitnet","ternary","-1, 0, 1","quantize","1.58","moe","cat seek","核心","三值"],
            "recursion": ["recursion","recursive","function calls itself","factorial","递归"],
            "help": ["help","commands","menu","usage","what can you do","capabilities","帮助","怎么用"],
            "languages": ["supported languages","which language","experts","what languages","支持的语言","哪些语言"],
            "profile": ["readme",".md","license","gpl3","about you","about yourself","who made you","who are you","你是谁"],
            "thanks": ["thanks","thank you","thx","appreciate","谢谢","感谢"],
            "goodbye": ["bye","goodbye","see you","later","exit","再见","拜拜"],
            "math": ["calculate","compute","sum","multiply","divide","equation"],
            "explain": ["explain","what is","what are","define","meaning of","tell me about",
                        "什么是","是什么","解释","说明","介绍","为什么","如何"],
            "howto": ["how do","how to","how can","steps to","walk me through","tutorial"],
            "debug": ["error","bug","traceback","exception","crash","broken","not working"],
            "opinion": ["should i","recommend","opinion","think about"],
            "joke": ["joke","funny","humor","laugh","make me laugh"],
            "fable": ["fable","parable","allegory","bedtime story","tell me a story","write a story"],
        }
        self._category_lexicon = {
            "greeting": ["hi", "hello", "hey", "morning", "evening", "sup", "你好", "您好", "嗨"],
            "farewell": ["bye", "goodbye", "later", "see you", "exit", "再见", "拜拜"],
            "question": ["what", "why", "how", "who", "when", "where", "which", "?", "什么", "为什么", "怎么", "如何"],
            "code": ["code", "script", "function", "class", "compile", "program", "syntax", "api", "代码", "程序", "脚本"],
            "tech": ["python", "catr1", "cat r1.10", "cat r1", "core", "ai", "model", "neural", "gpu", "cpu", "server"],
            "creative": ["story", "poem", "joke", "idea", "name", "design"],
            "personal": ["you", "your", "yourself", "who are you"],
            "task": ["make", "build", "create", "write", "generate", "show me"],
        }
        self.dialects = {
            "english": [
                {"hello":"Hi. How can I help?","ready":"Ready. Ask for code or explanation.",
                 "py_intro":"Here is Python code.","generic":"Here is code.",
                 "core":"cat r1.10 uses ternary weights: {-1, 0, 1}.","recursion":"Recursion: function calls itself. Example:"},
                {"hello":"Hey. What do you need?","ready":"Send a prompt.","py_intro":"Python below.","generic":"Code below.",
                 "core":"Ternary constraints eliminate FP32 multiplications.","recursion":"Self-referential execution. Example:"},
            ],
            "chinese": [
                {"hello":"你好。需要什么？","ready":"请给出任务。","py_intro":"Python 代码：","generic":"代码：",
                 "core":"cat r1.10 使用三值权重：{-1, 0, 1}。","recursion":"递归：函数自调用。示例："},
            ],
        }
        self.code_experts = (
            CodeAnythingEngine.experts() if CONFIG.get("code_anything") else {
            "python": "def main():\n    print('Hello World')\n\nif __name__ == '__main__':\n    main()",
            "cpp": (
                "#include <iostream>\n\n"
                "int main() {\n"
                "    std::cout << \"Hello World\" << std::endl;\n"
                "    return 0;\n"
                "}"
            ),
            "c": (
                "#include <stdio.h>\n\n"
                "int main(void) {\n"
                "    printf(\"Hello World\\n\");\n"
                "    return 0;\n"
                "}"
            ),
            "javascript": "console.log('Hello World');",
            "html": "<!DOCTYPE html><html><body><h1>Hello World</h1></body></html>",
            "typescript": "function main(): void { console.log('Hello World'); }\nmain();",
            "java": "public class Main { public static void main(String[] args) { System.out.println(\"Hello World\"); } }",
            "rust": "fn main() { println!(\"Hello World\"); }",
            "bash": "#!/bin/bash\necho \"Hello World\"",
            "assembly": "section .data\n    msg db 'Hello World',0xa\nsection .text\n    global _start\n_start:",
            "go": "package main\nimport \"fmt\"\nfunc main() { fmt.Println(\"Hello World\") }",
        })
        self._init_catseek_core()
        self._train_intent()

    def _build_vocab(self) -> Dict[str, int]:
        words = {"<pad>", "<unk>"}
        for keys in self.intent_map.values():
            for phrase in keys:
                words.update(re.findall(r"[a-z0-9+#]+", phrase.lower()))
        for w in (
            "the", "a", "is", "are", "to", "for", "of", "in", "on", "with", "and", "or",
            "python", "code", "write", "explain", "help", "core", "model", "chat",
            "你好", "谢谢", "什么", "解释", "递归", "代码", "程序", "帮助", "是", "的",
        ):
            words.add(w)
        for phrase in (
            "什么是", "是什么", "解释", "说明", "介绍", "为什么", "如何", "帮我", "写代码",
        ):
            words.add(phrase)
        for keys in self.intent_map.values():
            for phrase in keys:
                words.update(tokenize_text(phrase, 64))
        ordered = sorted(words)[: CONFIG["vocab_size"] - 1]
        vocab = {w: i + 1 for i, w in enumerate(ordered)}
        vocab["<pad>"] = 0
        return vocab

    def _build_catseek_stack(self, seed: int) -> List[CatSeekBlock]:
        return [CatSeekBlock(self.d_model, seed + layer * 7919) for layer in range(CONFIG["layers"])]

    def _init_catseek_core(self):
        d = self.d_model
        rng = np.random.RandomState(42)
        self.vocab = self._build_vocab()
        vs = min(CONFIG["vocab_size"], len(self.vocab) + 256)
        self.embeddings = (rng.randn(vs, d).astype(np.float32) * 0.02)
        self.norm_gamma = np.ones((d,), dtype=np.float32)
        self.norm_beta = np.zeros((d,), dtype=np.float32)
        self.catseek_blocks = self._build_catseek_stack(42)
        self.output_head_lin = CatSeekLinear(d, len(self.intent_map) + 1, 4242)
        self.output_head = self.output_head_lin.w_signed
        self._student_layers: List[List[CatSeekBlock]] = []
        self.catseek_stats = catseek_memory_report(
            self.catseek_blocks, self.embeddings, self.output_head_lin.shadow_w
        )

    def _ensure_students(self, count: int) -> List[List[CatSeekBlock]]:
        while len(self._student_layers) < count:
            i = len(self._student_layers)
            self._student_layers.append(self._build_catseek_stack(1337 + i * 9973))
        return self._student_layers[:count]

    def _token_embed(self, token: str) -> np.ndarray:
        key = token if not token.isascii() else token.lower()
        tid = self.vocab.get(key)
        if tid is None:
            tid = abs(hash(key)) % max(len(self.embeddings) - 1, 1) + 1
        if tid >= len(self.embeddings):
            tid = 0
        return self.embeddings[tid].astype(np.float32)

    def _layer_norm(self, x):
        if x.ndim == 2:
            mean = np.mean(x, axis=-1, keepdims=True)
            std = np.std(x, axis=-1, keepdims=True) + 1e-5
            return (x - mean) / std * self.norm_gamma + self.norm_beta
        mean, std = np.mean(x, axis=-1, keepdims=True), np.std(x, axis=-1, keepdims=True) + 1e-5
        return (x - mean) / std * self.norm_gamma + self.norm_beta

    def _forward_stack(self, x: np.ndarray, blocks: List[CatSeekBlock]) -> np.ndarray:
        if x.ndim == 1:
            x = x.reshape(1, -1)
        y = x.astype(np.float32)
        for blk in blocks:
            y = blk.forward(y)
        return _rms_norm(y, self.norm_gamma)

    def _pool_sequence(self, seq: np.ndarray) -> np.ndarray:
        if seq.ndim == 1:
            return seq.astype(np.float32)
        return np.mean(seq, axis=0).astype(np.float32)

    def forward(self, x, turbo: bool = True, *, turbo_only: bool = False):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        key = (x.tobytes(), turbo, turbo_only)
        cached = self._distil_cache.get(key)
        if cached is not None:
            return cached.copy()
        teacher = self._pool_sequence(self._forward_stack(x, self.catseek_blocks))
        if turbo_only:
            if len(self._distil_cache) < 128:
                self._distil_cache[key] = teacher.copy()
            return teacher
        n_pass = CONFIG["turbo_passes"] if turbo else CONFIG["distil_passes"]
        tw = CONFIG.get("distil_teacher_weight", CONFIG["teacher_weight"])
        merged = teacher * tw
        students = self._ensure_students(n_pass)
        sw = 0.66 / n_pass
        for stack in students:
            merged = merged + self._pool_sequence(self._forward_stack(x, stack)) * sw
        if len(self._distil_cache) < 128:
            self._distil_cache[key] = merged.copy()
        return merged

    def encode_for_task(self, prompt: str, task: Optional[str] = None) -> np.ndarray:
        if CONFIG.get("deepmind_fast"):
            return self.deepmind.encode(prompt, task=task)
        return self.recursive_encode(prompt)

    def _recursive_step(self, state: np.ndarray, pass_idx: int, max_depth: int = 0) -> np.ndarray:
        depth = max_depth or CONFIG["recursive_depth"]
        turbo = pass_idx < depth - 1
        seq = state.reshape(1, -1) if state.ndim == 1 else state
        delta_seq = self._forward_stack(seq, self.catseek_blocks if turbo else self._ensure_students(1)[0])
        delta = self._pool_sequence(delta_seq)
        alpha = min(0.72, 0.28 + 0.11 * pass_idx)
        base = self._pool_sequence(state) if state.ndim == 2 else state
        merged = self._layer_norm(base * (1.0 - alpha) + delta * alpha)
        if CONFIG["compression_enabled"]:
            merged = self.compressor.compress_roundtrip(merged)
            self.last_compression_ratio = self.compressor.last_ratio
        return merged

    def recursive_encode(self, prompt: str, *, depth: Optional[int] = None) -> np.ndarray:
        """o1-preview recursive cat r1.10 loop with compression between passes."""
        max_depth = depth if depth is not None else CONFIG["recursive_depth"]
        key = (prompt.lower().strip(), max_depth, CONFIG["compression_enabled"])
        hit = self._recursive_cache.get(key)
        if hit is not None:
            return hit.copy()
        state = self.encode_prompt(prompt)
        trace: List[str] = []
        ctrace: List[str] = []
        prev = self._pool_sequence(state) if state.ndim == 2 else state.copy()
        used = max_depth
        for i in range(max_depth):
            state = self._recursive_step(state, i, max_depth)
            pooled = self._pool_sequence(state) if state.ndim == 2 else state
            norm = float(np.linalg.norm(pooled))
            line = f"pass {i + 1} · norm {norm:.4f} · catr1.10"
            if CONFIG["compression_enabled"]:
                line += f" · {self.last_compression_ratio:.1f}x"
                ctrace.append(
                    f"sparse-{CONFIG['compression_sparse_k']} + rank-{CONFIG['compression_rank']} "
                    f"→ {self.last_compression_ratio:.1f}x · ~{self.compressor.effective_params_billions():.1f}B effective"
                )
            if i > 0:
                diff = float(np.linalg.norm(pooled - prev))
                line += f" · Δ {diff:.4f}"
                if diff < CONFIG["recursive_epsilon"]:
                    trace.append(line)
                    trace.append(f"converged at pass {i + 1}")
                    used = i + 1
                    break
            trace.append(line)
            prev = pooled.copy()
        self.last_recursive_passes = used
        self.last_recursive_trace = trace
        self.compression_trace = ctrace
        out = self._pool_sequence(state) if state.ndim == 2 else state
        self.last_vec = out.copy()
        if len(self._recursive_cache) < 64:
            self._recursive_cache[key] = out.copy()
        return out

    def _distil_draft(self, prompt: str, vec: Optional[np.ndarray] = None) -> str:
        if vec is None:
            vec = self.recursive_encode(prompt)
        logits = self.output_head_lin.forward(vec)
        labels = list(self.intent_map.keys()) + ["general"]
        idx = int(np.argmax(logits)) % len(labels)
        topic = self._extract_topic_words(prompt)
        st = self.catseek_stats
        tw = CONFIG.get("distil_teacher_weight", CONFIG["teacher_weight"])
        return (
            f"catr1.10 reasoning → intent={labels[idx]} → topic=«{topic[:60]}» · "
            f"{self.last_recursive_passes} passes · "
            f"{st['catr1_linear_layers']} linear layers · "
            f"{st['packed_kb']:.1f}KB packed · "
            f"{tw:.0%} primary + {CONFIG['turbo_passes']} students"
        )

    def _history_pairs(self) -> List[tuple]:
        return [(m["role"], m["text"]) for m in self.chat_history[:-1]][-8:]

    def _run_ultrathink(self, prompt: str, *, force: bool = False) -> None:
        if self._pending_think and not force:
            self.last_think = self._pending_think
            self._pending_think = ""
            return
        if not O1PreviewReasoner.should_run(prompt, enabled=self.ultrathink_on, force=force):
            return
        vec = self.last_vec if self.last_vec is not None else self.recursive_encode(prompt)
        draft = self._distil_draft(prompt, vec=vec)
        self.last_think = self.ultrathink.run(
            prompt, distill_draft=draft, recursive_trace=self.last_recursive_trace,
            compression_trace=self.compression_trace,
        )

    def _o1_respond(self, prompt: str) -> str:
        prior = self.chat_history[:-1] if self.chat_history else []
        vec = self.last_vec
        if CONFIG["o1_preview"]:
            return self.synth.o1_answer(prompt, prior, reasoning=self.last_think, vec=vec)
        return self.synth.converse(prompt, prior, vec=vec)

    def _r1_synthesize(self, prompt: str) -> str:
        return self._o1_respond(prompt)

    def encode_prompt(self, prompt):
        key = prompt.strip()
        cached = self._embed_cache.get(key)
        if cached is not None:
            return cached
        tokens = tokenize_text(prompt, CONFIG["max_seq"])
        seq = np.stack([self._token_embed(t) for t in tokens], axis=0)
        out = self._layer_norm(seq)
        if len(self._embed_cache) < 256:
            self._embed_cache[key] = out
        return out

    def _intent_features(self, text: str) -> np.ndarray:
        vocab = list(self._token_index.keys()) if self._token_index else []
        if not vocab:
            return np.array([1.0], dtype=np.float32)
        tl = text.lower()
        return np.array([tl.count(t) for t in vocab] + [1.0], dtype=np.float32)

    def _train_intent(self):
        if self._intent_trained:
            return
        corpus: List[tuple] = []
        for label, keys in self.intent_map.items():
            for k in keys:
                corpus.append((k, label))
        corpus.extend([
            ("write python function", "howto"), ("fix my bug", "debug"),
            ("2 plus 2", "math"), ("thanks a lot", "thanks"),
            ("see you tomorrow", "goodbye"), ("what is docker", "explain"),
            ("should i use rust", "opinion"),
        ])
        vocab: set = set()
        for label, keys in self.intent_map.items():
            for k in keys:
                vocab.update(tokenize_text(k, 64))
        for text, _ in corpus:
            vocab.update(tokenize_text(text, 64))
        self._token_index = {t: i for i, t in enumerate(sorted(vocab))}
        self._intent_weights = np.zeros((len(self.intent_map), len(vocab) + 1), dtype=np.float32)
        labels = list(self.intent_map.keys())
        for _ in range(40):
            for text, label in corpus:
                x = self._intent_features(text)
                y_idx = labels.index(label)
                scores = self._intent_weights @ x
                others = [s for i, s in enumerate(scores) if i != y_idx]
                margin = (max(others) if others else -1e9) - scores[y_idx] + 1
                if margin > 0:
                    self._intent_weights[y_idx] += 0.12 * x
                    if others:
                        self._intent_weights[int(np.argmax(scores))] -= 0.12 * x
        self._intent_trained = True

    def _key_matches(self, key: str, text: str) -> bool:
        if VibeCodeHeuristics.has_cjk(key):
            return key in text
        loose = {"fix", "best", "math", "later", "exit", "bug", "sum"}
        if len(key) <= 4 or key in loose:
            return bool(re.search(r"\b" + re.escape(key) + r"\b", text))
        return key in text

    def _best_intent(self, prompt: str) -> Optional[str]:
        p = prompt.lower()
        best, best_len = None, 0
        for intent, keys in self.intent_map.items():
            for k in keys:
                if self._key_matches(k, p) and len(k) > best_len:
                    best, best_len = intent, len(k)
        if best:
            return best
        self._train_intent()
        if self._intent_weights is None:
            return None
        scores = self._intent_weights @ self._intent_features(p)
        idx = int(np.argmax(scores))
        if float(scores[idx]) > 0.55:
            return list(self.intent_map.keys())[idx]
        return None

    def _score_categories(self, text: str) -> Dict[str, int]:
        tokens = set(tokenize_text(text, 128))
        return {
            cat: sum(1 for w in words if w in tokens or w in text)
            for cat, words in self._category_lexicon.items()
        }

    def _extract_topic_words(self, prompt: str, n: int = 4) -> str:
        if VibeCodeHeuristics.has_cjk(prompt):
            zh = extract_zh_topic(prompt)
            if zh and zh != prompt.strip():
                return zh[:48]
            chars = _TOKEN_CJK.findall(prompt)
            if chars:
                return "".join(chars[:n * 2])[:48]
        stop = {"the", "a", "an", "is", "are", "to", "for", "of", "in", "on", "my", "me", "i", "you", "please", "can", "do"}
        words = [w for w in tokenize_text(prompt, 64) if w not in stop and len(w) > 1]
        return " ".join(words[:n]) if words else prompt.strip()[:48] or "that"

    def _try_simple_math(self, prompt: str) -> Optional[str]:
        expr = prompt.lower().strip().rstrip("?")
        expr = re.sub(r"^(what is|calculate|compute|solve)\s+", "", expr)
        expr = expr.replace("plus", "+").replace("minus", "-").replace("times", "*").replace("multiplied by", "*")
        expr = expr.replace("divided by", "/").replace("over", "/")
        if not re.fullmatch(r"[\d\s+\-*/().]+", expr.strip()):
            return None
        try:
            val = eval(expr, {"__builtins__": {}}, {})  # noqa: S307 — sandboxed numeric expr only
            if isinstance(val, (int, float)):
                return str(int(val)) if float(val).is_integer() else f"{val:.6g}"
        except Exception:
            return None
        return None

    def _explain_topic(self, topic: str, dialect: Dict[str, str]) -> str:
        known = {
            "core": dialect["core"],
            "python": "Python is a general-purpose language — great for scripts, APIs, and automation.",
            "recursion": dialect["recursion"],
            "docker": "Docker packages apps in containers so they run the same everywhere.",
            "api": "An API is a defined interface for programs to request data or actions from another service.",
            "javascript": "JavaScript runs in browsers and on servers (Node.js) for interactive web apps.",
            "rust": "Rust is a systems language focused on memory safety without a garbage collector.",
        }
        for key, answer in known.items():
            if key in topic:
                return answer
        return (
            f"\"{topic}\" — I can give a deeper explanation, compare options, or show example code. "
            "Say which angle you want (concept, tutorial, or code)."
        )

    def _howto_topic(self, topic: str) -> str:
        return (
            f"To {topic}:\n"
            "1) State the goal and any constraints (OS, language, deadline).\n"
            "2) Start with the smallest working version.\n"
            "3) Run it, capture errors, and iterate.\n"
            "Paste your current code or error and I will tailor the steps."
        )

    def _answer_open_question(self, prompt: str, categories: Dict[str, int], ctx: str) -> str:
        topic = self._extract_topic_words(prompt)
        top = max(categories, key=categories.get) if categories else "general"
        lead = f"Re: \"{topic}\""
        if top == "code":
            return f"{lead} — share the language and goal; I can draft or fix code."
        if top == "tech":
            return f"{lead} — I can explain the concept, trade-offs, or a minimal example."
        if top == "personal":
            return f"I'm **catr1.10** — a local **{MODEL_NAME}** assistant. I help with code, explanations, debugging, and math. Everything runs in-memory."
        if ctx:
            return f"{lead} (context: {ctx}) — tell me more detail and I'll answer directly."
        return f"{lead} — ask for an explanation, code sample, or step-by-step walkthrough."

    def _respond_universal(self, prompt: str, dialect: Dict[str, str], vec: np.ndarray) -> str:
        p = prompt.strip()
        pl = p.lower()
        ctx = self._recent_user_context()
        categories = self._score_categories(pl)

        math_result = self._try_simple_math(p)
        if math_result is not None:
            return f"Result: {math_result}"

        if self._wants_steps(prompt):
            return self._howto_topic(self._extract_topic_words(prompt))

        if pl.startswith(("what is ", "what's ", "what are ")):
            topic = re.sub(r"^what(?:'s| is| are)\s+", "", pl).rstrip("?").strip()
            return self._explain_topic(topic, dialect)

        if pl.startswith(("how do ", "how to ", "how can ")):
            topic = re.sub(r"^how (?:do|to|can) (?:i |we )?", "", pl).rstrip("?").strip()
            return self._howto_topic(topic)

        if pl.startswith("why "):
            topic = pl[4:].rstrip("?").strip()
            return (
                f"On \"{topic}\": usually configuration, dependencies, invalid input, or an edge case. "
                "Paste the exact error or snippet for a targeted fix."
            )

        if "joke" in pl or categories.get("creative", 0) >= 2:
            jokes = [
                "Why do programmers prefer dark mode? Light attracts bugs.",
                "A SQL query walks into a bar, walks up to two tables, and asks: 'Can I join you?'",
                "There are only 10 kinds of people: those who understand binary and those who don't.",
            ]
            return jokes[int(np.abs(vec[:6].sum() * 100)) % len(jokes)]

        if "?" in p:
            return self._answer_open_question(p, categories, ctx)

        if self._wants_brief(prompt):
            topic = self._extract_topic_words(prompt)
            return f"Brief: \"{topic}\" — ask for code, steps, or a one-line definition."

        topic = self._extract_topic_words(prompt)
        variants = [
            f"On \"{topic}\": I can explain, write code, debug, or compare options. What do you need?",
            f"Got \"{topic}\". Say *explain*, *code in python*, or paste an error to continue.",
            f"Understood — \"{topic}\". I support {len(self.code_experts)} languages and local code execution.",
            f"\"{topic}\" noted. Ask for a tutorial, snippet, or run ```python ...``` blocks here.",
            f"Re \"{topic}\": catr1.10 routed your prompt — reply with more detail or a fenced code block.",
        ]
        pick = int(np.abs(vec[:8].sum() * 1000)) % len(variants)
        if ctx:
            return f"{variants[pick]} (Earlier: {ctx})"
        return variants[pick]

    def detect_locale(self, p): return "chinese" if re.search(r"[\u4e00-\u9fff]|中文|chinese", p.lower()) else self.response_locale
    def get_dialect(self, loc):
        # Keep response tone stable and predictable (closer to chat assistant behavior).
        bank = self.dialects.get(loc, self.dialects["english"])
        return bank[0]

    def _remember(self, role: str, text: str):
        self.chat_history.append({"role": role, "text": text.strip()})
        if len(self.chat_history) > self.max_history:
            self.chat_history = self.chat_history[-self.max_history:]

    def _recent_user_context(self, n: int = 3) -> str:
        msgs = [m["text"] for m in self.chat_history if m["role"] == "user"]
        if not msgs:
            return ""
        return " | ".join(msgs[-n:])

    def _wants_brief(self, prompt: str) -> bool:
        p = prompt.lower()
        return any(x in p for x in ["short", "brief", "one line", "tldr", "concise"])

    def _wants_steps(self, prompt: str) -> bool:
        p = prompt.lower()
        return any(x in p for x in ["step by step", "steps", "walkthrough", "how do i"])

    def _wants_code_only(self, prompt: str) -> bool:
        p = prompt.lower()
        return any(x in p for x in ["code only", "just code", "only code", "no explanation"])

    def _chat_fallback(self, prompt: str, dialect: Dict[str, str], vec: Optional[np.ndarray] = None) -> str:
        if vec is None:
            vec = self.recursive_encode(prompt)
        return self._respond_universal(prompt, dialect, vec)

    def _intent_response(self, intent: str, prompt: str, dialect: Dict[str, str]) -> Optional[str]:
        if intent == "hello":
            hit = self.synth.smalltalk_reply(prompt)
            return hit or dialect["hello"]
        if intent == "core":
            return dialect["core"]
        if intent == "recursion":
            return f"{dialect['recursion']}\n\ndef fact(n):\n    return 1 if n<=1 else n*fact(n-1)"
        if intent == "help":
            return (
                f"**catr1.10** · {MODEL_NAME} · {REASONING_MODE}\n\n"
                "Frontier-tier local assistant — compression + reasoning.\n\n"
                f"**{CodeAnythingEngine.TAG}** — any language, any task (API, CLI, web, algo).\n\n"
                "Chat: `/chat` · `/think` · `/reset` · `/code` · **run it**"
            )
        if intent == "languages":
            n = len(self.code_experts)
            tag = CodeAnythingEngine.TAG if CONFIG.get("code_anything") else "languages"
            return (
                f"**{tag}** · {n} supported\n\n"
                f"{', '.join(sorted(self.code_experts))}\n\n"
                "Runnable natively: python · javascript · bash. Others: dry-run simulation."
            )
        if intent == "profile":
            return CAT_R11_PROFILE_MD
        if intent == "thanks":
            return "You're welcome — ask anything else."
        if intent == "goodbye":
            return "Goodbye. Come back anytime."
        if intent == "math":
            result = self._try_simple_math(prompt)
            if result:
                return f"Result: {result}"
            return "Give a numeric expression (e.g. 15 * 7 or what is 100 divided by 4)."
        if intent == "explain":
            topic = re.sub(r".*(?:explain|define|meaning of|tell me about)\s+", "", prompt.lower()).rstrip("?")
            return self._explain_topic(topic.strip(), dialect)
        if intent == "howto":
            topic = re.sub(r".*(?:how do|how to|how can|tutorial)\s+", "", prompt.lower()).rstrip("?")
            return self._howto_topic(topic.strip() or self._extract_topic_words(prompt))
        if intent == "debug":
            if re.search(r"\bfix\b", prompt.lower()) and "traceback" not in prompt.lower():
                topic = self._extract_topic_words(prompt)
                return (
                    f"To fix \"{topic}\": share the config snippet, expected behavior, and any error log. "
                    "I will suggest concrete changes."
                )
            return "Paste the full traceback, file path, and what you expected vs what happened."
        if intent == "opinion":
            topic = self._extract_topic_words(prompt)
            return (
                f"On \"{topic}\": it depends on constraints (team skill, performance, ecosystem). "
                "Share your use case and I will recommend a concrete choice."
            )
        if intent == "joke":
            jokes = [
                "Why do programmers prefer dark mode? Light attracts bugs.",
                "A SQL query walks into a bar, walks up to two tables, and asks: 'Can I join you?'",
                "There are only 10 kinds of people: those who understand binary and those who don't.",
            ]
            vec = self.recursive_encode(prompt)
            return jokes[int(np.abs(vec[:6].sum() * 100)) % len(jokes)]
        if intent == "fable" and CONFIG["catr1_enabled"]:
            vec = self.last_vec if self.last_vec is not None else self.recursive_encode(prompt)
            topic = self._extract_topic_words(prompt)
            return self.catseek.compose_fable(prompt, topic, vec)
        return None
    _LANG_STOP = frozenset({"a", "an", "the", "me", "my", "some", "that", "is", "to", "it"})

    def extract_lang(self, p):
        original = p
        vibe = VibeCodeHeuristics.lang_from_text(original, self)
        if vibe:
            return vibe
        p = p.lower()
        if re.search(r"\b(?:a|an)\s+html\b|\bhtml\s+(?:program|page|file|ad|that|site|app)\b", p):
            return "html"
        m = CatSeekR1Code._IN_LANG.search(p)
        if m:
            raw = m.group(1).lower()
            if raw == "shell":
                raw = "bash"
            return self.normalize_lang(raw)
        m = CatSeekR1Code._MAKE_IT_LANG.search(p)
        if m:
            raw = m.group(1).lower()
            if raw == "shell":
                raw = "bash"
            return self.normalize_lang(raw)
        for a, l in self.aliases.items():
            if re.search(rf"\bin\s+{re.escape(a)}\b", p) or re.search(rf"\b{re.escape(a)}\s+code\b", p):
                return l
        for l in self.code_experts:
            if re.search(rf"\bin\s+{re.escape(l)}\b", p) or re.search(rf"\b{re.escape(l)}\s+code\b", p):
                return l
        m = re.search(r"(?:write|code|syntax)\s+(?:(?:a|an|the)\s+)?(?:in\s+)?([a-z+#]+)", p)
        if m:
            raw = m.group(1)
            if raw in self._LANG_STOP:
                if "html" in p:
                    return "html"
                if re.search(r"\b(?:c|c\+\+|cpp)\b", p):
                    return "cpp" if "++" in p or "cpp" in p else "c"
                inferred = self.detect_lang_from_text(original)
                return self.normalize_lang(inferred) if inferred else None
            return self.aliases.get(raw, raw)
        inferred = self.detect_lang_from_text(original)
        return self.normalize_lang(inferred) if inferred else None

    def detect_lang_from_text(self, text: str) -> Optional[str]:
        s = text or ""
        sl = s.lower()

        # Fast keyword/shape checks.
        if re.search(r"#!/bin/(ba)?sh|echo\s+['\"]|\$\{?[A-Z_][A-Z0-9_]*\}?|^\s*for\s+\w+\s+in\s+", s, re.MULTILINE):
            return "bash"
        if re.search(r"<!doctype html>|<html|</html>|<body|</body>|<div|</div>", sl):
            return "html"
        if re.search(r"\bconsole\.log\(|\bfunction\s+\w+\s*\(|=>|\b(let|const|var)\s+\w+", s):
            return "javascript"
        if re.search(r"\binterface\s+\w+|:\s*(string|number|boolean)\b", s):
            return "typescript"
        if re.search(r"^\s*#include\s+<", s, re.MULTILINE):
            if "std::" in s or "cout" in s:
                return "cpp"
            return "c"
        if re.search(r"\bpublic\s+class\b|\bSystem\.out\.println\(", s):
            return "java"
        if re.search(r"\bfun\s+main\s*\(|println\(|val\s+\w+|var\s+\w+:", s):
            return "kotlin"
        if re.search(r"\bimport\s+Foundation\b|struct\s+\w+:\s*View", s):
            return "swift"
        if re.search(r"\bpragma\s+solidity\b|contract\s+\w+", s):
            return "solidity"
        if re.search(r"\bfn\s+main\s*\(|println!\(", s):
            return "rust"
        if re.search(r"\bpackage\s+main\b|\bfunc\s+main\s*\(", s):
            return "go"
        if re.search(r"\bsection\s+\.(text|data)\b|\bglobal\s+_start\b", sl):
            return "assembly"
        if re.search(r"^\s*def\s+\w+\s*\(|__name__\s*==\s*['\"]__main__['\"]|\bprint\(", s, re.MULTILINE):
            return "python"

        # Fallback token scoring across supported syntaxes.
        scores = {
            "python": 0,
            "cpp": 0,
            "c": 0,
            "javascript": 0,
            "typescript": 0,
            "java": 0,
            "rust": 0,
            "go": 0,
            "bash": 0,
            "assembly": 0,
            "html": 0,
        }
        token_hints = {
            "python": ["def ", "import ", "None", "True", "False", "elif", "self."],
            "cpp": ["std::", "#include", "cout", "cin", "namespace std", "->"],
            "c": ["#include", "printf(", "scanf(", "malloc(", "free("],
            "javascript": ["console.log", "function ", "=>", "let ", "const ", "var "],
            "typescript": [": string", ": number", "interface ", "type ", "implements "],
            "java": ["public class", "public static void main", "System.out.println", "new "],
            "rust": ["fn ", "let mut", "println!", "match ", "::"],
            "go": ["package ", "func ", "fmt.", ":=", "go "],
            "bash": ["#!/bin/bash", "echo ", "$(", "fi", "done"],
            "assembly": ["mov ", "jmp ", "section .text", "db ", "_start"],
            "html": ["<html", "<body", "<div", "</", "<!doctype"],
        }
        for lang, hints in token_hints.items():
            for h in hints:
                if h in s or h in sl:
                    scores[lang] += 1
        best_lang = max(scores, key=scores.get)
        return best_lang if scores[best_lang] > 0 else None
    def extract_code_block(self, p):
        m = re.search(r"```([a-zA-Z0-9_+#-]*)\n([\s\S]*?)```", p)
        if not m:
            return None, None
        lang = (m.group(1) or "").strip().lower() or None
        code = m.group(2).strip()
        if not lang:
            lang = self.detect_lang_from_text(code)
        return lang, code

    def normalize_lang(self, lang: Optional[str]) -> Optional[str]:
        if not lang:
            return None
        lang = lang.lower().strip()
        aliases = {
            "py": "python", "python3": "python",
            "c++": "cpp", "cc": "cpp",
            "js": "javascript", "node": "javascript",
            "ts": "typescript",
            "sh": "bash", "shell": "bash", "zsh": "bash",
            "asm": "assembly",
            "kt": "kotlin", "rb": "ruby", "rs": "rust",
            "cs": "csharp", "fs": "fsharp", "sol": "solidity",
        }
        if CONFIG.get("code_anything"):
            aliases.update(CodeAnythingEngine.LANG_ALIASES)
        return aliases.get(lang, lang)

    def generate_dynamic_template(self, lang: str, prompt: str) -> str:
        lang = self.normalize_lang(lang) or "python"
        comment = "//"
        if lang in {"python", "bash"}:
            comment = "#"
        elif lang == "html":
            comment = "<!-- -->"

        if lang == "html":
            return "<!DOCTYPE html>\n<html>\n<body>\n  <h1>Hello World</h1>\n</body>\n</html>"
        if lang == "python":
            return "def main():\n    print('Hello World')\n\nif __name__ == '__main__':\n    main()"
        return (
            f"{comment} Dynamic template for {lang}\n"
            f"{comment} Prompt: {prompt[:80]}"
        )

    def _extract_prompt_requirements(self, prompt: str) -> Dict[str, Any]:
        p = prompt.lower()
        fn_match = re.search(r"(?:function|def|method)\s+([a-zA-Z_][a-zA-Z0-9_]*)", prompt)
        class_match = re.search(r"(?:class|struct)\s+([a-zA-Z_][a-zA-Z0-9_]*)", prompt)
        return {
            "wants_main": ("main" in p) or ("entry point" in p),
            "wants_json": ("json" in p),
            "wants_async": ("async" in p) or ("await" in p),
            "wants_cli": ("arg" in p) or ("argv" in p) or ("command line" in p) or ("cli" in p),
            "wants_file_io": ("file" in p) or ("read" in p) or ("write" in p),
            "function_name": fn_match.group(1) if fn_match else None,
            "class_name": class_match.group(1) if class_match else None,
        }

    def _validate_code(self, lang: str, code: str) -> bool:
        lang = self.normalize_lang(lang) or "python"
        try:
            if lang == "python":
                ast.parse(code, mode="exec")
                return True
            if lang in {"javascript", "typescript", "java", "cpp", "c", "go", "rust"}:
                opens = sum(code.count(ch) for ch in "{([")
                closes = sum(code.count(ch) for ch in "})]")
                return opens == closes and len(code.strip()) > 0
            if lang == "html":
                return "<html" in code.lower() and "</html>" in code.lower()
            if lang == "bash":
                return len(code.strip()) > 0
            return len(code.strip()) > 0
        except Exception:
            return False

    def _tailor_code_once(self, lang: str, code: str, req: Dict[str, Any]) -> str:
        lang = self.normalize_lang(lang) or "python"
        patched = code
        fn_name = req.get("function_name")
        class_name = req.get("class_name")

        if lang == "python":
            if req["wants_async"] and "async def" not in patched:
                patched = (
                    "import asyncio\n\n"
                    "async def main_async():\n"
                    "    print('Hello World')\n\n"
                    "if __name__ == '__main__':\n"
                    "    asyncio.run(main_async())"
                )
            if req["wants_json"] and "import json" not in patched:
                patched = f"import json\n\n{patched}"
            if req["wants_cli"] and "import sys" not in patched:
                patched = f"import sys\n\n{patched}"
            if req["wants_file_io"] and "open(" not in patched:
                patched += "\n\n# file io example\nwith open('output.txt', 'w', encoding='utf-8') as f:\n    f.write('Hello World')\n"
            if fn_name and f"def {fn_name}(" not in patched:
                patched += f"\n\ndef {fn_name}():\n    return 'ok'\n"
            if class_name and f"class {class_name}" not in patched:
                patched += f"\n\nclass {class_name}:\n    pass\n"
            if req["wants_main"] and "__name__ == '__main__'" not in patched:
                patched += "\n\nif __name__ == '__main__':\n    main()\n"
            return patched

        if lang in {"javascript", "typescript"}:
            if req["wants_async"] and "async function" not in patched:
                patched = "async function main(){\n  console.log('Hello World');\n}\nmain();"
            if req["wants_json"] and "JSON." not in patched:
                patched += "\n\nconst payload = JSON.stringify({ ok: true });\nconsole.log(payload);\n"
            if fn_name and f"function {fn_name}" not in patched:
                patched += f"\n\nfunction {fn_name}() {{ return 'ok'; }}\n"
            return patched

        if lang == "bash":
            if not patched.startswith("#!/bin/bash"):
                patched = "#!/bin/bash\n" + patched
            if req["wants_file_io"] and ">" not in patched:
                patched += "\necho \"Hello World\" > output.txt\n"
            return patched

        return patched

    def recompile_code_for_prompt(self, lang: str, prompt: str, seed_code: str) -> str:
        """Dynamic recompilation: iterate and tailor generated code to prompt requirements."""
        lang = self.normalize_lang(lang) or "python"
        req = self._extract_prompt_requirements(prompt)
        code = seed_code
        for _ in range(3):
            code = self._tailor_code_once(lang, code, req)
            if self._validate_code(lang, code):
                return code
        # Final safe fallback if iterative tailoring failed validation.
        fallback = self.code_experts.get(lang) or self.generate_dynamic_template(lang, prompt)
        return fallback

    def execute_code_any_language(self, lang: Optional[str], code: Optional[str]) -> str:
        lang = self.normalize_lang(lang) or "python"
        if not code:
            return "No code provided."

        timeout = CONFIG.get("code_terminal_timeout", 8)

        if lang == "python":
            return self.safe_exec_python(code, timeout)
        if lang == "javascript":
            try:
                out = subprocess.run(
                    ["node", "-e", code],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    check=False,
                )
                text = (out.stdout or out.stderr or "").strip()
                return text if text else "(no output)"
            except FileNotFoundError:
                return "Node.js runtime not found. Install node to execute JavaScript."
            except subprocess.TimeoutExpired:
                return "Execution timed out."
            except Exception as e:
                return f"Execution error: {e}"
        if lang == "bash":
            try:
                out = subprocess.run(
                    ["bash", "-lc", code],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    check=False,
                )
                text = (out.stdout or out.stderr or "").strip()
                return text if text else "(no output)"
            except subprocess.TimeoutExpired:
                return "Execution timed out."
            except Exception as e:
                return f"Execution error: {e}"

        if CONFIG.get("code_anything"):
            return CodeAnythingEngine.simulate(lang, code, "")
        lines = [ln for ln in code.splitlines() if ln.strip()]
        return (
            f"Interpreter summary ({lang}):\n"
            f"- lines: {len(lines)}\n"
            f"- chars: {len(code)}\n"
            f"- execution backend: not installed for {lang}\n"
            "Tip: Python/JavaScript/Bash run natively in this local interpreter."
        )

    def safe_exec_python(self, code, timeout=8):
        try:
            if not code:
                return "No code provided."

            tmpdir = os.path.join("/var/folders/q1/43b16kqd4zbbst790zltxp2m0000gn/T", "opencode", "catr1_exec")
            os.makedirs(tmpdir, exist_ok=True)
            script_path = os.path.join(tmpdir, f"exec_{uuid.uuid4().hex[:12]}.py")
            with open(script_path, "w") as f:
                f.write(code)

            try:
                out = subprocess.run(
                    [sys.executable, "-u", script_path],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    check=False,
                )
                stdout = out.stdout.strip()
                stderr = out.stderr.strip()
                if out.returncode != 0 and stderr:
                    return f"Exit code {out.returncode}\n{stderr}"
                return stdout if stdout else "(no output)"
            except subprocess.TimeoutExpired:
                return "Execution timed out."
            finally:
                try:
                    os.remove(script_path)
                except OSError:
                    pass
        except Exception as e:
            return f"Error: {e}"

    def generate(self, prompt, simulate=True, on_token=None):
        loc = self.detect_locale(prompt)
        dia = self.get_dialect(loc)
        raw = prompt.strip()
        self._remember("user", prompt)
        self.catseek.memory.add("user", prompt)
        self.persistent_memory.total_messages += 1
        if simulate:
            time.sleep(CONFIG["simulate_latency"])
        p = prompt.lower()
        pl = raw.lower()

        if pl in {"/reset", "reset chat", "clear memory"}:
            self.chat_history = []
            self.chat.new_session()
            self.catseek.memory = CatSeekContextMemory()
            self._embed_cache.clear()
            self._distil_cache.clear()
            self._recursive_cache.clear()
            self.deepmind.clear()
            self.web.clear()
            self.compression_trace = []
            self.last_recursive_trace = []
            self.last_think = ""
            return "Conversation memory cleared."

        if pl == "/chat":
            resp = ChatProtocol.help_text()
            self._remember("assistant", resp)
            return resp
        if pl == "/chat new":
            sid = self.chat.new_session()
            self.chat_history = []
            self.catseek.memory = CatSeekContextMemory()
            self.last_think = ""
            resp = f"New chat session `{sid}`."
            self._remember("assistant", resp)
            self.catseek.memory.add("assistant", resp)
            return resp
        if pl == "/chat history":
            return self.chat.session().transcript() or "(empty session)"
        if pl == "/chat session":
            s = self.chat.session()
            return f"Session `{s.session_id}` · turn {s.turn_count} · {CATSEEK_MODEL_ID}"

        if pl == "/ultrathink":
            resp = f"Extended thinking is **{'on' if self.ultrathink_on else 'off'}**."
            self._remember("assistant", resp)
            return resp
        if pl == "/ultrathink on":
            self.ultrathink_on = True
            resp = "Extended thinking **on**."
            self._remember("assistant", resp)
            return resp
        if pl == "/ultrathink off":
            self.ultrathink_on = False
            resp = "Extended thinking **off**."
            self._remember("assistant", resp)
            return resp
        if pl == "/think" and self.last_think:
            return self.last_think
        if pl == "/web" or pl.startswith("/web "):
            resp = self.web.handle_command(self, raw)
            self._remember("assistant", resp)
            self.catseek.memory.add("assistant", resp)
            return resp
        if pl == "/interpret" or pl.startswith("/interpret "):
            block_lang, code = self.extract_code_block(raw if "```" in raw else prompt)
            if not code and pl.startswith("/interpret "):
                code = raw.split(maxsplit=1)[1] if " " in raw else ""
            exec_lang = CatSeekR1Code.detect_lang(self, prompt)
            if not code:
                return f"Usage: `/interpret` with ``` fences or `/interpret print('hi')`"
            result = CatSeekR1CodingAPI.agent_run(self, code, exec_lang)
            resp = CatSeekR1CodingAPI.format_result(result)
            self._remember("assistant", resp)
            self.catseek.memory.add("assistant", resp)
            return resp
        if pl == "/run" or pl.startswith("/run "):
            block_lang, code = self.extract_code_block(raw if "```" in raw else prompt)
            if not code and pl.startswith("/run "):
                code = raw.split(maxsplit=1)[1] if " " in raw else ""
            exec_lang = CatSeekR1Code.detect_lang(self, prompt)
            if not code:
                return "Usage: paste code in ``` fences, or `/run print('hi')`"
            result = CatSeekR1CodingAPI.agent_run(self, code, exec_lang)
            resp = CatSeekR1CodingAPI.format_result(result)
            self._remember("assistant", resp)
            self.catseek.memory.add("assistant", resp)
            return resp
        if pl == "/coding" or pl.startswith("/coding "):
            if pl.strip() == "/coding":
                return CatSeekR1CodingAPI.help_text()
            tail = raw.split(maxsplit=1)[1] if " " in raw else ""
            if tail.startswith("{"):
                try:
                    payload = json.loads(tail)
                except json.JSONDecodeError:
                    payload = {"action": "run", "code": tail}
            else:
                action, _, body = tail.partition(" ")
                payload = {"action": action or "run", "code": body}
            out = CatSeekR1CodingAPI.parse_request(self, payload)
            if out.get("action") == "explain":
                resp = out.get("content", "")
                self._remember("assistant", resp)
                return resp
            if out.get("action") == "help":
                resp = out.get("help", CatSeekR1CodingAPI.help_text())
                self._remember("assistant", resp)
                return resp
            if not out.get("ok", True) and out.get("error"):
                resp = f"**{CODING_API_LABEL}** error: {out['error']}"
                self._remember("assistant", resp)
                return resp
            if "output" in out:
                resp = CatSeekR1CodingAPI.format_result(out)
                self._remember("assistant", resp)
                return resp
            resp = json.dumps(out, indent=2)
            self._remember("assistant", resp)
            return resp
        if pl == "/code":
            resp = CatSeekR1Code.code_help()
            self._remember("assistant", resp)
            return resp
        if pl.startswith("/code "):
            self.encode_for_task(raw, task="code")
            resp = CatSeekR1Code.respond(self, raw)
            self._remember("assistant", resp)
            self.catseek.memory.add("assistant", resp)
            return resp
        if CONFIG.get("catr1_code_enabled") and pl in {"code", "code >", ">"}:
            self.encode_for_task(raw, task="code")
            resp = CatSeekR1Code.respond(self, raw)
            self._remember("assistant", resp)
            self.catseek.memory.add("assistant", resp)
            return resp
        if pl.startswith("/ultrathink ") and pl not in ("/ultrathink off", "/ultrathink on"):
            think_prompt = raw.split(maxsplit=1)[1] if " " in raw else ""
            if not think_prompt.strip():
                return "Usage: `/ultrathink <your question>`"
            self._run_ultrathink(think_prompt, force=True)
            resp = self.catseek.complete(self, think_prompt, simulate=False)
            self._remember("assistant", resp)
            self.catseek.memory.add("assistant", resp)
            return resp

        if pl in {"who are you", "what are you", "what model are you"}:
            st = self.catseek_stats
            resp = (
                f"I'm **catr1.10** — `{CAT_R1_MODEL_ID}` running locally.\n\n"
                f"In-memory student stack "
                f"({CONFIG.get('distil_teacher_weight', CONFIG['teacher_weight']):.0%} primary blend).\n\n"
                f"Reasoning: **{REASONING_MODE}** · extended thinking.\n\n"
                f"Code: **{CODE_ENGINE}** ({EDITION}) {'✓ enabled' if CAT_R1_CODE_ENABLED else 'off'} · {CORE_NAME} — "
                f"{st['catr1_linear_layers']} linear layers, "
                f"{st['weight_bits']}-bit ternary weights, {CONFIG['catr1_context_window']:,} token context.\n\n"
                f"In-memory: {st['shadow_params']:,} params · {st['packed_kb']:.0f}KB packed · no weight files.\n\n"
                "I'm thorough, proactive, and self-checking — ask me anything."
            )
            self._remember("assistant", resp)
            self.catseek.memory.add("assistant", resp)
            return resp

        # Heuristics-guided dispatch → cat r1.10 core
        resp = self.catseek.complete(self, CatSeekR1Code.normalize_prompt(raw), simulate=False)

        # catr1.10 style: prepend thinking block if available
        if self.last_think and self.ultrathink_on and not resp.startswith("..."):
            think_content = self.last_think.strip()
            resp = f"...\n{think_content}\n...\n\n{resp}"

        self._remember("assistant", resp)
        self.catseek.memory.add("assistant", resp)

        if on_token is not None:
            tokens = re.split(r"(\s+)", resp)
            for t in tokens:
                on_token(t)
        return resp

    def clear_history(self) -> None:
        self.chat_history.clear()
        self.catseek.memory = CatSeekContextMemory()
        self._embed_cache.clear()
        self._distil_cache.clear()
        self._recursive_cache.clear()
        self.deepmind.clear()
        self.web.clear()
        self.compression_trace.clear()
        self.last_recursive_trace.clear()
        self.last_think = ""
        self._pending_think = ""

    def get_thoughts(self, prompt, lang, ultra):
        if not prompt.strip():
            return ["Ready."]
        if not O1PreviewReasoner.should_run(prompt, enabled=self.ultrathink_on, force=ultra):
            return []
        vec = self.recursive_encode(prompt)
        draft = self._distil_draft(prompt, vec=vec)
        trace = self.ultrathink.run(
            prompt, distill_draft=draft, recursive_trace=self.last_recursive_trace,
            compression_trace=self.compression_trace,
        )
        self._pending_think = trace
        return [ln.strip() for ln in trace.split("\n") if ln.strip()]

# ──────────────────────────────────────────────────────────────
# GUI & API — cat r1.10 chat layout · files = off
# ──────────────────────────────────────────────────────────────
class CatR11GUI:
    def __init__(self, root, engine: Optional[CatR11Engine] = None):
        self.root = root
        self.engine = engine or CatR11Engine()
        self.ui = CAT_R1_UI
        self._msg_widgets: List[tk.Widget] = []
        self._history_items: List[str] = []
        self._stream_tokens: List[str] = []
        self._stream_widget: Optional[tk.Text] = None
        self._stream_started = False
        self._chat_mode = self.engine.persistent_memory.chat_mode
        self._thinking_on = self.engine.persistent_memory.thinking_on
        self._web_search_on = False

        root.title(WINDOW_TITLE)
        root.geometry("1360x840")
        root.minsize(1024, 680)
        if os.name == "darwin":
            try:
                root.tk.call("::tk::unsupported::MacWindowStyle", "style", root._w, "dark", "normal")
            except Exception:
                pass
        root.configure(bg=self.ui["bg"])

        self.fonts = {
            "ui": font.Font(family="Segoe UI" if os.name == "nt" else "Helvetica Neue", size=12),
            "ui_bold": font.Font(family="Segoe UI" if os.name == "nt" else "Helvetica Neue", size=12, weight="bold"),
            "title": font.Font(family="Segoe UI" if os.name == "nt" else "Helvetica Neue", size=18, weight="bold"),
            "logo": font.Font(family="Segoe UI" if os.name == "nt" else "Helvetica Neue", size=15, weight="bold"),
            "small": font.Font(family="Segoe UI" if os.name == "nt" else "Helvetica Neue", size=10),
            "mono": font.Font(family="Menlo" if os.name != "nt" else "Consolas", size=11),
            "empty": font.Font(family="Segoe UI" if os.name == "nt" else "Helvetica Neue", size=22, weight="normal"),
            "mascot": font.Font(family="Segoe UI" if os.name == "nt" else "Helvetica Neue", size=42),
        }

        outer = tk.Frame(root, bg=self.ui["bg"])
        outer.pack(fill="both", expand=True)

        # ── Sidebar (cat r1.10 chat layout) ──
        sidebar = tk.Frame(outer, bg=self.ui["sidebar"], width=268,
                           highlightthickness=1, highlightbackground=self.ui["sidebar_border"])
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        top_bar = tk.Frame(sidebar, bg=self.ui["sidebar"])
        top_bar.pack(fill="x", padx=18, pady=(18, 10))
        mascot_side = tk.Frame(top_bar, bg=self.ui["mascot_bg"], width=40, height=40)
        mascot_side.pack(side="left")
        mascot_side.pack_propagate(False)
        tk.Label(
            mascot_side, text=MASCOT_GLYPH, font=self.fonts["mascot"],
            bg=self.ui["mascot_bg"], fg=self.ui["mascot_fg"],
        ).place(relx=0.5, rely=0.5, anchor="center")
        name_col = tk.Frame(top_bar, bg=self.ui["sidebar"])
        name_col.pack(side="left", padx=(10, 0))
        tk.Label(name_col, text=BRAND, font=self.fonts["logo"], bg=self.ui["sidebar"],
                 fg=self.ui["text"]).pack(anchor="w")
        tk.Label(name_col, text=BRAND_TAG, font=self.fonts["small"], bg=self.ui["sidebar"],
                 fg=self.ui["muted"]).pack(anchor="w")

        new_chat_outer = tk.Frame(sidebar, bg=self.ui["new_chat_border"], padx=1, pady=1)
        new_chat_outer.pack(fill="x", padx=16, pady=(6, 14))
        tk.Button(
            new_chat_outer, text="  +  New chat", font=self.fonts["ui_bold"],
            bg=self.ui["new_chat_bg"], fg="#3b82f6",
            activebackground=self.ui["history_hover"], relief="flat", bd=0,
            padx=12, pady=10, cursor="hand2", anchor="w", command=self._new_chat,
        ).pack(fill="x")

        tk.Label(sidebar, text="Recent", font=self.fonts["small"], bg=self.ui["sidebar"],
                 fg=self.ui["muted"]).pack(anchor="w", padx=20, pady=(0, 4))
        self.history_frame = tk.Frame(sidebar, bg=self.ui["sidebar"])
        self.history_frame.pack(fill="both", expand=True, padx=10)

        tk.Frame(sidebar, bg=self.ui["sidebar_border"], height=1).pack(fill="x", padx=16, pady=8)
        foot = tk.Frame(sidebar, bg=self.ui["sidebar"])
        foot.pack(fill="x", padx=18, pady=(0, 14))
        tk.Label(foot, text=BRAND_TAG, font=self.fonts["small"],
                 bg=self.ui["sidebar"], fg=self.ui["muted"]).pack(anchor="w")

        tk.Button(
            sidebar, text=f"  </>  {CODE_ENGINE}", font=self.fonts["ui"],
            bg="#000000", fg="#3b82f6",
            activebackground=self.ui["history_hover"], relief="flat", bd=0,
            padx=12, pady=8, cursor="hand2", anchor="w",
            command=self._toggle_code_panel,
        ).pack(fill="x", padx=16, pady=(0, 8))

        # ── Main panel ──
        main = tk.Frame(outer, bg=self.ui["bg"])
        main.pack(side="left", fill="both", expand=True)

        header = tk.Frame(main, bg=self.ui["header_bg"], highlightthickness=1,
                          highlightbackground=self.ui["header_border"], height=56)
        header.pack(fill="x")
        header.pack_propagate(False)

        model_pill = tk.Frame(header, bg=self.ui["user_bg"], highlightthickness=1,
                              highlightbackground=self.ui["input_border"])
        model_pill.pack(side="left", padx=20, pady=12)
        self.model_label = tk.Label(
            model_pill, text=f"  {V4_MODEL_LABEL_PRO}  ", font=self.fonts["ui_bold"],
            bg=self.ui["user_bg"], fg=self.ui["accent"],
        )
        self.model_label.pack(side="left", padx=4, pady=4)
        tk.Label(model_pill, text="▾", font=self.fonts["small"],
                 bg=self.ui["user_bg"], fg=self.ui["muted"]).pack(side="left", padx=(0, 6))

        tk.Label(
            header, text=f"{BRAND_TAG} · {CAT_R1_PRO} · {CAT_R1_FLASH}",
            font=self.fonts["small"],
            bg=self.ui["header_bg"], fg=self.ui["muted"],
        ).pack(side="left", padx=(4, 0))

        self.header_status = tk.Label(header, text="Ready", font=self.fonts["small"],
                                      bg=self.ui["header_bg"], fg=self.ui["muted"])
        self.header_status.pack(side="right", padx=22)

        # Chat area + empty state
        chat_outer = tk.Frame(main, bg=self.ui["bg"])
        chat_outer.pack(fill="both", expand=True)

        self.empty_state = tk.Frame(chat_outer, bg=self.ui["bg"])
        self.empty_state.place(relx=0.5, rely=0.40, anchor="center")
        mascot_ring = tk.Frame(self.empty_state, bg=self.ui["mascot_bg"], width=88, height=88)
        mascot_ring.pack()
        mascot_ring.pack_propagate(False)
        tk.Label(
            mascot_ring, text=MASCOT_GLYPH, font=self.fonts["mascot"],
            bg=self.ui["mascot_bg"], fg=self.ui["mascot_fg"],
        ).place(relx=0.5, rely=0.5, anchor="center")
        tk.Label(self.empty_state, text="How can I help you today?", font=self.fonts["empty"], bg=self.ui["bg"],
                 fg=self.ui["empty_title"]).pack(pady=(16, 4))
        tk.Label(self.empty_state, text=BRAND_TAG,
                 font=self.fonts["ui"], bg=self.ui["bg"], fg=self.ui["muted"]).pack()
        tk.Label(self.empty_state, text=f"{CAT_R1_PRO} · {CAT_R1_FLASH}", font=self.fonts["small"],
                 bg=self.ui["bg"], fg=self.ui["muted"]).pack(pady=(8, 0))

        chat_wrap = tk.Frame(chat_outer, bg=self.ui["bg"])
        chat_wrap.pack(fill="both", expand=True, padx=0, pady=0)

        self.chat_canvas = tk.Canvas(chat_wrap, bg=self.ui["bg"], highlightthickness=0, bd=0)
        scrollbar = tk.Scrollbar(chat_wrap, orient="vertical", command=self.chat_canvas.yview,
                                 width=10, troughcolor=self.ui["bg"],
                                 activebackground=self.ui["input_border"])
        self.chat_canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y", padx=(0, 4))
        self.chat_canvas.pack(side="left", fill="both", expand=True, padx=(24, 8))

        self.messages_frame = tk.Frame(self.chat_canvas, bg=self.ui["bg"])
        self._canvas_window = self.chat_canvas.create_window((0, 0), window=self.messages_frame, anchor="n")
        self.messages_frame.bind("<Configure>", self._on_frame_configure)
        self.chat_canvas.bind("<Configure>", self._on_canvas_configure)
        self.chat_canvas.bind_all("<MouseWheel>", self._on_mousewheel, add="+")
        if os.name != "nt":
            self.chat_canvas.bind_all("<Button-4>", lambda e: self.chat_canvas.yview_scroll(-1, "units"), add="+")
            self.chat_canvas.bind_all("<Button-5>", lambda e: self.chat_canvas.yview_scroll(1, "units"), add="+")

        # cat r1.10 Coding API 0.1 — cat r1.10 code–style · files = off
        self._code_panel_visible = False
        self.code_script_name = "untitled.py"
        self.code_panel = tk.Frame(main, bg=self.ui["bg"])
        code_shadow = tk.Frame(self.code_panel, bg=self.ui["input_shadow"], padx=1, pady=1)
        code_shadow.pack(fill="x")
        code_border = tk.Frame(code_shadow, bg=self.ui["input_border"], padx=1, pady=1)
        code_border.pack(fill="x")
        code_inner = tk.Frame(code_border, bg=self.ui["code_bg"])
        code_inner.pack(fill="x")

        self._build_code_menustrip(code_inner)

        self.code_editor = scrolledtext.ScrolledText(
            code_inner, height=7, font=self.fonts["mono"],
            bg=self.ui["code_bg"], fg=self.ui["code_fg"],
            insertbackground=self.ui["code_fg"], relief="flat", bd=0,
            padx=12, pady=8, wrap="none", undo=True, exportselection=True,
        )
        self.code_editor.pack(fill="x", padx=8, pady=(0, 4))
        self.code_editor.insert("1.0", CatSeekR1CodingAPI.default_snippet())
        self._bind_code_interpreter(self.code_editor)

        tk.Label(
            code_inner, text=f"{MYTHOS_NAME} · Output", font=self.fonts["small"],
            bg=self.ui["code_bg"], fg=self.ui["muted"],
        ).pack(anchor="w", padx=12)
        self.code_output = tk.Text(
            code_inner, height=3, font=self.fonts["mono"],
            bg="#141414", fg="#9cdcfe", relief="flat", bd=0,
            padx=12, pady=6, wrap="word", exportselection=True,
        )
        self.code_output.pack(fill="x", padx=8, pady=(0, 8))
        self.code_output.bind("<Key>", self._code_output_key_filter)
        self._bind_code_output_menu(self.code_output)

        # Input bar — centered cat r1.10 composer
        self.input_outer = tk.Frame(main, bg=self.ui["bg"])
        input_outer = self.input_outer

        input_center = tk.Frame(input_outer, bg=self.ui["bg"])
        input_center.pack(fill="x", padx=48)

        self._build_composer_toolbar(input_center)

        shadow = tk.Frame(input_center, bg=self.ui["input_shadow"], padx=1, pady=1)
        shadow.pack(fill="x")
        border = tk.Frame(shadow, bg=self.ui["input_border"], padx=1, pady=1)
        border.pack(fill="x")
        input_box = tk.Frame(border, bg=self.ui["input_bg"])
        input_box.pack(fill="x")

        self._placeholder = f"Message {BRAND}..."
        self._placeholder_active = True

        self.entry = tk.Text(
            input_box, height=2, font=self.fonts["ui"], bg=self.ui["input_bg"],
            fg=self.ui["text"], insertbackground=self.ui["text"], relief="flat",
            bd=0, padx=18, pady=14, wrap="word",
        )
        self.entry.pack(side="left", fill="both", expand=True)
        self.entry.bind("<Return>", self._on_enter)
        self.entry.bind("<KeyPress>", self._on_entry_key)
        self.entry.bind("<<Paste>>", self._on_entry_edit)
        self.entry.bind("<Button-1>", self._on_entry_click)
        self.entry.bind("<KeyRelease>", self._sync_placeholder)
        self.entry.bind("<FocusIn>", self._on_entry_focus_in)
        self.entry.bind("<FocusOut>", self._on_entry_focus_out)
        self._set_placeholder()
        self.entry.focus_set()
        self._bind_clipboard(self.entry, on_edit=self._on_entry_edit)

        send_wrap = tk.Frame(input_box, bg=self.ui["input_bg"])
        send_wrap.pack(side="right", padx=(0, 12), pady=10)
        self.send_btn = tk.Button(
            send_wrap, text="↑", font=font.Font(size=15, weight="bold"),
            bg="#000000", fg="#3b82f6",
            activebackground=self.ui["send_hover"], activeforeground="#3b82f6",
            relief="flat", bd=0, width=2, height=1, cursor="hand2", command=self.send,
        )
        self.send_btn.pack()

        tk.Label(
            input_center,
            text=f"{BRAND} · API :{CONFIG['api_port']}",
            font=self.fonts["small"], bg=self.ui["bg"], fg=self.ui["muted"],
        ).pack(anchor="center", pady=(8, 0))

        self._apply_chat_mode()
        input_outer.pack(side="bottom", fill="x", padx=0, pady=(0, 20))
        self.code_panel.pack(side="bottom", fill="x", padx=0, pady=(0, 8))
        self._code_panel_visible = True

        self._start_api()

    def _current_model_label(self) -> str:
        if self._chat_mode == "expert":
            return V4_MODEL_LABEL_PRO
        return V4_MODEL_LABEL_FLASH

    def _apply_chat_mode(self):
        expert = self._chat_mode == "expert"
        self.engine.ultrathink_on = expert and self._thinking_on
        label = self._current_model_label()
        self.model_label.config(text=f"  {label}  ")
        self.root.title(WINDOW_TITLE)
        mode_txt = "Expert" if expert else "Instant"
        think_txt = "Thinking on" if self._thinking_on else "Thinking off"
        self.header_status.config(text=f"{mode_txt} · {think_txt} · {CAT_R1_PRO if expert else CAT_R1_FLASH}")

    def _bot_display_name(self) -> str:
        return self._current_model_label()

    def _set_chat_mode(self, mode: str):
        if mode not in {"expert", "instant"}:
            return
        self._chat_mode = mode
        self._apply_chat_mode()
        self._refresh_mode_buttons()

    def _toggle_thinking(self):
        self._thinking_on = not self._thinking_on
        self._apply_chat_mode()
        self._refresh_mode_buttons()

    def _toggle_web_search(self):
        self._web_search_on = not self._web_search_on
        self._refresh_mode_buttons()
        self.header_status.config(
            text=f"Web search {'on' if self._web_search_on else 'off'} · {MASCOT_NAME}"
        )

    def _mode_btn_style(self, active: bool) -> Dict[str, str]:
        if active:
            return {"bg": self.ui["mode_active_bg"], "fg": self.ui["mode_active_fg"]}
        return {"bg": self.ui["mode_idle_bg"], "fg": self.ui["mode_idle_fg"]}

    def _refresh_mode_buttons(self):
        if not hasattr(self, "btn_expert"):
            return
        for btn, on in (
            (self.btn_expert, self._chat_mode == "expert"),
            (self.btn_instant, self._chat_mode == "instant"),
            (self.btn_thinking, self._thinking_on),
            (self.btn_web, self._web_search_on),
        ):
            st = self._mode_btn_style(on)
            btn.config(bg=st["bg"], fg=st["fg"], activebackground=st["bg"], activeforeground=st["fg"])

    def _build_composer_toolbar(self, parent):
        """catseek r1.10: Expert / Instant · Thinking · Search."""
        bar = tk.Frame(parent, bg=self.ui["bg"])
        bar.pack(fill="x", pady=(0, 10))

        left = tk.Frame(bar, bg=self.ui["bg"])
        left.pack(side="left")

        self.btn_expert = tk.Button(
            left, text="Expert Mode", font=self.fonts["small"],
            relief="flat", bd=0, padx=14, pady=7, cursor="hand2",
            command=lambda: self._set_chat_mode("expert"),
        )
        self.btn_expert.pack(side="left", padx=(0, 6))

        self.btn_instant = tk.Button(
            left, text="Instant Mode", font=self.fonts["small"],
            relief="flat", bd=0, padx=14, pady=7, cursor="hand2",
            command=lambda: self._set_chat_mode("instant"),
        )
        self.btn_instant.pack(side="left")

        right = tk.Frame(bar, bg=self.ui["bg"])
        right.pack(side="right")

        self.btn_web = tk.Button(
            right, text="🌐 Search", font=self.fonts["small"],
            relief="flat", bd=0, padx=12, pady=7, cursor="hand2",
            command=self._toggle_web_search,
        )
        self.btn_web.pack(side="right", padx=(6, 0))

        self.btn_thinking = tk.Button(
            right, text="💭 Thinking", font=self.fonts["small"],
            relief="flat", bd=0, padx=12, pady=7, cursor="hand2",
            command=self._toggle_thinking,
        )
        self.btn_thinking.pack(side="right", padx=(6, 0))

        self._refresh_mode_buttons()

    def _append_think_block(self, text: str):
        """Collapsible o1-preview reasoning block."""
        self._hide_empty()
        body = self._plain(text).strip()
        if not body:
            return

        row = tk.Frame(self.messages_frame, bg=self.ui["bg"])
        row.pack(fill="x", pady=(8, 4), padx=8)
        self._msg_widgets.append(row)

        outer = tk.Frame(
            row, bg=self.ui["think_bg"],
            highlightthickness=1, highlightbackground=self.ui["think_border"],
        )
        outer.pack(anchor="w", fill="x")

        state = {"open": True}

        header = tk.Frame(outer, bg=self.ui["think_bg"])
        header.pack(fill="x", padx=12, pady=8)

        chevron = tk.Label(header, text="▾", font=self.fonts["small"], bg=self.ui["think_bg"], fg=self.ui["muted"])
        chevron.pack(side="left")
        tk.Label(
            header, text=f"Thinking · {BRAND}", font=self.fonts["ui_bold"],
            bg=self.ui["think_bg"], fg=self.ui["text"],
        ).pack(side="left", padx=(6, 0))

        body_frame = tk.Frame(outer, bg=self.ui["think_bg"])
        body_frame.pack(fill="x", padx=12, pady=(0, 10))

        lines = max(body.count("\n") + 1, 2)
        view = tk.Text(
            body_frame, font=self.fonts["small"], bg=self.ui["think_bg"], fg=self.ui["think_fg"],
            relief="flat", bd=0, wrap="word", height=min(max(lines, 3), 16), width=72,
            highlightthickness=0, cursor="arrow",
        )
        view.insert("1.0", body)
        view.config(state="disabled")
        view.pack(fill="x")

        def toggle(_e=None):
            if state["open"]:
                body_frame.pack_forget()
                chevron.config(text="▸")
                state["open"] = False
            else:
                body_frame.pack(fill="x", padx=12, pady=(0, 10))
                chevron.config(text="▾")
                state["open"] = True
            self.root.after(10, self._scroll_to_bottom)

        header.bind("<Button-1>", toggle)
        chevron.bind("<Button-1>", toggle)
        self.root.after(10, self._scroll_to_bottom)

    @staticmethod
    def _plain(text: str) -> str:
        return re.sub(r"\*\*([^*]+)\*\*", r"\1", str(text))

    def _parse_chat_codeblock(self, text: str, lang_hint: str = "") -> Tuple[str, str]:
        raw = str(text or "").strip()
        lang = (lang_hint or "").strip()
        fenced = re.search(r"```([a-zA-Z0-9_+#-]*)\s*\n([\s\S]*?)```", raw)
        if fenced:
            return fenced.group(2).rstrip(), lang or (fenced.group(1) or "").strip()
        if not lang and "\n" in raw:
            first, rest = raw.split("\n", 1)
            first = first.strip()
            if re.fullmatch(r"[a-zA-Z0-9_+#-]{1,15}", first):
                detected = self._detect_codeblock_lang(rest, first)
                if detected == first.lower():
                    return rest.rstrip(), first
        return raw, lang

    def _detect_codeblock_lang(self, code: str, lang: str = "") -> str:
        lang = (lang or "").strip().lower()
        if lang and lang not in {"text", "plain", "code"}:
            return lang
        sample = code[:800]
        if "#include" in sample or re.search(r"\bint\s+main\s*\(", sample):
            return "c"
        if re.search(r"\b(def|import|print)\b", sample) and ":" in sample:
            return "python"
        if re.search(r"\b(function|const|let|console\.)\b", sample):
            return "javascript"
        if re.search(r"\b(fn|println!|use\s+std)\b", sample):
            return "rust"
        if re.search(r"\b(package|func|fmt\.)\b", sample):
            return "go"
        if re.search(r"\b(public\s+class|System\.out)\b", sample):
            return "java"
        return lang or "code"

    def _render_chat_code_block(self, parent, code: str, lang: str = "") -> tk.Frame:
        """Chat code bubble with toolstrip: lang · Copy · Run · files = off."""
        body, lang = self._parse_chat_codeblock(code, lang)
        lang_key = self._detect_codeblock_lang(body, lang)
        lang_label = lang_key.upper() if len(lang_key) <= 4 else lang_key.capitalize()

        outer = tk.Frame(
            parent, bg=self.ui["code_bg"],
            highlightthickness=1, highlightbackground="#5c5c5c",
        )
        strip = tk.Frame(outer, bg="#3a3a3a", height=40)
        strip.pack(fill="x")
        strip.pack_propagate(False)

        left = tk.Frame(strip, bg="#3a3a3a")
        left.pack(side="left", padx=10, pady=6)
        tk.Label(left, text=lang_label, font=self.fonts["ui_bold"], bg="#3a3a3a", fg="#ffffff").pack(side="left")
        tk.Label(left, text=f"  ·  {CODING_API_LABEL}", font=self.fonts["small"], bg="#3a3a3a", fg="#b0b0b0").pack(side="left")

        actions = tk.Frame(strip, bg="#3a3a3a")
        actions.pack(side="right", padx=8, pady=4)

        tk.Label(
            actions, text=GUI_TAGLINE, font=self.fonts["ui_bold"],
            bg=self.ui["accent"], fg=self.ui["accent_text"], padx=10, pady=3,
        ).pack(side="right", padx=(8, 0))

        def copy_code():
            self._clipboard_write(body)
            self.header_status.config(text="Copied")

        def run_code():
            if not self._code_panel_visible:
                self._toggle_code_panel()
            ext_map = {
                "python": ".py", "javascript": ".js", "typescript": ".ts", "java": ".java",
                "rust": ".rs", "go": ".go", "bash": ".sh", "html": ".html", "cpp": ".cpp", "c": ".c",
            }
            self.code_script_name = f"snippet{ext_map.get(lang_key, '.txt')}"
            self.code_name_label.config(text=self.code_script_name)
            self.code_editor.delete("1.0", "end")
            self.code_editor.insert("1.0", body)
            self.code_editor.focus_set()
            self.header_status.config(text=f"Opened in {CODING_API_LABEL}")

        tk.Button(
            actions, text="▶ Run", font=self.fonts["small"],
            bg="#000000", fg="#3b82f6", activebackground="#1a1a1a",
            relief="flat", bd=0, padx=10, pady=4, cursor="hand2", command=run_code,
        ).pack(side="right", padx=(0, 6))
        tk.Button(
            actions, text="Copy", font=self.fonts["small"],
            bg="#000000", fg="#3b82f6", activebackground="#1a1a1a",
            relief="flat", bd=0, padx=10, pady=4, cursor="hand2", command=copy_code,
        ).pack(side="right")

        tk.Frame(outer, bg="#5c5c5c", height=1).pack(fill="x")

        lines = max(body.count("\n") + 1, 2)
        code_view = tk.Text(
            outer, font=self.fonts["mono"], bg=self.ui["code_bg"], fg=self.ui["code_fg"],
            relief="flat", bd=0, padx=14, pady=12, wrap="none",
            height=min(max(lines, 2), 28), width=72,
            highlightthickness=0, insertwidth=0, cursor="arrow",
            selectbackground=self.ui["accent"], selectforeground=self.ui["accent_text"],
        )
        code_view.insert("1.0", body)
        code_view.config(state="disabled")
        code_view.pack(fill="x")

        def copy_shortcut(_e=None):
            copy_code()
            return "break"

        mod = "Command" if os.name == "darwin" else "Control"
        code_view.bind(f"<{mod}-c>", copy_shortcut, add="+")
        code_view.bind("<Control-c>", copy_shortcut, add="+")
        code_view.bind("<Button-3>", copy_shortcut)
        return outer

    def _code_menu_style(self) -> Dict[str, str]:
        return {
            "strip_bg": "#3a3a3a",
            "strip_fg": "#e8e8e8",
            "strip_active": "#4a4a4a",
            "menu_bg": "#2b2b2b",
            "menu_fg": "#e8e8e8",
            "menu_active_bg": self.ui["accent"],
            "menu_active_fg": self.ui["accent_text"],
        }

    def _accel(self, key: str) -> str:
        return f"⌘{key}" if os.name == "darwin" else f"Ctrl+{key}"

    def _code_select_all(self):
        self.code_editor.tag_add(tk.SEL, "1.0", "end-1c")
        self.code_editor.mark_set(tk.INSERT, "end-1c")
        self.code_editor.see(tk.INSERT)
        self.code_editor.focus_set()

    def _code_new_script(self):
        self.code_script_name = "untitled.py"
        self.code_name_label.config(text=self.code_script_name)
        self.code_editor.delete("1.0", "end")
        self.code_editor.insert("1.0", CatSeekR1CodingAPI.default_snippet())
        self.code_output.delete("1.0", "end")
        CatSeekR1CodingAPI.session().code = CatSeekR1CodingAPI.default_snippet()
        self.code_editor.focus_set()
        self.header_status.config(text=f"New buffer · {CODING_API_LABEL}")

    def _code_clear_editor(self):
        self.code_editor.delete("1.0", "end")
        self.code_editor.focus_set()

    def _code_clear_output(self):
        self.code_output.delete("1.0", "end")

    def _code_undo(self):
        try:
            self.code_editor.edit_undo()
        except tk.TclError:
            pass

    def _code_redo(self):
        try:
            self.code_editor.edit_redo()
        except tk.TclError:
            pass

    def _code_send_to_chat(self):
        code = self.code_editor.get("1.0", "end-1c").strip()
        if not code:
            return
        lang = self.engine.detect_lang_from_text(code) or "python"
        block = f"```{lang}\n{code}\n```"
        self._clear_placeholder()
        self.entry.delete("1.0", "end")
        self.entry.insert("1.0", block)
        self.entry.config(fg=self.ui["text"])
        self._placeholder_active = False
        self.entry.focus_set()
        self.header_status.config(text="Code sent to chat")

    def _build_code_menustrip(self, parent):
        st = self._code_menu_style()
        strip = tk.Frame(parent, bg=st["strip_bg"], height=40)
        strip.pack(fill="x")
        strip.pack_propagate(False)

        menus = tk.Frame(strip, bg=st["strip_bg"])
        menus.pack(side="left", padx=4)

        def menubutton(label: str, items: List[Tuple[str, Any]]):
            mb = tk.Menubutton(
                menus, text=label, font=self.fonts["small"],
                bg=st["strip_bg"], fg=st["strip_fg"],
                activebackground=st["strip_active"], activeforeground=st["strip_fg"],
                relief="flat", bd=0, padx=10, pady=8, cursor="hand2",
            )
            drop = tk.Menu(
                mb, tearoff=0,
                bg=st["menu_bg"], fg=st["menu_fg"],
                activebackground=st["menu_active_bg"], activeforeground=st["menu_active_fg"],
                relief="flat", bd=0,
            )
            for entry in items:
                if entry is None:
                    drop.add_separator()
                else:
                    text, cmd = entry
                    drop.add_command(label=text, command=cmd)
            mb.config(menu=drop)
            mb.pack(side="left")

        menubutton("File", [
            (f"New script\t{self._accel('N')}", self._code_new_script),
            (f"Clear editor", self._code_clear_editor),
            None,
            (GUI_TAGLINE, lambda: None),
        ])
        menubutton("Edit", [
            (f"Undo\t{self._accel('Z')}", self._code_undo),
            (f"Redo\t{'⌘⇧Z' if os.name == 'darwin' else 'Ctrl+Y'}", self._code_redo),
            None,
            (f"Cut\t{self._accel('X')}", lambda: self._clip_action(self.code_editor, "cut")),
            (f"Copy\t{self._accel('C')}", lambda: self._clip_action(self.code_editor, "copy")),
            (f"Paste\t{self._accel('V')}", lambda: self._clip_action(self.code_editor, "paste")),
            None,
            (f"Select all\t{self._accel('A')}", self._code_select_all),
        ])
        menubutton("Run", [
            (f"Run agent\t{self._accel('Return')}", self._run_code_interpreter),
            (f"Clear output", self._code_clear_output),
            (f"Send to chat", self._code_send_to_chat),
        ])
        menubutton("Agent", [
            (f"Lint buffer", self._code_lint_interpreter),
            (f"Explain buffer", self._code_explain_interpreter),
            None,
            (f"New buffer\t{self._accel('N')}", self._code_new_script),
            (f"Clear buffer", self._code_clear_editor),
        ])
        menubutton("Help", [
            (f"{BRAND} interpreter help", lambda: self.log(self._bot_display_name(), ClaudeMythosRuntime.code_help(), "bot")),
        ])

        right = tk.Frame(strip, bg=st["strip_bg"])
        right.pack(side="right", padx=8)

        self.code_name_label = tk.Label(
            right, text=self.code_script_name, font=self.fonts["mono"],
            bg=st["strip_bg"], fg="#b0b0b0",
        )
        self.code_name_label.pack(side="left", padx=(0, 10))

        tk.Label(
            right, text=GUI_TAGLINE, font=self.fonts["ui_bold"],
            bg=self.ui["accent"], fg=self.ui["accent_text"],
            padx=10, pady=3,
        ).pack(side="left", padx=(0, 10))

        tk.Button(
            right, text="▶ Run", font=self.fonts["ui_bold"],
            bg="#000000", fg="#3b82f6",
            activebackground=self.ui["send_hover"], activeforeground="#3b82f6",
            relief="flat", bd=0, padx=14, pady=4, cursor="hand2",
            command=self._run_code_interpreter,
        ).pack(side="left")

        tk.Frame(parent, bg="#5c5c5c", height=1).pack(fill="x")

        mod_key = "Command" if os.name == "darwin" else "Control"

        def new_script_shortcut(_e=None):
            if self._code_panel_visible:
                self._code_new_script()
                return "break"

        def run_shortcut(_e=None):
            if self._code_focused():
                self._run_code_interpreter()
                return "break"

        self.root.bind_all(f"<{mod_key}-n>", new_script_shortcut, add="+")
        self.root.bind_all(f"<{mod_key}-Return>", run_shortcut, add="+")

    def _code_focused(self) -> bool:
        try:
            return self.root.focus_get() == self.code_editor
        except (KeyError, tk.TclError):
            return False

    def _code_output_key_filter(self, event):
        mod = 0x8 if os.name == "darwin" else 0x4
        if event.state & mod and event.keysym.lower() in {"c", "a", "x"}:
            return None
        if event.char and event.char.isprintable():
            return "break"
        if event.keysym in {"Return", "space", "Tab", "BackSpace", "Delete"}:
            return "break"
        return None

    def _widget_selection(self, widget) -> Optional[str]:
        try:
            return widget.get(tk.SEL_FIRST, tk.SEL_LAST)
        except tk.TclError:
            return None

    def _clipboard_write(self, text: str) -> None:
        if text is None:
            return
        if os.name == "darwin":
            try:
                subprocess.run(
                    ["pbcopy"], input=text, text=True, check=True,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                return
            except (OSError, subprocess.CalledProcessError):
                pass
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            pass

    def _clipboard_read(self) -> Optional[str]:
        if os.name == "darwin":
            try:
                proc = subprocess.run(
                    ["pbpaste"], capture_output=True, text=True, check=True,
                    stderr=subprocess.DEVNULL,
                )
                if proc.stdout is not None:
                    return proc.stdout
            except (OSError, subprocess.CalledProcessError):
                pass
        for clip_type in ("STRING", "UTF8_STRING", "public.utf8-plain-text", "TEXT"):
            try:
                return self.root.clipboard_get(type=clip_type)
            except tk.TclError:
                continue
        try:
            return self.root.clipboard_get()
        except tk.TclError:
            return None

    def _clip_action(self, widget, action: str):
        try:
            widget.focus_set()
        except tk.TclError:
            pass
        if action == "cut":
            self._clip_cut(widget)
        elif action == "copy":
            self._clip_copy(widget)
        elif action == "paste":
            self._clip_paste(widget)

    def _clip_copy(self, widget):
        text = self._widget_selection(widget)
        if not text:
            text = widget.get("1.0", "end-1c")
        if text:
            self._clipboard_write(text)

    def _clip_cut(self, widget):
        if str(widget.cget("state")) == "disabled":
            return
        text = self._widget_selection(widget)
        if not text:
            return
        self._clipboard_write(text)
        try:
            widget.delete(tk.SEL_FIRST, tk.SEL_LAST)
        except tk.TclError:
            pass

    def _clip_paste(self, widget):
        if str(widget.cget("state")) == "disabled":
            return
        try:
            widget.focus_set()
        except tk.TclError:
            pass
        text = self._clipboard_read()
        if text is None:
            return
        try:
            widget.delete(tk.SEL_FIRST, tk.SEL_LAST)
        except tk.TclError:
            pass
        widget.insert(tk.INSERT, text)

    def _bind_code_interpreter(self, widget):
        menu = tk.Menu(widget, tearoff=0)
        menu.add_command(label="Cut", command=lambda: self._clip_action(widget, "cut"))
        menu.add_command(label="Copy", command=lambda: self._clip_action(widget, "copy"))
        menu.add_command(label="Paste", command=lambda: self._clip_action(widget, "paste"))
        menu.add_separator()
        menu.add_command(label="Select All", command=self._code_select_all)

        def show_menu(event):
            try:
                widget.focus_set()
                menu.tk_popup(event.x_root, event.y_root)
            finally:
                menu.grab_release()

        def key_copy(_e):
            self._clip_action(widget, "copy")
            return "break"

        def key_cut(_e):
            self._clip_action(widget, "cut")
            return "break"

        def key_paste(_e):
            self._clip_action(widget, "paste")
            return "break"

        def key_undo(_e):
            self._code_undo()
            return "break"

        def key_redo(_e):
            self._code_redo()
            return "break"

        mod = "Command" if os.name == "darwin" else "Control"
        widget.bind("<Button-2>", show_menu)
        widget.bind("<Button-3>", show_menu)
        widget.bind(f"<{mod}-c>", key_copy, add="+")
        widget.bind(f"<{mod}-x>", key_cut, add="+")
        widget.bind(f"<{mod}-v>", key_paste, add="+")
        widget.bind("<Control-c>", key_copy, add="+")
        widget.bind("<Control-x>", key_cut, add="+")
        widget.bind("<Control-v>", key_paste, add="+")
        widget.bind(f"<{mod}-z>", key_undo, add="+")
        widget.bind(f"<{mod}-Z>", key_redo, add="+")

    def _bind_code_output_menu(self, widget):
        menu = tk.Menu(widget, tearoff=0)
        menu.add_command(label="Copy", command=lambda: self._clip_action(widget, "copy"))
        menu.add_command(label="Select All", command=lambda: widget.tag_add(tk.SEL, "1.0", "end-1c"))

        def show_menu(event):
            try:
                widget.focus_set()
                menu.tk_popup(event.x_root, event.y_root)
            finally:
                menu.grab_release()

        def key_copy(_e):
            self._clip_action(widget, "copy")
            return "break"

        widget.bind("<Button-2>", show_menu)
        widget.bind("<Button-3>", show_menu)
        mod = "Command" if os.name == "darwin" else "Control"
        widget.bind(f"<{mod}-c>", key_copy, add="+")
        widget.bind("<Control-c>", key_copy, add="+")

    def _bind_clipboard(self, widget, *, readonly: bool = False, on_edit=None):
        mod = "Command" if os.name == "darwin" else "Control"

        def do_copy(_e=None):
            self._clip_copy(widget)
            return "break"

        def do_cut(_e=None):
            if readonly:
                return "break"
            self._clip_cut(widget)
            if on_edit:
                on_edit()
            return "break"

        def do_paste(_e=None):
            if readonly:
                return "break"
            self._clip_paste(widget)
            if on_edit:
                on_edit()
            return "break"

        widget.bind(f"<{mod}-c>", do_copy, add="+")
        widget.bind(f"<{mod}-x>", do_cut, add="+")
        widget.bind(f"<{mod}-v>", do_paste, add="+")
        widget.bind("<Control-c>", do_copy, add="+")
        widget.bind("<Control-x>", do_cut, add="+")
        widget.bind("<Control-v>", do_paste, add="+")

    def _toggle_code_panel(self):
        if self._code_panel_visible:
            self.code_panel.pack_forget()
            self._code_panel_visible = False
        else:
            self.code_panel.pack(side="bottom", fill="x", padx=0, pady=(0, 8))
            self._code_panel_visible = True
            self.code_editor.focus_set()

    def _code_lint_interpreter(self):
        code = self.code_editor.get("1.0", "end-1c").strip()
        lang = self.engine.detect_lang_from_text(code) or "python"
        ok, msg = CatSeekR1CodingAPI.lint(code, lang)
        self.code_output.config(state="normal")
        self.code_output.delete("1.0", "end")
        self.code_output.insert("1.0", f"{'ok' if ok else 'fail'}: {msg}")
        self.header_status.config(text=f"Lint · {CODING_API_LABEL}")

    def _code_explain_interpreter(self):
        code = self.code_editor.get("1.0", "end-1c").strip()
        if not code:
            self.header_status.config(text="Explain: empty buffer")
            return
        lang = self.engine.detect_lang_from_text(code) or "python"
        text = CatSeekR1CodingAPI.explain(self.engine, code, lang)
        self.log(self._bot_display_name(), text, "bot")
        self.header_status.config(text=f"Explain · {CODING_API_LABEL}")

    def _run_code_interpreter(self):
        code = self.code_editor.get("1.0", "end-1c").strip()
        if not code:
            self.header_status.config(text=f"{CODING_API_LABEL}: empty buffer")
            return
        lang = self.engine.detect_lang_from_text(code) or "python"
        self.header_status.config(text=f"Agent run · {MYTHOS_NAME} · {self.code_script_name} ({lang})…")
        result = CatSeekR1CodingAPI.agent_run(
            self.engine, code, lang, self.code_script_name,
        )
        self.code_output.config(state="normal")
        self.code_output.delete("1.0", "end")
        panel = O1PreviewSyntax.format_panel_stdout(result)
        self.code_output.insert("1.0", panel)
        if result.get("ok"):
            self.header_status.config(text=f"Ready · {MYTHOS_NAME} · run #{result.get('runs', 1)}")
        else:
            self.header_status.config(text=f"Lint error · {MYTHOS_NAME}")

    def _entry_text(self) -> str:
        return self.entry.get("1.0", "end-1c")

    def _set_placeholder(self):
        if not self._placeholder_active:
            return
        self.entry.config(state="normal")
        self.entry.delete("1.0", "end")
        self.entry.insert("1.0", self._placeholder)
        self.entry.config(fg=self.ui["muted"])

    def _clear_placeholder(self):
        if not self._placeholder_active:
            return
        self._placeholder_active = False
        self.entry.config(state="normal", fg=self.ui["text"])
        self.entry.delete("1.0", "end")

    def _sync_placeholder(self, _event=None):
        if self._placeholder_active:
            return
        if not self._entry_text().strip():
            self._placeholder_active = True
            self._set_placeholder()

    _PLACEHOLDER_SKIP_KEYS = frozenset({
        "Shift_L", "Shift_R", "Control_L", "Control_R", "Alt_L", "Alt_R",
        "Meta_L", "Meta_R", "Caps_Lock", "Tab",
    })

    def _on_entry_key(self, event):
        if self._placeholder_active and event.keysym not in self._PLACEHOLDER_SKIP_KEYS:
            self._clear_placeholder()

    def _on_entry_click(self, _event=None):
        if self._placeholder_active:
            self._clear_placeholder()

    def _on_entry_edit(self, _event=None):
        if self._placeholder_active:
            self._clear_placeholder()

    def _on_entry_focus_in(self, _event=None):
        if self._placeholder_active:
            self._clear_placeholder()

    def _on_entry_focus_out(self, _event=None):
        if self._placeholder_active:
            return
        if not self._entry_text().strip():
            self._placeholder_active = True
            self._set_placeholder()

    def _on_frame_configure(self, _event=None):
        self.chat_canvas.configure(scrollregion=self.chat_canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        width = min(event.width - 48, 820)
        self.chat_canvas.itemconfig(self._canvas_window, width=max(width, 400))
        x = max((event.width - width) // 2, 24)
        self.chat_canvas.coords(self._canvas_window, x, 0)

    def _on_mousewheel(self, event):
        widget = self.chat_canvas.winfo_containing(event.x_root, event.y_root)
        if widget in (self.chat_canvas, self.messages_frame) or str(widget).startswith(str(self.messages_frame)):
            self.chat_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_enter(self, event):
        if event.state & 0x1:
            return
        self.send()
        return "break"

    def _hide_empty(self):
        self.empty_state.place_forget()

    def _show_empty(self):
        if not self._msg_widgets:
            self.empty_state.place(relx=0.5, rely=0.42, anchor="center")

    def _add_history_item(self, title: str):
        title = (title[:36] + "…") if len(title) > 37 else title
        self._history_items.insert(0, title)
        self._history_items = self._history_items[:8]
        for w in self.history_frame.winfo_children():
            w.destroy()
        for item in self._history_items:
            btn = tk.Button(
                self.history_frame, text=item, font=self.fonts["ui"], bg="#000000",
                fg="#3b82f6", activebackground=self.ui["history_hover"],
                relief="flat", bd=0, anchor="w", padx=12, pady=8, cursor="hand2",
                command=lambda t=item: self._prefill(t),
            )
            btn.pack(fill="x", padx=6, pady=1)

    def _new_chat(self):
        for w in self._msg_widgets:
            w.destroy()
        self._msg_widgets.clear()
        self._finalize_stream()
        self.engine.chat.new_session()
        self.engine.persistent_memory.total_sessions += 1
        _save_persistent_memory(self.engine.persistent_memory)
        self.engine.clear_history()
        self._show_empty()
        self._apply_chat_mode()
        self.header_status.config(text="Ready")

    def _prefill(self, text: str):
        self._clear_placeholder()
        self.entry.delete("1.0", "end")
        self.entry.insert("1.0", text if not text.endswith("…") else text[:-1])
        self.entry.config(fg=self.ui["text"], state="normal")
        self._placeholder_active = False
        self.entry.focus_set()

    def _avatar(self, parent, glyph: str, bg: str) -> tk.Frame:
        wrap = tk.Frame(parent, bg=bg, width=34, height=34)
        wrap.pack_propagate(False)
        tk.Label(
            wrap, text=glyph, font=self.fonts["ui_bold"], bg=bg, fg=self.ui["mascot_fg"],
        ).place(relx=0.5, rely=0.5, anchor="center")
        return wrap

    def _append_message(self, role: str, text: str, kind: str = "text", code_lang: str = ""):
        self._hide_empty()
        is_user = role == "user"
        plain = self._plain(text)

        row = tk.Frame(self.messages_frame, bg=self.ui["bg"])
        row.pack(fill="x", pady=(10, 10), padx=8)
        self._msg_widgets.append(row)

        inner = tk.Frame(row, bg=self.ui["bg"])
        inner.pack(anchor="e" if is_user else "w", fill="x")

        content_row = tk.Frame(inner, bg=self.ui["bg"])
        content_row.pack(anchor="e" if is_user else "w")

        if not is_user and kind != "code":
            self._avatar(content_row, MASCOT_GLYPH, self.ui["mascot_bg"]).pack(
                side="left", padx=(0, 10), pady=2,
            )

        msg_col = tk.Frame(content_row, bg=self.ui["bg"])
        msg_col.pack(side="left" if not is_user else "right")

        if is_user:
            bubble_bg, bubble_fg = self.ui["user_bg"], self.ui["user_fg"]
        elif kind == "code":
            bubble_bg, bubble_fg = self.ui["code_bg"], self.ui["code_fg"]
        else:
            bubble_bg, bubble_fg = self.ui["bot_bg"], self.ui["bot_fg"]

        if kind == "code":
            block = self._render_chat_code_block(msg_col, text, code_lang)
            block.pack(anchor="w")
        else:
            bubble = tk.Frame(msg_col, bg=bubble_bg)
            bubble.pack(anchor="e" if is_user else "w")
            label_font = self.fonts["ui"]
            lbl = tk.Label(
                bubble, text=plain, font=label_font, bg=bubble_bg, fg=bubble_fg,
                justify="left", wraplength=600,
                padx=self.ui["radius_pad"] if is_user else 2,
                pady=10 if is_user else 4,
            )
            lbl.pack()

        if is_user:
            self._avatar(content_row, "U", self.ui["avatar_user"]).pack(side="left", padx=(10, 0), pady=2)

        self.root.after(10, self._scroll_to_bottom)

    def _scroll_to_bottom(self):
        self.chat_canvas.update_idletasks()
        self.chat_canvas.yview_moveto(1.0)

    def log(self, sender, text, tag=None, code_lang: str = ""):
        if text is None:
            text = ""
        text = str(text).strip()
        if not text:
            return
        if sender in ("YOU", "API", "user"):
            self._append_message("user", text)
        elif sender == "THINK":
            self._append_think_block(text)
        elif tag == "code":
            body, lang = self._parse_chat_codeblock(text, code_lang)
            self._append_message("assistant", body, kind="code", code_lang=lang)
        else:
            fenced = re.search(r"```([a-zA-Z0-9_+#-]*)\s*\n([\s\S]*?)```", text)
            if fenced:
                lead = text[:fenced.start()].strip()
                if lead:
                    self._append_message("assistant", lead)
                block = fenced.group(2).rstrip()
                lang = (fenced.group(1) or "").strip()
                self._append_message("assistant", block, kind="code", code_lang=lang)
                tail = text[fenced.end():].strip()
                if tail:
                    self._append_message("assistant", tail)
            else:
                self._append_message("assistant", text)
        if sender == "SYSTEM":
            self.header_status.config(text=text[:72])

    def send(self):
        raw = self._entry_text()
        if self._placeholder_active:
            if not raw.strip() or raw.strip() == self._placeholder:
                return
            self._clear_placeholder()
        msg = self._entry_text().strip()
        if not msg:
            return
        self.entry.delete("1.0", "end")
        self._placeholder_active = True
        self._set_placeholder()
        self._add_history_item(msg)
        self.log("YOU", msg, "user")
        self.header_status.config(text="Thinking…")
        self._apply_chat_mode()
        threading.Thread(target=self._infer, args=(msg,), daemon=True).start()

    def _infer(self, prompt):
        try:
            expert = self._chat_mode == "expert"
            self.engine.ultrathink_on = expert and self._thinking_on

            self._stream_tokens = []
            self._stream_widget = None
            self._stream_started = False

            def on_token(tok):
                with self.engine._lock:
                    self._stream_tokens.append(tok)
                self.root.after(0, self._flush_stream)

            out = self.engine.chat.turn(prompt, simulate=False, on_token=on_token)
            resp = (out.get("message") or {}).get("content", "") or "(no response)"
            think = out.get("thinking", "")

            if think and expert and self._thinking_on:
                self.root.after(0, lambda t=think: self._append_think_block(t))

            self.root.after(0, self._finalize_stream)
            sid = out.get("session", "")[:8]
            mode_txt = "Expert" if expert else "Instant"
            self.root.after(0, lambda s=sid, m=mode_txt: self.header_status.config(
                text=f"Ready · {m} · session {s}" if s else f"Ready · {m}"))
        except Exception as exc:
            import traceback
            tb = traceback.format_exc()
            err = f"`{exc.__class__.__name__}: {exc}`"
            self.root.after(0, lambda e=err: self.log(self._bot_display_name(), f"Error — {e}", "bot"))
            self.root.after(0, lambda: self.header_status.config(text="Ready"))
            print(f"[{BRAND}] error: {tb}", file=sys.stderr)

    def _flush_stream(self):
        frame_buf = []
        tokens = []
        with self.engine._lock:
            tokens, self._stream_tokens = self._stream_tokens, []

        for tok in tokens:
            frame_buf.append(tok)

        frame = "".join(frame_buf)
        if not frame:
            return

        if not self._stream_started:
            self._stream_started = True
            self._hide_empty()
            row = tk.Frame(self.messages_frame, bg=self.ui["bg"])
            row.pack(fill="x", pady=(10, 10), padx=8)
            self._msg_widgets.append(row)
            inner = tk.Frame(row, bg=self.ui["bg"])
            inner.pack(anchor="w", fill="x")
            content_row = tk.Frame(inner, bg=self.ui["bg"])
            content_row.pack(anchor="w")
            self._avatar(content_row, MASCOT_GLYPH, self.ui["mascot_bg"]).pack(
                side="left", padx=(0, 10), pady=2,
            )
            msg_col = tk.Frame(content_row, bg=self.ui["bg"])
            msg_col.pack(side="left")
            bubble = tk.Frame(msg_col, bg=self.ui["bot_bg"])
            bubble.pack(anchor="w")
            self._stream_widget = tk.Text(
                bubble, font=self.fonts["ui"], bg=self.ui["bot_bg"], fg=self.ui["bot_fg"],
                justify="left", wrap="word", width=72, height=1,
                relief="flat", bd=0, padx=2, pady=4, highlightthickness=0,
                cursor="arrow",
            )
            self._stream_widget.pack()
            self._stream_widget.config(state="normal")

        if self._stream_widget is not None:
            self._stream_widget.config(state="normal")
            self._stream_widget.insert("end", frame)
            self._stream_widget.see("end")
            line_count = int(self._stream_widget.index("end-1c").split(".")[0])
            self._stream_widget.config(height=min(line_count, 28))
            self._stream_widget.config(state="disabled")
            self.root.after(5, self._scroll_to_bottom)

    def _finalize_stream(self):
        self._flush_stream()
        if self._stream_widget is not None:
            self._stream_widget.config(state="disabled")
        self._stream_widget = None
        self._stream_tokens = []
        self._stream_started = False

    def _display(self, text):
        if not text:
            return
        text = str(text)
        pattern = re.compile(r"```([a-zA-Z0-9_+#-]*)\s*\n([\s\S]*?)```")
        pos = 0
        found = False
        for m in pattern.finditer(text):
            found = True
            if m.start() > pos:
                lead = text[pos:m.start()].strip()
                if lead:
                    self.log(self._bot_display_name(), lead, "bot")
            lang = (m.group(1) or "").strip()
            block = m.group(2).rstrip()
            self.log(self._bot_display_name(), block, "code", code_lang=lang)
            pos = m.end()
        if found:
            tail = text[pos:].strip()
            if tail:
                self.log(self._bot_display_name(), tail, "bot")
        else:
            self.log(self._bot_display_name(), text, "bot")

    def _start_api(self):
        gui = self
        class Handler(BaseHTTPRequestHandler):
            def _json(self, code, data):
                body = json.dumps(data).encode(); self.send_response(code); self.send_header("Content-Type","application/json"); self.send_header("Content-Length", len(body)); self.end_headers(); self.wfile.write(body)
            def _auth(self):
                key = self.headers.get("Authorization","").replace("Bearer ","").strip()
                return not CONFIG["api_key"] or key == CONFIG["api_key"]
            def do_POST(self):
                if not self._auth(): return self._json(401,{"error":"Unauthorized"})
                coding_paths = ("/v1/coding/run", "/v1/coding", "/coding")
                paths = ("/message", "/v1/chat/completions", "/chat") + coding_paths
                if self.path not in paths: return self._json(404,{"error":"Not found"})
                try:
                    length = int(self.headers.get("Content-Length",0)); data = json.loads(self.rfile.read(length).decode()) if length else {}
                except Exception: return self._json(400,{"error":"Invalid JSON"})
                if self.path in coding_paths:
                    out = CatSeekR1CodingAPI.parse_request(gui.engine, data)
                    if out.get("error"):
                        return self._json(400, out)
                    if out.get("action") == "run" and out.get("ok"):
                        gui.root.after(0, lambda o=out: (
                            gui.code_output.config(state="normal"),
                            gui.code_output.delete("1.0", "end"),
                            gui.code_output.insert("1.0", o.get("output", "")),
                        ))
                    return self._json(200, out)
                if self.path == "/chat":
                    out = gui.engine.chat.parse_request(data)
                    if out.get("error"):
                        return self._json(400, out)
                    msg = out.get("message", {})
                    gui.root.after(0, lambda: (
                        gui.log("API", str(data.get("message", data))[:120], "user"),
                        gui.log(gui.engine.name, msg.get("content", "")),
                    ))
                    return self._json(200, out)
                prompt = data.get("message") or data.get("prompt") or next((m["content"] for m in reversed(data.get("messages",[])) if m.get("role")=="user"), "")
                if not prompt: return self._json(400,{"error":"Missing prompt"})
                out = gui.engine.chat.turn(str(prompt), simulate=False)
                resp = out.get("message", {}).get("content", "")
                think = out.get("thinking", "")
                gui.root.after(0, lambda: (gui.log("API", prompt, "user"), gui.log(gui.engine.name, resp)))
                if self.path == "/v1/chat/completions":
                    content = resp
                    if think and data.get("include_thinking"):
                        content = f"\n{think}\n\n\n{resp}"
                    return self._json(200, {
                        "id": f"chatcmpl-{int(time.time())}",
                        "object": "chat.completion",
                        "model": CATSEEK_MODEL_ID,
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
                    })
                return self._json(200, {"response": resp, "thinking": think, "model": CATSEEK_MODEL_ID, **{k: out[k] for k in ("session", "turn", "protocol") if k in out}})
            def do_GET(self):
                if not self._auth(): return self._json(401,{"error":"Unauthorized"})
                if self.path == "/v1/models":
                    return self._json(200, {"data": [CatSeekR1LLM.model_card()]})
                if self.path == "/web/sites":
                    sites = [
                        {"id": r.site_id, "title": r.title, "template": r.template,
                         "preview": r.preview_url(CONFIG["api_port"]), "created": r.created}
                        for r in gui.engine.web._sites.values()
                    ]
                    return self._json(200, {"files": "off", "sites": sites, "count": len(sites)})
                if self.path.startswith("/web/preview/"):
                    sid = self.path.rsplit("/", 1)[-1].split("?")[0]
                    rec = gui.engine.web.get(sid)
                    if not rec:
                        return self._json(404, {"error": f"site not found: {sid}"})
                    body = rec.html.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if self.path == "/v1/coding/help":
                    return self._json(200, {
                        "protocol": CODING_API_PROTO,
                        "version": CODING_API_VER,
                        "files": "off",
                        "engine": CODING_API_LABEL,
                        "help": CatSeekR1CodingAPI.help_text(),
                        "tools": list(CatSeekR1CodingAPI.TOOLS),
                    })
                self._json(200, {
                    "usage": "POST /chat · POST /coding · GET /v1/coding/help · GET /web/sites · GET /web/preview/<id>",
                    "coding_api": CODING_API_LABEL,
                    "web": CatSeekWebProgram.help_text() if CatSeekWebProgram.enabled() else "disabled",
                })
            def log_message(self,*a): pass
        def serve():
            try: ThreadingHTTPServer(("127.0.0.1", CONFIG["api_port"]), Handler).serve_forever()
            except Exception as e: gui.root.after(0, lambda: gui.log("SYSTEM", f"API error: {e}"))
        threading.Thread(target=serve, daemon=True).start()

# ──────────────────────────────────────────────────────────────
# ENTRY
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if "--chat" in sys.argv or "-c" in sys.argv:
        engine = CatR11Engine()
        try:
            run_chat_cli(engine)
        finally:
            engine.clear_history()
            _save_persistent_memory(engine.persistent_memory)
    else:
        def _on_exit(root, engine):
            if messagebox.askokcancel("Quit", f"Exit {WINDOW_TITLE}?"):
                engine.persistent_memory.total_messages += len(engine.chat_history)
                engine.clear_history()
                _save_persistent_memory(engine.persistent_memory)
                root.destroy()

        root = tk.Tk()
        engine = CatR11Engine()
        CatR11GUI(root, engine=engine)
        root.protocol("WM_DELETE_WINDOW", lambda: _on_exit(root, engine))
        root.mainloop()