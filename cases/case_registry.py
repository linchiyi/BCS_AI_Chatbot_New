"""
教案註冊中心 - 管理所有可用的 OSCE 教案
每個教案完全獨立，不會互相影響 context embedding
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


@dataclass
class CaseConfig:
    """單一教案的完整配置"""
    case_id: str
    case_name: str
    case_description: str
    page_title: str
    page_icon: str
    
    # 病人/家屬角色資訊
    patient_persona: Dict[str, Any]
    
    # 情緒模式設定
    emotion_modes: Dict[str, Dict[str, Any]]
    
    # 對話階段
    stages: List[str]
    stage_guidance: Dict[str, str]
    
    # 系統提示詞模板
    system_prompt_template: str
    
    # 評分提示詞
    evaluation_system_prompt: str
    
    # 路徑配置
    transcripts_dir: Optional[Path] = None
    scores_dir: Optional[Path] = None
    script_files: List[Path] = field(default_factory=list)
    context_embeddings_path: Optional[Path] = None
    
    # Context Engine 載入器（延遲載入）
    context_engine_loader: Optional[Callable] = None


# 全域教案註冊表
CASE_REGISTRY: Dict[str, CaseConfig] = {}


def register_case(case_config: CaseConfig) -> None:
    """註冊一個教案到全域註冊表"""
    CASE_REGISTRY[case_config.case_id] = case_config


def get_case(case_id: str) -> Optional[CaseConfig]:
    """根據 ID 取得教案配置"""
    return CASE_REGISTRY.get(case_id)


def get_all_cases() -> Dict[str, CaseConfig]:
    """取得所有已註冊的教案"""
    return CASE_REGISTRY.copy()


def list_case_options() -> List[Dict[str, str]]:
    """列出所有教案選項（用於 UI 選擇器）"""
    return [
        {
            "case_id": case.case_id,
            "case_name": case.case_name,
            "case_description": case.case_description,
            "page_icon": case.page_icon,
        }
        for case in CASE_REGISTRY.values()
    ]
