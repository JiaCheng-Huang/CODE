from abc import ABC, abstractmethod


class BaseMode(ABC):
    name: str = "base"

    #: 是否走多运行评估流程（只有 SC 需要置 True）
    multi_run: bool = False

    @abstractmethod
    def run(self, client, target_sentence: str, context: str) -> tuple[str, str]:
        """单次运行：返回 (raw_log, prediction)"""
        raise NotImplementedError
