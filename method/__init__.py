from method.base import BaseMode
from method.sa import SaMode
from method.cot import CotMode
from method.self_contrast import SelfContrastMode
from method.self_consistency import SelfConsistencyMode
from method.mad import MadMode
from method.cfmad import CfMadMode
from method.ekmad import EKMadMode

_REGISTRY: dict[str, type[BaseMode]] = {
    SaMode.name: SaMode,
    CotMode.name: CotMode,
    SelfContrastMode.name: SelfContrastMode,
    SelfConsistencyMode.name: SelfConsistencyMode,
    MadMode.name: MadMode,
    CfMadMode.name: CfMadMode,
    EKMadMode.name: EKMadMode,
}


def available_modes() -> list[str]:
    return list(_REGISTRY.keys())


def get_mode(name: str) -> BaseMode:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown mode '{name}'. Available: {available_modes()}")
    return _REGISTRY[name]()
