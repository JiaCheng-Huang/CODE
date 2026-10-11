"""DeepSeek 空闲时段检测工具。

用法：
    from idle_checker import is_idle_time, seconds_until_idle, status_text
    if is_idle_time():
        ...

判断规则：
    - 法定节假日（含调休放假日）→ 全天空闲
    - 工作日（含调休上班的周末）→ 高峰时段 09:00–12:00 / 14:00–18:00 之外为空闲

首次运行若发现 chinese_calendar 数据未覆盖今年，会自动 pip install -U 并退出，
重新运行即可使用最新数据。
"""

import subprocess
import sys
from datetime import datetime, time as dtime, timezone, timedelta, date
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# 0. 自动升级 chinese_calendar（每年国务院发文后，本地库可能还没更新）
# ---------------------------------------------------------------------------
def _ensure_calendar_up_to_date() -> None:
    """若 chinese_calendar 未覆盖今年，尝试自动升级；升级后提示重新运行。"""
    try:
        import chinese_calendar  # noqa: F401
    except ImportError:
        print("未安装 chinesecalendar，正在安装...", file=sys.stderr)
        _do_upgrade(exit_after=True)
        return

    import chinese_calendar
    try:
        # 用今年 12/31 试探，未覆盖会抛 NotImplementedError
        chinese_calendar.is_workday(date(date.today().year, 12, 31))
        return  # 已覆盖，正常往下走
    except NotImplementedError:
        pass

    print(
        f"chinese_calendar 数据未覆盖 {date.today().year} 年，正在自动升级...",
        file=sys.stderr,
    )
    _do_upgrade(exit_after=True)


def _do_upgrade(exit_after: bool = True) -> None:
    """调用 pip 升级 chinesecalendar；成功后清掉模块缓存。"""
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-U", "chinesecalendar"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        # 清掉已加载的旧模块，防止后续 import 拿到旧对象
        for mod in [m for m in list(sys.modules) if m.startswith("chinese_calendar")]:
            del sys.modules[mod]
        print("chinese_calendar 已升级，请重新运行脚本。", file=sys.stderr)
    except Exception as e:
        print(
            f"自动升级失败：{e}\n"
            f"请手动执行：{sys.executable} -m pip install -U chinesecalendar",
            file=sys.stderr,
        )
    if exit_after:
        sys.exit(1)


_ensure_calendar_up_to_date()

# 自检通过后再正式 import
from chinese_calendar import is_workday, is_holiday, get_holiday_detail  # noqa: E402


# ---------------------------------------------------------------------------
# 1. 常量
# ---------------------------------------------------------------------------
BEIJING_TZ = timezone(timedelta(hours=8))

# 高峰时段（北京时间，工作日）：09:00–12:00 与 14:00–18:00
PEAK_RANGES = [
    (dtime(9, 0), dtime(12, 0)),
    (dtime(14, 0), dtime(18, 0)),
]

# 等待时状态刷新间隔（秒）
STATUS_REFRESH_INTERVAL = 30

# 节日英文名 → 中文显示名（便于状态栏可读）
HOLIDAY_CN = {
    "new_years_day": "元旦",
    "spring_festival": "春节",
    "chinese_new_year_eve": "除夕",
    "tomb_sweeping_day": "清明",
    "labour_day": "劳动节",
    "dragon_boat_festival": "端午",
    "mid_autumn_festival": "中秋",
    "national_day": "国庆",
    "national_day_2": "国庆",
    "anti_fascist_70th_day": "抗战胜利纪念日",
}


# ---------------------------------------------------------------------------
# 2. 核心逻辑
# ---------------------------------------------------------------------------
def now_beijing() -> datetime:
    return datetime.now(BEIJING_TZ)


def _day_info(now: datetime) -> Tuple[bool, Optional[str]]:
    """返回 (是否休息日, 节日中文名或 None)。

    休息日：法定节假日、调休放假日、普通周末。
    工作日：法定工作日（含调休上班的周末）。
    """
    d: date = now.date()
    try:
        if is_workday(d):
            return False, None

        # 非工作日：可能是节假日，也可能是普通周末
        if is_holiday(d):
            _, name = get_holiday_detail(d)
            if name is None:
                return True, None
            raw = getattr(name, "name", str(name))  # 兼容枚举/字符串
            return True, HOLIDAY_CN.get(raw, raw)
        return True, None
    except NotImplementedError:
        # 极端情况下兜底：只看周末
        return (now.weekday() >= 5), None


def is_idle_time(now: Optional[datetime] = None) -> bool:
    """判断给定时刻是否为 DeepSeek 空闲时段（北京时间）。"""
    if now is None:
        now = now_beijing()

    is_rest, _ = _day_info(now)
    if is_rest:  # 休息日全天
        return True

    t = now.time()
    for start, end in PEAK_RANGES:
        if start <= t < end:
            return False
    return True


def seconds_until_idle(now: Optional[datetime] = None) -> int:
    """距下一个空闲时段开始还有多少秒；若已空闲返回 0。"""
    if now is None:
        now = now_beijing()
    if is_idle_time(now):
        return 0

    # 此时必为工作日且处于某个高峰时段
    t = now.time()
    for start, end in PEAK_RANGES:
        if start <= t < end:
            end_dt = now.replace(hour=end.hour, minute=end.minute,
                                 second=0, microsecond=0)
            return max(0, int((end_dt - now).total_seconds()))
    return 0


def status_text() -> str:
    """单行状态文本，用于控制台持续显示。"""
    now = now_beijing()
    is_rest, holiday_name = _day_info(now)

    if is_idle_time(now):
        if is_rest and holiday_name:
            tag, note = "HOLIDAY", f" {holiday_name}"
        elif is_rest:
            tag, note = "WEEKEND", ""
        else:
            tag, note = "OFF-PEAK", ""
        return (
            f"[{now.strftime('%H:%M:%S')} BJT] IDLE  "
            f"({tag}{note}, 空闲阶段, API便宜)"
        )

    wait_s = seconds_until_idle(now)
    mins, secs = divmod(wait_s, 60)
    return (
        f"[{now.strftime('%H:%M:%S')} BJT] PEAK  "
        f"(高峰阶段, 距空闲 {mins}m{secs:02d}s)"
    )


# ---------------------------------------------------------------------------
# 3. 自检
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    for d in ["2026-01-01", "2026-10-01", "2026-10-05",
              "2026-10-08", "2026-10-10", "2026-10-11"]:
        dt = datetime.fromisoformat(d).replace(tzinfo=BEIJING_TZ, hour=10)
        rest, name = _day_info(dt)
        print(
            f"{d} (10:00 BJT)  休息日={rest}  "
            f"节日={name or '-'}  空闲={is_idle_time(dt)}"
        )
    print()
    print(status_text())