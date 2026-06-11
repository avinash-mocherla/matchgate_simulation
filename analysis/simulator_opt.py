"""
Compatibility shim. The optimized exact simulator was promoted to the repo root
as the canonical ``simulator.Simulator`` (with ``SimulatorOpt`` kept as an alias).

This module is retained so the existing analysis/ harness (bench.py, compare.py,
profile_highrank.py) keeps importing ``SimulatorOpt`` unchanged.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator import Simulator, SimulatorOpt, _R  # noqa: F401
