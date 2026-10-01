"""Environment doctor. Run this before anything else.

    python src/check_env.py

Catches the two failure modes that actually bite: the wrong virtualenv being
active, and the wrong `ee` package being installed. There is an unrelated,
long-abandoned Python 2 package on PyPI also called `ee`; installing it instead
of `earthengine-api` produces a confusing "No module named 'StringIO'" error.
"""

import importlib
import os
import sys
from pathlib import Path

OK, WARN, BAD = "  OK  ", " WARN ", " FAIL "
problems = []


def line(status, msg):
    print(f"[{status}] {msg}")


print("=" * 70)
print("ENVIRONMENT CHECK")
print("=" * 70)

# --- interpreter -----------------------------------------------------------
print(f"\npython   : {sys.version.split()[0]}")
print(f"executable: {sys.executable}")

project_root = Path(__file__).resolve().parents[1]
in_venv = sys.prefix != sys.base_prefix
venv_path = Path(sys.prefix)

if not in_venv:
    line(WARN, "not running inside a virtualenv or conda env")
elif project_root in venv_path.parents or venv_path.parent == project_root:
    line(OK, f"virtualenv belongs to this project ({venv_path})")
else:
    line(BAD, f"active env lives OUTSIDE this project: {venv_path}")
    problems.append(
        f"You are using an environment from another project ({venv_path}).\n"
        f"    Create one here instead:\n"
        f"        deactivate\n"
        f"        cd {project_root}\n"
        f"        python3 -m venv .venv && source .venv/bin/activate\n"
        f"        pip install -U pip && pip install -r env/requirements.txt"
    )

major, minor = sys.version_info[:2]
if (major, minor) < (3, 10):
    line(BAD, f"Python {major}.{minor} - the `earthengine` CLI will not run")
    problems.append(
        f"Python {major}.{minor} is too old. ee/cli/commands.py uses PEP 604\n"
        f"    syntax (`str | None`) at module level, which is a hard syntax error\n"
        f"    before 3.10, so `earthengine authenticate` dies on import with\n"
        f"    \"TypeError: unsupported operand type(s) for |\". The library itself\n"
        f"    imports fine; only the command-line tool is affected.\n"
        f"    Rebuild the venv on 3.11:\n"
        f"        deactivate\n"
        f"        cd {project_root}\n"
        f"        rm -rf .venv\n"
        f"        python3.11 -m venv .venv && source .venv/bin/activate\n"
        f"        pip install -U pip && pip install -r env/requirements.txt\n"
        f"    If python3.11 is missing: brew install python@3.11\n"
        f"    To authenticate without upgrading (library path, works on 3.9):\n"
        f"        python -c \"import ee; ee.Authenticate()\""
    )
elif (major, minor) == (3, 10):
    line(WARN, f"Python {major}.{minor} works, but google-api-core is dropping "
               "support; 3.11+ recommended")
else:
    line(OK, f"Python {major}.{minor}")

# --- the ee package --------------------------------------------------------
print()
try:
    import ee
    ee_file = getattr(ee, "__file__", "?")
    if hasattr(ee, "Initialize") and hasattr(ee, "FeatureCollection"):
        line(OK, f"earthengine-api {getattr(ee, '__version__', '?')} ({ee_file})")
    else:
        line(BAD, f"a package named 'ee' is installed but it is NOT "
                  f"earthengine-api ({ee_file})")
        problems.append(
            "Wrong 'ee' package. PyPI has an unrelated Python 2 package called\n"
            "    'ee' that shadows the real one. Remove it and install the right one:\n"
            "        pip uninstall -y ee\n"
            "        pip install earthengine-api"
        )
except ModuleNotFoundError as exc:
    if "StringIO" in str(exc):
        line(BAD, "the installed 'ee' is the obsolete Python 2 package, "
                  "not earthengine-api")
        problems.append(
            "Wrong 'ee' package (it fails on 'import StringIO', which is Python 2).\n"
            "        pip uninstall -y ee\n"
            "        pip install earthengine-api"
        )
    else:
        line(BAD, f"earthengine-api not installed ({exc})")
        problems.append("pip install earthengine-api")
except Exception as exc:
    line(BAD, f"'import ee' failed: {type(exc).__name__}: {exc}")
    problems.append("pip uninstall -y ee && pip install earthengine-api")

# --- everything else -------------------------------------------------------
print()
required = ["numpy", "pandas", "scipy", "sklearn", "geemap"]
optional = ["xgboost", "lightgbm", "geopandas", "rasterio", "matplotlib"]

for name in required:
    try:
        m = importlib.import_module(name)
        line(OK, f"{name} {getattr(m, '__version__', '')}")
    except Exception as exc:
        line(BAD, f"{name} missing ({type(exc).__name__})")
        problems.append(f"pip install {name}")

for name in optional:
    try:
        m = importlib.import_module(name)
        line(OK, f"{name} {getattr(m, '__version__', '')} (optional)")
    except Exception:
        line(WARN, f"{name} missing - optional, but needed for part of the "
                   f"revision")

# --- credentials -----------------------------------------------------------
print()
cred = Path.home() / ".config" / "earthengine" / "credentials"
if cred.exists():
    line(OK, f"Earth Engine credentials found ({cred})")
else:
    line(WARN, "no Earth Engine credentials yet - run: earthengine authenticate")

# --- project config --------------------------------------------------------
sys.path.insert(0, str(project_root / "src"))
C = None
try:
    import config as C
    if C.EE_PROJECT == "ee-yourusername":
        line(BAD, "EE_PROJECT in src/config.py is still the placeholder")
        problems.append("Edit src/config.py and set EE_PROJECT to your Cloud "
                        "project (Code Editor -> gear icon -> Cloud Project)")
    else:
        line(OK, f"EE_PROJECT = {C.EE_PROJECT}")
    if "yourusername" in C.ASSETS["roi_polygons"]:
        line(WARN, "ASSETS in src/config.py still contain placeholder paths")
    else:
        line(OK, f"ASSET_ROOT = {C.ASSET_ROOT}")
except Exception as exc:
    line(WARN, f"could not read src/config.py ({exc})")

# --- live check: does GEE agree? -------------------------------------------
# String-matching config.py only proves it is no longer a placeholder, not that
# the values are right. If credentials exist, ask the server.
if C is not None and cred.exists() and "ee" in dir():
    print()
    try:
        ee.Initialize(project=C.EE_PROJECT)
        line(OK, f"ee.Initialize succeeded with project '{C.EE_PROJECT}'")

        for key in ("roi_polygons", "taipei_boundary"):
            asset_id = C.ASSETS[key]
            try:
                info = ee.data.getAsset(asset_id)
                n = ee.FeatureCollection(asset_id).size().getInfo()
                line(OK, f"{key}: {asset_id} ({info['type']}, {n} features)")
            except Exception as exc:
                line(BAD, f"{key}: cannot read {asset_id} ({exc})")
                problems.append(
                    f"Asset '{asset_id}' is not readable. Open the Assets tab at\n"
                    f"        https://code.earthengine.google.com and copy the exact\n"
                    f"        ID, then fix ASSETS['{key}'] in src/config.py."
                )
    except Exception as exc:
        line(BAD, f"ee.Initialize failed: {exc}")
        problems.append(
            f"EE_PROJECT is set to '{C.EE_PROJECT}', which the server rejected.\n"
            f"        Get the real one from the Code Editor: gear icon (top right of\n"
            f"        the Scripts panel) -> the value shown under 'Cloud Project'.\n"
            f"        Then set it in src/config.py and re-run this script."
        )
elif C is not None:
    print()
    line(WARN, "skipping the live asset check until you have authenticated")

# --- verdict ---------------------------------------------------------------
print("\n" + "=" * 70)
if problems:
    print(f"{len(problems)} thing(s) to fix:\n")
    for i, p in enumerate(problems, 1):
        print(f"  {i}. {p}\n")
    sys.exit(1)
print("All checks passed. Next: python src/download_assets.py")
print("=" * 70)
