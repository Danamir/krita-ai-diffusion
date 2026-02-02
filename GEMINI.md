# Gemini Context: Krita AI Diffusion

## Project Overview
**krita-ai-diffusion** is a Krita plugin that enables generative AI features (inpainting, live painting, upscaling) directly within Krita. It leverages [ComfyUI](https://github.com/comfyanonymous/ComfyUI) as the backend generation engine.

## Architecture
- **Host Application:** Krita (Python Scripting API).
- **UI Framework:** PyQt5 / PySide2 (standard Krita UI toolkit). See `ai_diffusion/ui/`.
- **Backend:** ComfyUI (Python-based node graph execution). The plugin manages a local ComfyUI instance or connects to a remote one.
- **Communication:** HTTP API and WebSockets are used to communicate with the ComfyUI server.

## Development Workflow

### Python Virtual Environment
- **Interpreter:** `.env\Scripts\python.exe`
- **Pip:** `.env\Scripts\pip.exe`
- **Note:** Ensure you use these executables for running scripts, installing dependencies, and running tests to maintain environment consistency.

### Prerequisites
- Python 3.10+
- Krita 5.2.0+
- `uv` (optional, suggested by pyproject.toml but not strictly required if using pip)

### Installation (Dev)
1.  Clone the repository.
2.  Initialize submodules: `git submodule update --init`
3.  Symlink the `ai_diffusion` folder and `ai_diffusion.desktop` file into your Krita pykrita resources folder.
    -   *Windows*: `%APPDATA%\krita\pykrita\`
    -   *Linux*: `~/.local/share/krita/pykrita/`
    -   *macOS*: `~/Library/Application Support/Krita/pykrita/`

### Testing
- **Framework:** `pytest`
- **Command:** `pytest tests`
- **Notes:**
    -   Dependencies: `pip install -r requirements.txt`
    -   Some tests (image generation) require a running ComfyUI server.
    -   Krita API-dependent functionality is largely mocked or untestable outside Krita.

### Code Quality
- **Linting:** `ruff check`
- **Formatting:** `ruff format` (or `black`)
- **Type Checking:** `pyright` (uses `scripts/typeshed` for Krita API stubs).

### Debugging
-   **Method:** Attach to running Krita process.
-   **Tool:** `debugpy` (included as submodule).
-   **Config:** VSCode `launch.json` is provided. Use "Run and Debug" (F5) to attach.
-   **Breakpoint:** `import debugpy; debugpy.breakpoint()` works in code.

## Key Directories & Files
-   `ai_diffusion/`: Main plugin source code.
    -   `ui/`: PyQt widget definitions and UI logic.
    -   `client.py`, `connection.py`: ComfyUI API interaction.
    -   `extension.py`: Plugin entry point.
-   `tests/`: Unit and integration tests.
-   `scripts/`: Utility scripts (packaging, model downloading, etc.).
-   `CONTRIBUTING.md`: Contribution guidelines.
-   `pyproject.toml`: Tool configuration (Ruff, Black, Pyright).

## Conventions
-   **Style:** Python standard recommendations (PEP 8), but NO `ALL_CAPS` constants.
-   **Translations:** Stored in `ai_diffusion/language/` as JSON files.
