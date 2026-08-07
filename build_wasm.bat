@echo off
:: Convenience wrapper that delegates to the real build script.
call "%~dp0scripts\build_wasm.bat" %*
