@echo off
rem GLSL to SPIR-V. Generated .spv files are committed (glslc is not needed at build time).
set SDK=C:\VulkanSDK\1.4.363.0\Bin
for %%f in ("%~dp0shaders\*.comp") do (
  "%SDK%\glslc.exe" --target-env=vulkan1.3 -O "%%f" -o "%%~dpnf.spv" || exit /b 1
  "%SDK%\spirv-val.exe" --target-env vulkan1.3 "%%~dpnf.spv" || exit /b 1
)
