#pragma once

#include <filesystem>

constexpr uint64_t MAX_TIMEOUT = UINT64_MAX;
constexpr unsigned int FRAME_OVERLAP = 3;

// project paths.
const auto CURRENT_SOURCE_PATH = std::filesystem::current_path();
const auto PROJECT_ROOT_PATH = CURRENT_SOURCE_PATH.parent_path().parent_path();
const auto SHADER_ROOT_PATH = PROJECT_ROOT_PATH / "shaders";
const auto ASSET_ROOT_PATH = PROJECT_ROOT_PATH / "assets";