#pragma once

#include <filesystem>
#include <array>

constexpr uint64_t MAX_TIMEOUT = UINT64_MAX;
constexpr unsigned int FRAME_OVERLAP = 3;

// project paths.
const auto CURRENT_SOURCE_PATH = std::filesystem::current_path();
const auto PROJECT_ROOT_PATH = CURRENT_SOURCE_PATH.parent_path().parent_path();
const auto SHADER_ROOT_PATH = PROJECT_ROOT_PATH / "shaders";
const auto ASSET_ROOT_PATH = PROJECT_ROOT_PATH / "assets";

const std::array<glm::vec3, 8> BOX_CORNERS {
    glm::vec3 { 1, 1, 1 },
    glm::vec3 { 1, 1, -1 },
    glm::vec3 { 1, -1, 1 },
    glm::vec3 { 1, -1, -1 },
    glm::vec3 { -1, 1, 1 },
    glm::vec3 { -1, 1, -1 },
    glm::vec3 { -1, -1, 1 },
    glm::vec3 { -1, -1, -1 },
};

const glm::vec3 CLIP_SPACE_MIN_BOUND = glm::vec3(-1, -1, 0);
const glm::vec3 CLIP_SPACE_MAX_BOUND = glm::vec3(1, 1, 1);