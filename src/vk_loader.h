#pragma once

#include "stb_image.h"
#include <iostream>
#include <vk_loader.h>

#include "vk_engine.h"
#include "vk_initializers.h"
#include "vk_types.h"
#include <glm/gtx/quaternion.hpp>

#include <fastgltf/glm_element_traits.hpp>
#include <fastgltf/parser.hpp>
#include <fastgltf/tools.hpp>

#include <vk_types.h>
#include <unordered_map>
#include <filesystem>

struct GLTFMaterial {
    MaterialInstance data;
};

struct GeoSurface {
    uint32_t startIndex;
    uint32_t count;
    std::shared_ptr<GLTFMaterial> material;
};

// each mesh is a different material. 
struct MeshAsset {
    std::string name;

    // each submesh of a mesh.
    //  subset of indices of each submesh.
    std::vector<GeoSurface> surfaces;

    // submesh data from this same buffers.
    GPUMeshBuffers meshBuffers;
};

//forward declaration
class VulkanEngine;

// load a list of meshes, each mesh will be its own material. 
std::optional<std::vector<std::shared_ptr<MeshAsset>>> loadGltfMeshes(VulkanEngine* engine, std::filesystem::path filePath);