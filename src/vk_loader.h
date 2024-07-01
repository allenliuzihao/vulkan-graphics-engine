#pragma once

#include <iostream>

#include <glm/gtx/quaternion.hpp>

#include <fastgltf/glm_element_traits.hpp>
#include <fastgltf/parser.hpp>
#include <fastgltf/tools.hpp>

#include "vk_initializers.h"
#include "vk_types.h"
#include <vk_descriptors.h>
#include "vk_images.h"
#include "vk_engine.h"

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

struct LoadedGLTF : public IRenderable {
    // storage for all the data on a given glTF file
    std::unordered_map<std::string, std::shared_ptr<MeshAsset>> meshes;
    std::unordered_map<std::string, std::shared_ptr<Node>> nodes;
    std::unordered_map<std::string, AllocatedImage> images;
    std::unordered_map<std::string, std::shared_ptr<GLTFMaterial>> materials;

    // nodes that dont have a parent, for iterating through the file in tree order
    std::vector<std::shared_ptr<Node>> topNodes;

    std::vector<VkSampler> samplers;

    DescriptorAllocatorGrowable descriptorPool;

    AllocatedBuffer materialDataBuffer;

    VulkanEngine* creator;

    ~LoadedGLTF() { clearAll(); };

    virtual void Draw(const glm::mat4& topMatrix, DrawContext& ctx);
private:
    void clearAll();
};

// load a list of meshes, each mesh will be its own material. 
std::optional<std::vector<std::shared_ptr<MeshAsset>>> loadGltfMeshes(VulkanEngine* engine, std::filesystem::path filePath);
std::optional<std::shared_ptr<LoadedGLTF>> loadGltf(VulkanEngine* engine, std::filesystem::path filePath);