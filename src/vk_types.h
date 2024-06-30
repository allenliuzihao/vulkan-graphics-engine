// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.
#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>
#include <span>
#include <array>
#include <functional>
#include <deque>

#include <vulkan/vulkan.h>
#include <vulkan/vk_enum_string_helper.h>
#include <vk_mem_alloc.h>

#include <fmt/core.h>

#include <glm/mat4x4.hpp>
#include <glm/vec4.hpp>

#define VK_CHECK(x)                                                     \
    do {                                                                \
        VkResult err = x;                                               \
        if (err) {                                                      \
            fmt::println("Detected Vulkan error: {}", string_VkResult(err)); \
            abort();                                                    \
        }                                                               \
    } while (0)

struct AllocatedImage {
    std::string name;
    VkImage image;
    VkImageView imageView;
    VmaAllocation allocation;
    VkExtent3D imageExtent;
    VkFormat imageFormat;
};

struct AllocatedBuffer {
    VkBuffer buffer;
    VmaAllocation allocation;
    VmaAllocationInfo info;
};

struct Vertex {
    glm::vec3 position;
    float uv_x;
    glm::vec3 normal;
    float uv_y;
    glm::vec4 color;
};

// holds the resources needed for a mesh
struct GPUMeshBuffers {
    AllocatedBuffer indexBuffer;
    AllocatedBuffer vertexBuffer;
    VkDeviceAddress vertexBufferAddress;
};

// push constants for our mesh object draws
struct GPUDrawPushConstants {
    glm::mat4 worldMatrix;
    VkDeviceAddress vertexBuffer;
};

struct GPUSceneData {
    glm::mat4 view;
    glm::mat4 proj;
    glm::mat4 viewproj;
    glm::vec4 ambientColor;
    glm::vec4 sunlightDirection; // w for sun power
    glm::vec4 sunlightColor;
};

enum class MaterialPass :uint8_t {
    MainColor,
    Transparent,
    Other
};

struct MaterialPipeline {
    VkPipeline pipeline;        // pbr opaque and pbr transparent pipeline. 
    VkPipelineLayout layout;    
};

// is it opaque or transparent for that material.
struct MaterialInstance {
    // pipeline and its layout.
    MaterialPipeline* pipeline;
    // one ds per material: textures (sampler) and material parameters (uniform buffer).
    VkDescriptorSet materialSet;
    MaterialPass passType;
};

struct RenderObject {
    // describes indices for a mesh
    uint32_t indexCount;
    uint32_t firstIndex;
    VkBuffer indexBuffer;

    // material for that mesh.
    //  pipeline and descriptor set for a material.
    MaterialInstance* material;

    // push constant.
    // transformation of that mesh.
    glm::mat4 transform;
    // vertex buffer GPU pointer.
    VkDeviceAddress vertexBufferAddress;
};

struct DrawContext {
    std::vector<RenderObject> OpaqueSurfaces;
};

// base class for a renderable dynamic object
class IRenderable {
    virtual void Draw(const glm::mat4& topMatrix, DrawContext& ctx) = 0;
};

// implementation of a drawable scene node.
//  the scene node can hold children and will also keep a transform to propagate to them.
struct Node : public IRenderable {
    // parent pointer must be a weak pointer to avoid circular dependencies
    //  this doesn't increment the reference count of parent node.
    std::weak_ptr<Node> parent;
    // this increases reference count for child nodes.
    std::vector<std::shared_ptr<Node>> children;

    // local transform by this node.
    glm::mat4 localTransform;
    // current world transform. 
    glm::mat4 worldTransform;

    // parent matrix: accumulated transform prior to this node.
    void refreshTransform(const glm::mat4& parentMatrix)
    {
        worldTransform = parentMatrix * localTransform;
        for (auto c : children) {
            // propagate transform to descendents. 
            c->refreshTransform(worldTransform);
        }
    }

    // nothing to draw here.
    virtual void Draw(const glm::mat4& topMatrix, DrawContext& ctx)
    {
        // draw children
        for (auto& c : children) {
            c->Draw(topMatrix, ctx);
        }
    }
};
