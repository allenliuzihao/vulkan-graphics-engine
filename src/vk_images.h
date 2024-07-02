#pragma once 

#include <fastgltf/glm_element_traits.hpp>
#include <fastgltf/parser.hpp>
#include <fastgltf/tools.hpp>

#include <vk_initializers.h>
#include "vk_engine.h"

class VulkanEngine;

namespace vkutil {

    uint32_t calculate_mip_levels(VkExtent2D size);

    void transition_image(VkCommandBuffer cmd, VkImage image, VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess, VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess, VkImageLayout currentLayout, VkImageLayout newLayout);

    void copy_image_to_image(VkCommandBuffer cmd, VkImage source, VkImage destination, VkExtent2D srcSize, VkExtent2D dstSize);

    std::optional<AllocatedImage> load_image(VulkanEngine* engine, fastgltf::Asset& asset, fastgltf::Image& image);

    void generate_mipmaps(VkCommandBuffer cmd, VkImage image, VkExtent2D imageSize);

    // images. 
    AllocatedImage create_image(VkDevice device, const VmaAllocator& allocator, VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped = false);
    AllocatedImage create_image(VkDevice device, VkQueue queue, const VmaAllocator& allocator, vkutil::ImmediateSubmit & submit, void* data, VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped = false);
    void destroy_image(VkDevice device, VmaAllocator allocator, const AllocatedImage& img);
};