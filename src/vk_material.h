#pragma once

#include <vk_constants.h>
#include <vk_types.h>
#include <vk_initializers.h>
#include <vk_descriptors.h>
#include <vk_pipelines.h>

struct GLTFMetallic_Roughness {
	MaterialPipeline opaquePipeline;
	MaterialPipeline transparentPipeline;
	// ds layout for both materials.
	VkDescriptorSetLayout materialLayout;

	struct alignas(64) MaterialConstants {
		glm::vec4 colorFactors;				// multiplied with color texture.
		glm::vec4 metal_rough_factors;		// metallic (R), roughness (G)
	};

	// textures and uniform buffers.
	struct MaterialResources {
		// textures.
		AllocatedImage colorImage;
		VkSampler colorSampler;
		AllocatedImage metalRoughImage;
		VkSampler metalRoughSampler;
		// uniform buffer.
		VkBuffer dataBuffer;
		uint32_t dataBufferOffset;
	};

	DescriptorWriter writer;

	void build_pipelines(VkDevice device, VkDescriptorSetLayout gpuGlobalSceneDescriptorLayout, VkFormat colorFormat, VkFormat depthFormat);
	void destroy_resources(VkDevice device);

	MaterialInstance write_material(VkDevice device, MaterialPass pass, const MaterialResources& resources, DescriptorAllocatorGrowable& descriptorAllocator);
};
