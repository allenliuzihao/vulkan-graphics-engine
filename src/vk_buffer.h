#pragma once
#include <vk_types.h>

namespace vkutil {
	// create buffer.
	AllocatedBuffer create_buffer(const VmaAllocator& allocator, size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage);
	void destroy_buffer(const VmaAllocator& allocator, const AllocatedBuffer& buffer);
};