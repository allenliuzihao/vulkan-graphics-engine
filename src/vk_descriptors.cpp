#include <vk_descriptors.h>

void DescriptorLayoutBuilder::add_binding(uint32_t binding, VkDescriptorType type, uint32_t count)
{
    VkDescriptorSetLayoutBinding newbind{};
    newbind.binding = binding;
    newbind.descriptorCount = count;
    newbind.descriptorType = type;

    bindings.push_back(newbind);
}

void DescriptorLayoutBuilder::clear()
{
    bindings.clear();
}

// which stages should this descriptor set be used. 
VkDescriptorSetLayout DescriptorLayoutBuilder::build(VkDevice device, VkShaderStageFlags shaderStages, void* pNext, VkDescriptorSetLayoutCreateFlags flags)
{
    // propagate stages to bindings of this set.
    for (auto& b : bindings) {
        b.stageFlags |= shaderStages;
    }

    VkDescriptorSetLayoutCreateInfo info = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    info.pNext = pNext;

    info.pBindings = bindings.data();
    info.bindingCount = (uint32_t)bindings.size();
    info.flags = flags;

    VkDescriptorSetLayout set;
    VK_CHECK(vkCreateDescriptorSetLayout(device, &info, nullptr, &set));

    return set;
}

void DescriptorAllocator::init_pool(VkDevice device, uint32_t maxSets, std::span<PoolSizeRatio> poolRatios)
{
    std::vector<VkDescriptorPoolSize> poolSizes;
    for (const PoolSizeRatio& ratio : poolRatios) {
        // how many descriptors allocated per descriptor type in this pool.
        poolSizes.push_back(VkDescriptorPoolSize{
            .type = ratio.type,
             //  this is some strange way to set the number of descriptors per type from this pool. Its based on maxSets. 
            .descriptorCount = uint32_t(ratio.ratio * maxSets)
        });
    }

    VkDescriptorPoolCreateInfo pool_info = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    pool_info.flags = 0;
    pool_info.maxSets = maxSets;
    // number of descriptors per descriptor type.
    pool_info.poolSizeCount = (uint32_t)poolSizes.size();
    pool_info.pPoolSizes = poolSizes.data();
    vkCreateDescriptorPool(device, &pool_info, nullptr, &pool);
}

void DescriptorAllocator::clear_descriptors(VkDevice device)
{
    vkResetDescriptorPool(device, pool, 0);
}

void DescriptorAllocator::destroy_pool(VkDevice device)
{
    vkDestroyDescriptorPool(device, pool, nullptr);
}

VkDescriptorSet DescriptorAllocator::allocate(VkDevice device, VkDescriptorSetLayout layout)
{
    VkDescriptorSetAllocateInfo allocInfo = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    allocInfo.pNext = nullptr;
    allocInfo.descriptorPool = pool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &layout;

    VkDescriptorSet ds;
    VK_CHECK(vkAllocateDescriptorSets(device, &allocInfo, &ds));

    return ds;
}

// get a descriptor pool when there is one that can be used.
//  otherwise, create a new one. Each time run out of free pool, increase the sets per pool to minimize the number
//  of new pools created. 
VkDescriptorPool DescriptorAllocatorGrowable::get_pool(VkDevice device)
{
    VkDescriptorPool newPool;
    // when there are potentially free one, grab it.
    if (readyPools.size() != 0) {
        newPool = readyPools.back();
        readyPools.pop_back();
    } else {
        // otherwise, no free pool, create a new pool with the existing ratio (how much descriptors to allocate per type.)
        newPool = create_pool(device, setsPerPool, ratios);
        // whenever run out of free pool, try to increase the next free pool with more descriptor sets per pool, effecitvely more decriptors per type.
        //  however, don't change the ratio.
        setsPerPool = (uint32_t) std::min(setsPerPool * 1.5, 4092.0);
    }
    return newPool;
}

// create descriptor pools with maximum of set count.
//  also a span of each descriptor type and the number of that descriptors.
VkDescriptorPool DescriptorAllocatorGrowable::create_pool(VkDevice device, uint32_t setCount, std::span<PoolSizeRatio> poolRatios)
{
    std::vector<VkDescriptorPoolSize> poolSizes;
    for (PoolSizeRatio ratio : poolRatios) {
        poolSizes.push_back(VkDescriptorPoolSize {
            .type = ratio.type,
            .descriptorCount = uint32_t(ratio.ratio * setCount)
        });
    }

    VkDescriptorPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.flags = 0;
    pool_info.maxSets = setCount;
    pool_info.poolSizeCount = (uint32_t)poolSizes.size();
    pool_info.pPoolSizes = poolSizes.data();

    VkDescriptorPool newPool;
    vkCreateDescriptorPool(device, &pool_info, nullptr, &newPool);
    return newPool;
}

void DescriptorAllocatorGrowable::init(VkDevice device, uint32_t maxSets, std::span<PoolSizeRatio> poolRatios)
{
    // this ratio don't get changed. 
    ratios.clear();
    for (auto r : poolRatios) {
        ratios.push_back(r);
    }

    // create a ready pool initially.
    VkDescriptorPool newPool = create_pool(device, maxSets, poolRatios);
    // increase capacity for next pool.
    setsPerPool = (uint32_t) std::round(maxSets * 1.5); // grow it for next allocation when current pool is full. but don't change ratio.
    readyPools.push_back(newPool);
}

void DescriptorAllocatorGrowable::clear_pools(VkDevice device)
{
    // free all descriptor allocations from existing pools.
    //  all existing pools are ready afterwards.
    for (auto p : readyPools) {
        vkResetDescriptorPool(device, p, 0);
    }
    for (auto p : fullPools) {
        vkResetDescriptorPool(device, p, 0);
        readyPools.push_back(p);
    }
    fullPools.clear();
    // all pools should be ready now as all allocated descriptors are freed.
}

void DescriptorAllocatorGrowable::destroy_pools(VkDevice device)
{
    // destroy all pools.
    for (auto p : readyPools) {
        vkDestroyDescriptorPool(device, p, nullptr);
    }
    readyPools.clear();
    for (auto p : fullPools) {
        vkDestroyDescriptorPool(device, p, nullptr);
    }
    fullPools.clear();
}

VkDescriptorSet DescriptorAllocatorGrowable::allocate(VkDevice device, VkDescriptorSetLayout layout, void* pNext)
{
    //get or create a pool to allocate from
    VkDescriptorPool poolToUse = get_pool(device);

    // allocate one descriptor set from the pool, based on layout.
    VkDescriptorSetAllocateInfo allocInfo = {};
    allocInfo.pNext = pNext;
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = poolToUse;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &layout;

    VkDescriptorSet ds;
    bool retry = false;
    //allocation failed. Try again. Implement a retry method, as its possible the next ready pool might be full.
    do {
        VkResult result = vkAllocateDescriptorSets(device, &allocInfo, &ds);
        retry = result != VK_SUCCESS;

        if (retry) {
            // the current ready pool is full. Add to full pool.
            fullPools.push_back(poolToUse);

            // get a new ready pool.
            poolToUse = get_pool(device);
            allocInfo.descriptorPool = poolToUse;
        }

    } while (retry);
    
    // pool to use is the pool that succeeds with the prior descriptor set allocation. 
    //  this is the next ready pool for future allocations.
    readyPools.push_back(poolToUse);
    return ds;
}

// VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
// VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
// VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC
// VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC
void DescriptorWriter::write_buffer(int binding, VkBuffer buffer, size_t size, size_t offset, VkDescriptorType type)
{
    // reference to buffer info within the bufferInfos vector.
    bufferInfos.emplace_back(VkDescriptorBufferInfo{
        .buffer = buffer,
        .offset = offset,
        .range = size
    });

    VkWriteDescriptorSet write = { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
    write.dstBinding = binding;
    write.dstSet = VK_NULL_HANDLE; //left empty for now until we need to write it
    write.descriptorCount = 1;
    write.descriptorType = type;
    write.pBufferInfo = nullptr;
    writes.push_back(write);
}

// write image at a specific descriptor set binding.
//  VK_DESCRIPTOR_TYPE_SAMPLER : sampler
//  VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE : image without sampler.
//  VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : image and sampler. 1 descriptor binding to access the texture.
//  VK_DESCRIPTOR_TYPE_STORAGE_IMAGE: no sampler needed. compute shader access pixel data through imageLoad and imageStore.
void DescriptorWriter::write_image(int binding, VkImageView image, VkSampler sampler, VkImageLayout layout, VkDescriptorType type)
{
    imageInfos.emplace_back(VkDescriptorImageInfo{
        .sampler = sampler,
        .imageView = image,
        .imageLayout = layout
    });

    VkWriteDescriptorSet write = { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
    write.dstBinding = binding;
    write.dstSet = VK_NULL_HANDLE; //left empty for now until we need to write it
    write.descriptorCount = 1;
    write.descriptorType = type;
    write.pImageInfo = nullptr; 
    writes.push_back(write);
}

void DescriptorWriter::clear()
{
    writes.clear();
    imageInfos.clear();
    bufferInfos.clear();
}

bool DescriptorWriter::is_image_type(VkDescriptorType type) {
    return type == VK_DESCRIPTOR_TYPE_SAMPLER || type == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE || type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER || type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
}

bool DescriptorWriter::is_buffer_type(VkDescriptorType type) {
    return type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER || type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER || type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC || type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC;
}

// update bindings of images/buffers for a specific descriptor set.
void DescriptorWriter::update_set(VkDevice device, VkDescriptorSet set)
{
    uint32_t imageIndex = 0, bufferIndex = 0;
    for (size_t i = 0; i < writes.size(); ++i) {
        auto& write = writes[i];
        write.dstSet = set;
        if (is_image_type(write.descriptorType)) {
            write.pImageInfo = &imageInfos[imageIndex++];
        } else {
            write.pBufferInfo = &bufferInfos[bufferIndex++];
        }
    }
    vkUpdateDescriptorSets(device, (uint32_t)writes.size(), writes.data(), 0, nullptr);
}