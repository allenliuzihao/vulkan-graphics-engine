#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"

//> includes
#include "vk_engine.h"
#include "vk_images.h"

#include <SDL.h>
#include <SDL_vulkan.h>

#include <vk_initializers.h>
#include <vk_types.h>
#include <vk_pipelines.h>
#include "VkBootstrap.h"

#include "imgui.h"
#include "imgui_impl_sdl2.h"
#include "imgui_impl_vulkan.h"

#include <glm/gtx/transform.hpp>

#include <chrono>
#include <thread>
#include <numbers>
#include <filesystem>

// global pointer for vulkan engine singleton reference.
VulkanEngine* loadedEngine = nullptr;

VulkanEngine& VulkanEngine::Get() { return *loadedEngine; }

constexpr bool bUseValidationLayers = false;
void VulkanEngine::init_vulkan()
{
    vkb::InstanceBuilder builder;

    //make the vulkan instance, with basic debug features
    auto inst_ret = builder.set_app_name("Vulkan Application")
        .request_validation_layers(bUseValidationLayers)
        .use_default_debug_messenger()
        .require_api_version(1, 3, 0)
        .build();

    vkb::Instance vkb_inst = inst_ret.value();

    //grab the instance 
    _instance = vkb_inst.instance;
    _debug_messenger = vkb_inst.debug_messenger;

    // create vulkan surface from SDL window library.
    SDL_Vulkan_CreateSurface(_window, _instance, &_surface);

    //vulkan 1.3 features
    VkPhysicalDeviceVulkan13Features features{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES };
    features.dynamicRendering = true;       // simplified vulkan render pass.
    features.synchronization2 = true;       // improved synchronizaion, why 2 in the end.
    
    //vulkan 1.2 features
    VkPhysicalDeviceVulkan12Features features12{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
    features12.bufferDeviceAddress = true;      // shader pointer access to buffer memory.
    features12.descriptorIndexing = true;       // bindless, allows arbitrary sized descriptor arrays. 
    features12.timelineSemaphore = true;

    //use vkbootstrap to select a gpu. 
    //We want a gpu that can write to the SDL surface and supports vulkan 1.3 with the correct features
    vkb::PhysicalDeviceSelector selector{ vkb_inst };
    vkb::PhysicalDevice physicalDevice = selector
        .set_minimum_version(1, 3)
        .set_required_features_13(features)
        .set_required_features_12(features12)
        .set_surface(_surface)
        .select()
        .value();

    //create the final vulkan device
    vkb::DeviceBuilder deviceBuilder{ physicalDevice };
    vkb::Device vkbDevice = deviceBuilder.build().value();

    // Get the VkDevice handle used in the rest of a vulkan application
    _device = vkbDevice.device;
    _chosenGPU = physicalDevice.physical_device;

    // use vkbootstrap to get a Graphics queue, which can do it all.
    _graphicsQueue = vkbDevice.get_queue(vkb::QueueType::graphics).value();
    _graphicsQueueFamily = vkbDevice.get_queue_index(vkb::QueueType::graphics).value();

    // initialize the memory allocator
    VmaAllocatorCreateInfo allocatorInfo = {};
    allocatorInfo.physicalDevice = _chosenGPU;
    allocatorInfo.device = _device;
    allocatorInfo.instance = _instance;
    allocatorInfo.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    vmaCreateAllocator(&allocatorInfo, &_allocator);

    _mainDeletionQueue.push_function([&]() {
        vmaDestroyAllocator(_allocator);
    });
}

void VulkanEngine::init_swapchain()
{
    create_swapchain(_windowExtent.width, _windowExtent.height);

    //draw image size will match the window
    VkExtent3D drawImageExtent = {
        _windowExtent.width,
        _windowExtent.height,
        1
    };

    VkImageUsageFlags drawImageUsages{};
    drawImageUsages |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    drawImageUsages |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    drawImageUsages |= VK_IMAGE_USAGE_STORAGE_BIT;
    drawImageUsages |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    
    for (int i = 0; i < FRAME_OVERLAP; ++i) {
        //hardcoding the draw format to 32 bit float
        _frames[i]._drawImage.imageFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
        _frames[i]._drawImage.imageExtent = drawImageExtent;

        VkImageCreateInfo rimg_info = vkinit::image_create_info(_frames[i]._drawImage.imageFormat, drawImageUsages, drawImageExtent);

        //for the draw image, we want to allocate it from gpu local memory
        VmaAllocationCreateInfo rimg_allocinfo = {};
        rimg_allocinfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
        rimg_allocinfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        // allocate and create the image on the gpu device local memory.
        //  return allocated image and allocation. 
        vmaCreateImage(_allocator, &rimg_info, &rimg_allocinfo, &_frames[i]._drawImage.image, &_frames[i]._drawImage.allocation, nullptr);

        //build a image-view for the draw image to use for rendering
        VkImageViewCreateInfo rview_info = vkinit::imageview_create_info(_frames[i]._drawImage.imageFormat, _frames[i]._drawImage.image, VK_IMAGE_ASPECT_COLOR_BIT);
        VK_CHECK(vkCreateImageView(_device, &rview_info, nullptr, &_frames[i]._drawImage.imageView));

        // add depth image
        _frames[i]._depthImage.imageFormat = VK_FORMAT_D32_SFLOAT;
        _frames[i]._depthImage.imageExtent = drawImageExtent;
        VkImageUsageFlags depthImageUsages{};
        depthImageUsages |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

        VkImageCreateInfo dimg_info = vkinit::image_create_info(_frames[i]._depthImage.imageFormat, depthImageUsages, drawImageExtent);

        //allocate and create the image
        vmaCreateImage(_allocator, &dimg_info, &rimg_allocinfo, &_frames[i]._depthImage.image, &_frames[i]._depthImage.allocation, nullptr);

        //build a image-view for the draw image to use for rendering
        VkImageViewCreateInfo dview_info = vkinit::imageview_create_info(_frames[i]._depthImage.imageFormat, _frames[i]._depthImage.image, VK_IMAGE_ASPECT_DEPTH_BIT);
        VK_CHECK(vkCreateImageView(_device, &dview_info, nullptr, &_frames[i]._depthImage.imageView));

        //add to deletion queues
        _mainDeletionQueue.push_function([=]() {
            vkDestroyImageView(_device, _frames[i]._drawImage.imageView, nullptr);
            vmaDestroyImage(_allocator, _frames[i]._drawImage.image, _frames[i]._drawImage.allocation);

            vkDestroyImageView(_device, _frames[i]._depthImage.imageView, nullptr);
            // free image and its allocation memory.
            vmaDestroyImage(_allocator, _frames[i]._depthImage.image, _frames[i]._depthImage.allocation);
        });
    }
}

void VulkanEngine::init_commands()
{
    //create a command pool for commands submitted to the graphics queue.
    //we also want the pool to allow for resetting of individual command buffers, via vkResetCommandBuffer().
    VkCommandPoolCreateInfo commandPoolInfo = vkinit::command_pool_create_info(_graphicsQueueFamily, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);

    for (int i = 0; i < FRAME_OVERLAP; i++) {
        // create command pool for each frame.
        VK_CHECK(vkCreateCommandPool(_device, &commandPoolInfo, nullptr, &_frames[i]._commandPool));

        // allocate the default command buffer that we will use for rendering
        VkCommandBufferAllocateInfo cmdAllocInfo = vkinit::command_buffer_allocate_info(_frames[i]._commandPool, 1);
        VK_CHECK(vkAllocateCommandBuffers(_device, &cmdAllocInfo, &_frames[i]._mainCommandBuffer));
    }

    // immediate submit mode command pool.
    VK_CHECK(vkCreateCommandPool(_device, &commandPoolInfo, nullptr, &_immCommandPool));

    // allocate the command buffer for immediate submits
    VkCommandBufferAllocateInfo cmdAllocInfo = vkinit::command_buffer_allocate_info(_immCommandPool, 1);
    // this command buffer is resettable, which means it can be allocated once and reset multiple times.
    VK_CHECK(vkAllocateCommandBuffers(_device, &cmdAllocInfo, &_immCommandBuffer));

    _mainDeletionQueue.push_function([=]() {
        vkDestroyCommandPool(_device, _immCommandPool, nullptr);
    });
}

void VulkanEngine::init_sync_structures()
{
    //create syncronization structures
    //one fence to control when the gpu has finished rendering the frame,
    //and 2 semaphores to syncronize rendering with swapchain
    //we want the fence to start signalled so we can wait on it on the first frame (unblock first frame wait)
    VkFenceCreateInfo fenceCreateInfo = vkinit::fence_create_info(VK_FENCE_CREATE_SIGNALED_BIT);
    VkSemaphoreTypeCreateInfo semaphoreTypeCreateInfo = vkinit::semaphore_type_create_info(VK_SEMAPHORE_TYPE_BINARY);
    VkSemaphoreCreateInfo semaphoreCreateInfo = vkinit::semaphore_create_info(&semaphoreTypeCreateInfo);

    // create per-frame resources.
    for (int i = 0; i < FRAME_OVERLAP; i++) {
        VK_CHECK(vkCreateFence(_device, &fenceCreateInfo, nullptr, &_frames[i]._renderFence));

        VK_CHECK(vkCreateSemaphore(_device, &semaphoreCreateInfo, nullptr, &_frames[i]._acquireSemaphore));
        VK_CHECK(vkCreateSemaphore(_device, &semaphoreCreateInfo, nullptr, &_frames[i]._renderSemaphore));
    }

    // fence is signaled when created.
    VK_CHECK(vkCreateFence(_device, &fenceCreateInfo, nullptr, &_immFence));
    _mainDeletionQueue.push_function([=]() { 
        // capture by value.
        vkDestroyFence(_device, _immFence, nullptr); 
    });
}

void VulkanEngine::init_descriptors()
{
    // create a descriptor pool that will hold 10 sets with 1 image each
    std::vector<DescriptorAllocatorGrowable::PoolSizeRatio> sizes =
    {
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1 } // storage image type has 10 descriptors in the entire pool.
    };
    // pool can create 10 descriptor sets, with 10 descriptors of type storage image. 
    globalDescriptorAllocator.init(_device, 10, sizes);
    //make the descriptor set layout for our compute draw
    {
        DescriptorLayoutBuilder builder;
        builder.add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        _drawImageDescriptorLayout = builder.build(_device, VK_SHADER_STAGE_COMPUTE_BIT);
    }

    // create a descriptor pool
    std::vector<DescriptorAllocatorGrowable::PoolSizeRatio> frame_sizes = {
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 3 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 3 },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4 },
    };

    for (uint32_t i = 0; i < FRAME_OVERLAP; ++i) {
        //allocate a descriptor set for our draw image
        _frames[i]._drawImageDescriptors = globalDescriptorAllocator.allocate(_device, _drawImageDescriptorLayout);

        // bind storage image to that set.
        DescriptorWriter writer;
        writer.write_image(0, _frames[i]._drawImage.imageView, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        writer.update_set(_device, _frames[i]._drawImageDescriptors);

        _frames[i]._frameDescriptors = std::move(DescriptorAllocatorGrowable{});
        _frames[i]._frameDescriptors.init(_device, 1000, frame_sizes);

        // capture by reference all variable except i.
        _mainDeletionQueue.push_function([&, i]() {
            _frames[i]._frameDescriptors.destroy_pools(_device);
        });
    }

    {
        DescriptorLayoutBuilder builder;
        builder.add_binding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
        _gpuSceneDataDescriptorLayout = builder.build(_device, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);
    }
    
    {
        DescriptorLayoutBuilder builder;
        builder.add_binding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
        _singleImageDescriptorLayout = builder.build(_device, VK_SHADER_STAGE_FRAGMENT_BIT);
    }

    //make sure both the descriptor allocator and the new layout get cleaned up properly
    _mainDeletionQueue.push_function([&]() {
        globalDescriptorAllocator.destroy_pools(_device);

        vkDestroyDescriptorSetLayout(_device, _singleImageDescriptorLayout, nullptr);
        vkDestroyDescriptorSetLayout(_device, _gpuSceneDataDescriptorLayout, nullptr);
        vkDestroyDescriptorSetLayout(_device, _drawImageDescriptorLayout, nullptr);
    });
}


void VulkanEngine::init_pipelines()
{
    init_background_pipelines();
    //init_triangle_pipeline();
    init_mesh_pipeline();
    // build pbr material pipeline.
    metalRoughMaterial.build_pipelines(_device, _gpuSceneDataDescriptorLayout, _frames[0]._drawImage.imageFormat, _frames[0]._depthImage.imageFormat);
}

void VulkanEngine::init_mesh_pipeline() {
    // build current folder.
    std::string triangleVertexShaderPath = (SHADER_ROOT_PATH / "colored_triangle_mesh.vert.spv").string();
    //std::string triangleFragmentShaderPath = (SHADER_ROOT_PATH / "colored_triangle.frag.spv").string();
    std::string triangleFragmentShaderPath = (SHADER_ROOT_PATH / "tex_image.frag.spv").string();

    VkShaderModule triangleFragShader;
    if (!vkutil::load_shader_module(triangleFragmentShaderPath.c_str(), _device, &triangleFragShader)) {
        fmt::print("Error when building the triangle fragment shader module");
    } else {
        fmt::print("Triangle fragment shader succesfully loaded");
    }

    VkShaderModule triangleVertexShader;
    if (!vkutil::load_shader_module(triangleVertexShaderPath.c_str(), _device, &triangleVertexShader)) {
        fmt::print("Error when building the triangle vertex shader module");
    } else {
        fmt::print("Triangle vertex shader succesfully loaded");
    }

    VkPushConstantRange bufferRange{};
    bufferRange.offset = 0;
    bufferRange.size = sizeof(GPUDrawPushConstants);
    bufferRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    VkPipelineLayoutCreateInfo pipeline_layout_info = vkinit::pipeline_layout_create_info();
    pipeline_layout_info.pPushConstantRanges = &bufferRange;
    pipeline_layout_info.pushConstantRangeCount = 1;
    pipeline_layout_info.pSetLayouts = &_singleImageDescriptorLayout;
    pipeline_layout_info.setLayoutCount = 1;
    VK_CHECK(vkCreatePipelineLayout(_device, &pipeline_layout_info, nullptr, &_meshPipelineLayout));

    PipelineBuilder pipelineBuilder;
    //use the triangle layout we created
    pipelineBuilder._pipelineLayout = _meshPipelineLayout;
    //connecting the vertex and pixel shaders to the pipeline
    pipelineBuilder.set_shaders(triangleVertexShader, triangleFragShader);
    //it will draw triangles
    pipelineBuilder.set_input_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    //filled triangles
    pipelineBuilder.set_polygon_mode(VK_POLYGON_MODE_FILL);
    //no backface culling
    pipelineBuilder.set_cull_mode(VK_CULL_MODE_NONE, VK_FRONT_FACE_CLOCKWISE);
    //no multisampling
    pipelineBuilder.set_multisampling_none();
    //no blending
    pipelineBuilder.enable_blending_alphablend();
    // reverse Z.
    pipelineBuilder.enable_depthtest(true, VK_COMPARE_OP_GREATER_OR_EQUAL);

    //connect the image format we will draw into, from draw image
    pipelineBuilder.set_color_attachment_format(_frames[0]._drawImage.imageFormat);
    pipelineBuilder.set_depth_format(_frames[0]._depthImage.imageFormat);

    //finally build the pipeline
    _meshPipeline = pipelineBuilder.build_pipeline(_device);

    //clean structures
    vkDestroyShaderModule(_device, triangleFragShader, nullptr);
    vkDestroyShaderModule(_device, triangleVertexShader, nullptr);

    _mainDeletionQueue.push_function([=]() {
        vkDestroyPipeline(_device, _meshPipeline, nullptr);
        vkDestroyPipelineLayout(_device, _meshPipelineLayout, nullptr);
    });
}

void VulkanEngine::init_triangle_pipeline() {
    // build current folder.
    std::string triangleVertexShaderPath = (SHADER_ROOT_PATH / "colored_triangle.vert.spv").string();
    std::string triangleFragmentShaderPath = (SHADER_ROOT_PATH / "colored_triangle.frag.spv").string();

    VkShaderModule triangleFragShader;
    if (!vkutil::load_shader_module(triangleFragmentShaderPath.c_str(), _device, &triangleFragShader)) {
        fmt::print("Error when building the triangle fragment shader module");
    } else {
        fmt::print("Triangle fragment shader succesfully loaded");
    }

    VkShaderModule triangleVertexShader;
    if (!vkutil::load_shader_module(triangleVertexShaderPath.c_str(), _device, &triangleVertexShader)) {
        fmt::print("Error when building the triangle vertex shader module");
    } else {
        fmt::print("Triangle vertex shader succesfully loaded");
    }

    //build the pipeline layout that controls the inputs/outputs of the shader
    //we are not using descriptor sets or other systems yet, so no need to use anything other than empty default
    VkPipelineLayoutCreateInfo pipeline_layout_info = vkinit::pipeline_layout_create_info();
    VK_CHECK(vkCreatePipelineLayout(_device, &pipeline_layout_info, nullptr, &_trianglePipelineLayout));

    PipelineBuilder pipelineBuilder;
    //use the triangle layout we created
    pipelineBuilder._pipelineLayout = _trianglePipelineLayout;
    //connecting the vertex and pixel shaders to the pipeline
    pipelineBuilder.set_shaders(triangleVertexShader, triangleFragShader);
    //it will draw triangles
    pipelineBuilder.set_input_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    //filled triangles
    pipelineBuilder.set_polygon_mode(VK_POLYGON_MODE_FILL);
    //no backface culling
    pipelineBuilder.set_cull_mode(VK_CULL_MODE_NONE, VK_FRONT_FACE_CLOCKWISE);
    //no multisampling
    pipelineBuilder.set_multisampling_none();
    //no blending
    pipelineBuilder.disable_blending();
    //no depth testing
    pipelineBuilder.disable_depthtest();

    //connect the image format we will draw into, from draw image
    pipelineBuilder.set_color_attachment_format(_frames[0]._drawImage.imageFormat);
    pipelineBuilder.set_depth_format(_frames[0]._depthImage.imageFormat);

    //finally build the pipeline
    _trianglePipeline = pipelineBuilder.build_pipeline(_device);

    //clean structures
    vkDestroyShaderModule(_device, triangleFragShader, nullptr);
    vkDestroyShaderModule(_device, triangleVertexShader, nullptr);

    _mainDeletionQueue.push_function([=]() {
        vkDestroyPipeline(_device, _trianglePipeline, nullptr);
        vkDestroyPipelineLayout(_device, _trianglePipelineLayout, nullptr);
    });
}

void VulkanEngine::init_background_pipelines()
{
    VkPipelineLayoutCreateInfo computeLayout{};
    computeLayout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    computeLayout.pNext = nullptr;
    // compute pipeline set = 0.
    computeLayout.pSetLayouts = &_drawImageDescriptorLayout;
    computeLayout.setLayoutCount = 1;
    // push constant.
    VkPushConstantRange pushConstant{};
    pushConstant.offset = 0;
    pushConstant.size = sizeof(ComputePushConstants);
    pushConstant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    computeLayout.pPushConstantRanges = &pushConstant;
    computeLayout.pushConstantRangeCount = 1;
    VK_CHECK(vkCreatePipelineLayout(_device, &computeLayout, nullptr, &_gradientPipelineLayout));

    // build current folder.
    std::string gradientColorShaderPath = (SHADER_ROOT_PATH / "gradient_color.comp.spv").string();
    std::string gradientShaderPath = (SHADER_ROOT_PATH / "gradient.comp.spv").string();
    std::string skyShaderPath = (SHADER_ROOT_PATH / "sky.comp.spv").string();

    //layout code
    VkShaderModule gradientColorShader;
    if (!vkutil::load_shader_module(gradientColorShaderPath.c_str(), _device, &gradientColorShader))
    {
        fmt::print("Error when building the compute shader \n");
    }
    VkShaderModule gradientShader;
    if (!vkutil::load_shader_module(gradientShaderPath.c_str(), _device, &gradientShader))
    {
        fmt::print("Error when building the compute shader \n");
    }
    VkShaderModule skyShader;
    if (!vkutil::load_shader_module(skyShaderPath.c_str(), _device, &skyShader))
    {
        fmt::print("Error when building the compute shader \n");
    }

    VkPipelineShaderStageCreateInfo stageinfo{};
    stageinfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stageinfo.pNext = nullptr;
    stageinfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stageinfo.module = gradientColorShader;
    stageinfo.pName = "main";

    VkComputePipelineCreateInfo computePipelineCreateInfo{};
    computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    computePipelineCreateInfo.pNext = nullptr;
    computePipelineCreateInfo.layout = _gradientPipelineLayout;
    computePipelineCreateInfo.stage = stageinfo;

    ComputeEffect gradientColor;
    gradientColor.layout = _gradientPipelineLayout;
    gradientColor.name = "gradient color";
    gradientColor.data = {};
    //default colors
    gradientColor.data.data1 = glm::vec4(1, 0, 0, 1);
    gradientColor.data.data2 = glm::vec4(0, 0, 1, 1);
    VK_CHECK(vkCreateComputePipelines(_device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, nullptr, &gradientColor.pipeline));

    computePipelineCreateInfo.stage.module = skyShader;
    ComputeEffect sky;
    sky.layout = _gradientPipelineLayout;
    sky.name = "sky";
    sky.data = {};
    VK_CHECK(vkCreateComputePipelines(_device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, nullptr, &sky.pipeline));

    computePipelineCreateInfo.stage.module = gradientShader;
    ComputeEffect gradient;
    gradient.layout = _gradientPipelineLayout;
    gradient.name = "gradient";
    gradient.data = {};
    VK_CHECK(vkCreateComputePipelines(_device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, nullptr, &gradient.pipeline));

    backgroundEffects.push_back(gradientColor);
    backgroundEffects.push_back(sky);
    backgroundEffects.push_back(gradient);

    // shader module not needed after compute pipelien creation.
    vkDestroyShaderModule(_device, gradientColorShader, nullptr);
    vkDestroyShaderModule(_device, gradientShader, nullptr);
    vkDestroyShaderModule(_device, skyShader, nullptr);

    // capture by value here.
    _mainDeletionQueue.push_function([=]() {
        vkDestroyPipeline(_device, gradient.pipeline, nullptr);
        vkDestroyPipeline(_device, sky.pipeline, nullptr);
        vkDestroyPipeline(_device, gradientColor.pipeline, nullptr);

        vkDestroyPipelineLayout(_device, _gradientPipelineLayout, nullptr);
    });
}

void VulkanEngine::init_imgui() {
    // 1: create descriptor pool for IMGUI
    //  the size of the pool is very oversize, but it's copied from imgui demo
    //  itself.
    VkDescriptorPoolSize pool_sizes[] = { 
        { VK_DESCRIPTOR_TYPE_SAMPLER, 1000 },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 },
        { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000 },
        { VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000 } 
    };

    VkDescriptorPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    // enable freeing individual descriptor sets.
    pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pool_info.maxSets = 1000;
    pool_info.poolSizeCount = (uint32_t)std::size(pool_sizes);
    pool_info.pPoolSizes = pool_sizes;

    VkDescriptorPool imguiPool;
    VK_CHECK(vkCreateDescriptorPool(_device, &pool_info, nullptr, &imguiPool));

    // 2: initialize imgui library

    // this initializes the core structures of imgui
    ImGui::CreateContext();

    // this initializes imgui for SDL
    ImGui_ImplSDL2_InitForVulkan(_window);

    // this initializes imgui for Vulkan
    ImGui_ImplVulkan_InitInfo init_info = {};
    init_info.Instance = _instance;
    init_info.PhysicalDevice = _chosenGPU;
    init_info.Device = _device;
    init_info.Queue = _graphicsQueue;
    init_info.DescriptorPool = imguiPool;
    init_info.MinImageCount = 3;
    init_info.ImageCount = 3;
    init_info.UseDynamicRendering = true;

    //dynamic rendering parameters for imgui to use
    init_info.PipelineRenderingCreateInfo = { .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
    init_info.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
    init_info.PipelineRenderingCreateInfo.pColorAttachmentFormats = &_swapchainImageFormat;
    init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;

    ImGui_ImplVulkan_Init(&init_info);

    ImGui_ImplVulkan_CreateFontsTexture();

    // add the destroy the imgui created structures
    _mainDeletionQueue.push_function([=]() {
        ImGui_ImplVulkan_Shutdown();
        vkDestroyDescriptorPool(_device, imguiPool, nullptr);
    });
}

void VulkanEngine::init_default_data() {
    /*
    std::array<Vertex, 4> rect_vertices;
    // furthrest. 
    rect_vertices[0].position = { 0.5,-0.5, 0 };
    rect_vertices[1].position = { 0.5,0.5, 0 };
    rect_vertices[2].position = { -0.5,-0.5, 0 };
    rect_vertices[3].position = { -0.5,0.5, 0 };

    rect_vertices[0].color = { 0,0, 0,1 };
    rect_vertices[1].color = { 0.5,0.5,0.5 ,1 };
    rect_vertices[2].color = { 1,0, 0,1 };
    rect_vertices[3].color = { 0,1, 0,1 };

    std::array<uint32_t, 6> rect_indices;
    rect_indices[0] = 0;
    rect_indices[1] = 1;
    rect_indices[2] = 2;
    rect_indices[3] = 2;
    rect_indices[4] = 1;
    rect_indices[5] = 3;
    // rect_vertices and rect_indices are passed in as pointers + sizes.
    _meshData = uploadMesh(rect_indices, rect_vertices);
    */

    std::string gltfFilePath = (ASSET_ROOT_PATH / "basicmesh.glb").string();
    _testMeshes = loadGltfMeshes(this, gltfFilePath).value();

    //3 default textures, white, grey, black. 1 pixel each, 
    //  0 -> 32
    //  R8G8B8A8, where each component is scaled between [0, 255].
    uint32_t white = glm::packUnorm4x8(glm::vec4(1, 1, 1, 1));
    _defaultImages[0] = std::move(create_image((void*)&white, VkExtent3D{1, 1, 1}, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT));
    _defaultImages[0].name = "white";

    uint32_t grey = glm::packUnorm4x8(glm::vec4(0.66f, 0.66f, 0.66f, 1));
    _defaultImages[1] = std::move(create_image((void*)&grey, VkExtent3D{1, 1, 1}, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT));
    _defaultImages[1].name = "grey";

    uint32_t black = glm::packUnorm4x8(glm::vec4(0, 0, 0, 0));
    _defaultImages[2] = std::move(create_image((void*)&black, VkExtent3D{1, 1, 1}, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT));
    _defaultImages[2].name = "black";

    //checkerboard image
    //uint32_t magenta = glm::packUnorm4x8(glm::vec4(1, 1, 1, 1));
    std::array<uint32_t, 16 * 16> pixels; //for 16x16 checkerboard texture
    for (int y = 0; y < 16; y++) {           // over row
        for (int x = 0; x < 16; x++) {       // over column
            pixels[y * 16 + x] = ((x % 2) ^ (y % 2)) ? white : black;
        }
    }
    _defaultImages[3] = std::move(create_image(pixels.data(), VkExtent3D{16, 16, 1}, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT));
    _defaultImages[3].name = "white-black-checkerboard";

    VkSamplerCreateInfo sampl = { .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
    sampl.magFilter = VK_FILTER_NEAREST;
    sampl.minFilter = VK_FILTER_NEAREST;
    vkCreateSampler(_device, &sampl, nullptr, &_defaultSamplerNearest);

    sampl.magFilter = VK_FILTER_LINEAR;
    sampl.minFilter = VK_FILTER_LINEAR;
    vkCreateSampler(_device, &sampl, nullptr, &_defaultSamplerLinear);

    GLTFMetallic_Roughness::MaterialResources materialResources;
    //default the material textures
    materialResources.colorImage = _defaultImages[0];       // white
    materialResources.colorSampler = _defaultSamplerLinear;
    materialResources.metalRoughImage = _defaultImages[0];  // white
    materialResources.metalRoughSampler = _defaultSamplerLinear;
    
    //set the uniform buffer for the material data
    AllocatedBuffer materialConstants = vkutil::create_buffer(_allocator, sizeof(GLTFMetallic_Roughness::MaterialConstants), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

    //write the buffer
    GLTFMetallic_Roughness::MaterialConstants* sceneUniformData = (GLTFMetallic_Roughness::MaterialConstants*) materialConstants.allocation->GetMappedData();
    sceneUniformData->colorFactors = glm::vec4{ 1, 1, 1, 1 };
    sceneUniformData->metal_rough_factors = glm::vec4{ 1, 0.5, 0, 0 };

    // local variables by value, class members by reference
    _mainDeletionQueue.push_function([=, this]() {
        vkutil::destroy_buffer(_allocator, materialConstants);
    });

    materialResources.dataBuffer = materialConstants.buffer;
    materialResources.dataBufferOffset = 0;
    defaultData = metalRoughMaterial.write_material(_device, MaterialPass::MainColor, materialResources, globalDescriptorAllocator);
    auto defaultMaterial = std::make_shared<GLTFMaterial>(defaultData);
    
    // load mesh nodes.
    for (auto& m : _testMeshes) {
        std::shared_ptr<MeshNode> newNode = std::make_shared<MeshNode>();
        newNode->mesh = m;

        newNode->localTransform = glm::mat4{ 1.f };
        newNode->worldTransform = glm::mat4{ 1.f };

        // for each surface, add its material.
        for (auto& s : newNode->mesh->surfaces) {
            // this will create a bunch of material that point to the same data.
            s.material = defaultMaterial;
        }
        // mesh nodes are stored in loaded nodes.
        loadedNodes[m->name] = std::move(newNode);
    }

    //delete the rectangle data on engine shutdown
    _mainDeletionQueue.push_function([&]() {
        //fmt::println("delete allocated meshes from index and vertex buffers.");
        //destroy_buffer(_meshData.indexBuffer);
        //destroy_buffer(_meshData.vertexBuffer);
        metalRoughMaterial.destroy_resources(_device);

        // delete allocated images.
        vkDestroySampler(_device, _defaultSamplerNearest, nullptr);
        vkDestroySampler(_device, _defaultSamplerLinear, nullptr);

        for (uint32_t i = 0; i < 4; ++i) {
            destroy_image(_defaultImages[i]);
        }

        // delete allocated meshes.
        fmt::println("delete allocated meshes from gltf.");
        for (auto& testMesh : _testMeshes) {
            vkutil::destroy_buffer(_allocator, testMesh->meshBuffers.indexBuffer);
            vkutil::destroy_buffer(_allocator, testMesh->meshBuffers.vertexBuffer);
        }
    });
}

void VulkanEngine::create_swapchain(uint32_t width, uint32_t height) {
    vkb::SwapchainBuilder swapchainBuilder{ _chosenGPU, _device, _surface };

    // this appear to be swizzled, but GPU will store image in this format. However, in shader, RGB is still treated as RGB. 
    _swapchainImageFormat = VK_FORMAT_B8G8R8A8_UNORM;

    vkb::Swapchain vkbSwapchain = swapchainBuilder
        //.use_default_format_selection()
        .set_desired_format(VkSurfaceFormatKHR{ .format = _swapchainImageFormat, .colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR })
        //use vsync present mode FIFO.
        .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
        .set_desired_extent(width, height)
        .add_image_usage_flags(VK_IMAGE_USAGE_TRANSFER_DST_BIT)
        .build()
        .value();

    _swapchainExtent = vkbSwapchain.extent;
    //store swapchain and its related images
    _swapchain = vkbSwapchain.swapchain;
    _swapchainImages = vkbSwapchain.get_images().value();
    _swapchainImageViews = vkbSwapchain.get_image_views().value();
}

void VulkanEngine::resize_swapchain() {
    vkDeviceWaitIdle(_device);

    // when swapchain gives out of date error, there might be outgoing GPU work on swapchain images.
    //  wait for those work to finsih before destroy swapchain resources.
    destroy_swapchain();

    int w, h;
    SDL_GetWindowSize(_window, &w, &h);
    _windowExtent.width = w;
    _windowExtent.height = h;

    create_swapchain(_windowExtent.width, _windowExtent.height);

    resize_requested = false;
}


void VulkanEngine::destroy_swapchain() {
    // destroy swapchain resources
    for (int i = 0; i < _swapchainImageViews.size(); i++) {
        vkDestroyImageView(_device, _swapchainImageViews[i], nullptr);
    }
    // this will also destroy the swapchain images. 
    vkDestroySwapchainKHR(_device, _swapchain, nullptr);
}

// create image and image view.
AllocatedImage VulkanEngine::create_image(void* data, VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped)
{
    // assume RBGA is 4 bytes, and 1 byte/8 bits per channel.
    size_t data_size = size.depth * size.width * size.height * 4;
    // write on CPU, read by GPU. 
    AllocatedBuffer uploadbuffer = vkutil::create_buffer(_allocator, data_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
    // copy source data into the mapped data. 
    memcpy(uploadbuffer.info.pMappedData, data, data_size);

    // destination is an image. 
    AllocatedImage new_image = create_image(size, format, usage | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, mipmapped);

    // do an immediate submit that transfer from buffer to image.
    immediate_submit([&](VkCommandBuffer cmd) {
        vkutil::transition_image(cmd, new_image.image, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

        // copy from buffer to image.
        VkBufferImageCopy copyRegion = {};
        // src buffer. 
        copyRegion.bufferOffset = 0;
        // images are tightly packed in buffer based on image extent, or image size.
        copyRegion.bufferRowLength = 0;
        copyRegion.bufferImageHeight = 0;
        // dst image: copy to mip 0, the base image.
        copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copyRegion.imageSubresource.mipLevel = 0;
        copyRegion.imageSubresource.baseArrayLayer = 0;
        copyRegion.imageSubresource.layerCount = 1;
        copyRegion.imageExtent = size;

        // copy the buffer into the image
        vkCmdCopyBufferToImage(cmd, uploadbuffer.buffer, new_image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);

        vkutil::transition_image(cmd, new_image.image, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    });

    vkutil::destroy_buffer(_allocator, uploadbuffer);
    return new_image;
}

AllocatedImage VulkanEngine::create_image(VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped)
{
    // image format and size.
    AllocatedImage newImage;
    newImage.imageFormat = format;
    newImage.imageExtent = size;

    // image format, usage and size.
    VkImageCreateInfo img_info = vkinit::image_create_info(format, usage, size);
    if (mipmapped) {
        // formula for computing the number of mips for an image.
        img_info.mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(size.width, size.height)))) + 1;
    }

    // always allocate images on dedicated GPU memory
    VmaAllocationCreateInfo allocinfo = {};
    allocinfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    allocinfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    // allocate and create the image on GPU. 
    VK_CHECK(vmaCreateImage(_allocator, &img_info, &allocinfo, &newImage.image, &newImage.allocation, nullptr));

    // if the format is a depth format, we will need to have it use the correct
    // aspect flag
    VkImageAspectFlags aspectFlag = VK_IMAGE_ASPECT_COLOR_BIT;
    if (format == VK_FORMAT_D32_SFLOAT) {
        aspectFlag = VK_IMAGE_ASPECT_DEPTH_BIT;
    }

    // build a image-view for the image
    VkImageViewCreateInfo view_info = vkinit::imageview_create_info(format, newImage.image, aspectFlag);
    view_info.subresourceRange.levelCount = img_info.mipLevels;

    VK_CHECK(vkCreateImageView(_device, &view_info, nullptr, &newImage.imageView));

    return newImage;
}

void VulkanEngine::destroy_image(const AllocatedImage& img)
{
    vkDestroyImageView(_device, img.imageView, nullptr);
    vmaDestroyImage(_allocator, img.image, img.allocation);
}

// span is pointer + size pair, it doesn't copy the data from array or vector.
//  it specifies a sequence of items.
// Optimization: 
//  put this on a background thread and then send copies to the transfer queue. 
GPUMeshBuffers VulkanEngine::uploadMesh(std::span<uint32_t> indices, std::span<Vertex> vertices) {
    const size_t vertexBufferSize = vertices.size() * sizeof(Vertex);
    const size_t indexBufferSize = indices.size() * sizeof(uint32_t);

    GPUMeshBuffers newSurface;

    //create vertex buffer
    newSurface.vertexBuffer = vkutil::create_buffer(_allocator, vertexBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

    //find the adress of the vertex buffer
    VkBufferDeviceAddressInfo deviceAdressInfo{ .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .buffer = newSurface.vertexBuffer.buffer };
    // pointer math on vertexBufferAddress.
    newSurface.vertexBufferAddress = vkGetBufferDeviceAddress(_device, &deviceAdressInfo);

    //create index buffer
    newSurface.indexBuffer = vkutil::create_buffer(_allocator, indexBufferSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

    // update data with a CPU side STAGING buffer.
    AllocatedBuffer staging = vkutil::create_buffer(_allocator, vertexBufferSize + indexBufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);

    void* data = staging.allocation->GetMappedData();

    // copy vertex buffer
    memcpy(data, vertices.data(), vertexBufferSize);
    // copy index buffer
    memcpy((char*)data + vertexBufferSize, indices.data(), indexBufferSize);

    // no need to pieplien barrier as host writes are implicit memory dependency guarantee of vkQueueSubmit.
    //  if those writes happen before queuesubmit.
    // wait on CPU after GPU work is finished.
    //  capture by reference.
    immediate_submit([&](VkCommandBuffer cmd) {
        VkBufferCopy vertexCopy{ 0 };
        vertexCopy.dstOffset = 0;
        vertexCopy.srcOffset = 0;
        vertexCopy.size = vertexBufferSize;
        // copy from CPU staging to GPU only buffer.
        vkCmdCopyBuffer(cmd, staging.buffer, newSurface.vertexBuffer.buffer, 1, &vertexCopy);

        // copy index buffer.
        VkBufferCopy indexCopy{ 0 };
        indexCopy.dstOffset = 0;
        indexCopy.srcOffset = vertexBufferSize;
        indexCopy.size = indexBufferSize;
        // copy from CPU staging to GPU only buffer.
        vkCmdCopyBuffer(cmd, staging.buffer, newSurface.indexBuffer.buffer, 1, &indexCopy);
    });

    vkutil::destroy_buffer(_allocator, staging);

    return newSurface;
}

void VulkanEngine::init()
{
    // only one engine initialization is allowed with the application.
    assert(loadedEngine == nullptr);
    loadedEngine = this;

    // We initialize SDL and create a window with it.
    SDL_Init(SDL_INIT_VIDEO);

    SDL_WindowFlags window_flags = (SDL_WindowFlags)(SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);

    _window = SDL_CreateWindow(
        "Vulkan Engine",
        SDL_WINDOWPOS_UNDEFINED,
        SDL_WINDOWPOS_UNDEFINED,
        _windowExtent.width,
        _windowExtent.height,
        window_flags);

    init_vulkan();

    init_swapchain();

    init_commands();

    init_sync_structures();

    init_descriptors();

    init_pipelines();

    init_default_data();

    init_imgui();

    // initialize camera.
    mainCamera.velocity = glm::vec3(0.f);
    mainCamera.position = glm::vec3(30.f, -00.f, -085.f);

    mainCamera.pitch = 0;
    mainCamera.yaw = 0;

    std::filesystem::path structurePath = ASSET_ROOT_PATH / "structure.glb";
    auto structureFile = loadGltf(this, structurePath);
    assert(structureFile.has_value());

    loadedScenes["structure"] = std::move( *structureFile);

    // everything went fine
    _isInitialized = true;
}

void VulkanEngine::cleanup()
{
    if (_isInitialized) {
        //make sure the gpu has stopped doing its things
        vkDeviceWaitIdle(_device);

        loadedScenes.clear();

        // destroy command pool. don't destroy individual command buffers.
        for (int i = 0; i < FRAME_OVERLAP; i++) {
            //destroy sync objects
            vkDestroyFence(_device, _frames[i]._renderFence, nullptr);
            vkDestroySemaphore(_device, _frames[i]._renderSemaphore, nullptr);
            vkDestroySemaphore(_device, _frames[i]._acquireSemaphore, nullptr);

            vkDestroyCommandPool(_device, _frames[i]._commandPool, nullptr);

            _frames[i]._deletionQueue.flush();
        }
        // delete.
        _mainDeletionQueue.flush();

        // destroy swapchain, swapchain images and views, and swapchain handle.
        destroy_swapchain();
        // destroy surface, depending on window and instance.
        vkDestroySurfaceKHR(_instance, _surface, nullptr);

        // destroy device.
        vkDestroyDevice(_device, nullptr);

        // destroy instance.
        vkb::destroy_debug_utils_messenger(_instance, _debug_messenger);
        vkDestroyInstance(_instance, nullptr);

        SDL_DestroyWindow(_window);
    }

    // clear engine pointer
    loadedEngine = nullptr;
}

void VulkanEngine::immediate_submit(std::function<void(VkCommandBuffer cmd)>&& function)
{
    // reset fence to unsignaled
    VK_CHECK(vkResetFences(_device, 1, &_immFence));
    // reset command buffer for re-recording.
    VK_CHECK(vkResetCommandBuffer(_immCommandBuffer, 0));

    VkCommandBuffer cmd = _immCommandBuffer;
    // need to reset for reuse after submit.
    VkCommandBufferBeginInfo cmdBeginInfo = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

    VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));

    function(cmd);

    VK_CHECK(vkEndCommandBuffer(cmd));

    // just submit the prior commands and then wait for it to finish.
    VkCommandBufferSubmitInfo cmdinfo = vkinit::command_buffer_submit_info(cmd);
    VkSubmitInfo2 submit = vkinit::submit_info(&cmdinfo, nullptr, nullptr);

    // submit command buffer to the queue and execute it.
    //  _renderFence will now block until the graphic commands finish execution
    VK_CHECK(vkQueueSubmit2(_graphicsQueue, 1, &submit, _immFence));
    // wait until prior queue submits finish on the CPU.
    VK_CHECK(vkWaitForFences(_device, 1, &_immFence, true, MAX_TIMEOUT));
}

void VulkanEngine::record_draw() {
    mainDrawContext.OpaqueSurfaces.clear();
    mainDrawContext.TransparentSurfaces.clear();

    auto& monkey = loadedNodes["Suzanne"];
    // draw mesh.
    monkey->Draw(glm::mat4{ 1.f }, mainDrawContext);

    auto& cube = loadedNodes["Cube"];
    for (int x = -3; x < 3; x++) {
        glm::mat4 scale = glm::scale(glm::vec3{ 0.2f });
        glm::mat4 translation = glm::translate(glm::vec3{ x, 1, 0 });

        cube->Draw(translation * scale, mainDrawContext);
    }

    // read only resources on GPU can be read simutanesouly. 
    loadedScenes["structure"]->Draw(glm::mat4{ 1.f }, mainDrawContext);
}

void VulkanEngine::update_scene(float deltaTime)
{
    auto start = std::chrono::system_clock::now();

    mainCamera.update(deltaTime);
   
    // view matrix.
    sceneData.view = mainCamera.getViewMatrix();
    // camera projection
    sceneData.proj = glm::perspective(glm::radians(70.f), (float)_windowExtent.width / (float)_windowExtent.height, 10000.f, 0.1f);

    // invert the Y direction on projection matrix so that we are more similar
    // to opengl and gltf axis in the screen space (-1, 1).
    sceneData.proj[1][1] *= -1;
    sceneData.viewproj = sceneData.proj * sceneData.view;

    //some default lighting parameters
    sceneData.ambientColor = glm::vec4(.1f);
    sceneData.sunlightColor = glm::vec4(1.f);
    //sceneData.sunlightDirection = glm::vec4(0, 1, 0.1, 1.f);
    // traverse scene nodes
    record_draw();

    auto end = std::chrono::system_clock::now();
    //convert to microseconds (integer), and then come back to miliseconds
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    stats.scene_update_time = elapsed.count() / 1000.f;
}

void VulkanEngine::draw_background(VkCommandBuffer cmd, const FrameData& frame) {
    ComputeEffect& effect = backgroundEffects[currentBackgroundEffect];

    // bind the gradient drawing compute pipeline
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, effect.pipeline);

    // bind the descriptor set containing the draw image for the compute pipeline, bind descriptr set at set = 0.
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, effect.layout, 0, 1, &frame._drawImageDescriptors, 0, nullptr);

    // push constants.
    vkCmdPushConstants(cmd, effect.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ComputePushConstants), &effect.data);

    // execute the compute pipeline dispatch. We are using 16x16 workgroup size so we need to divide by it
    uint32_t DISPATCH_X = static_cast<uint32_t>(std::ceil(static_cast<double>(_drawExtent.width) / 16.0)), DISPATCH_Y = static_cast<uint32_t>(std::ceil(static_cast<double>(_drawExtent.height) / 16.0));
    vkCmdDispatch(cmd, DISPATCH_X, DISPATCH_Y, 1);
}

void VulkanEngine::draw_geometry(VkCommandBuffer cmd, const FrameData& frame)
{
    //reset counters
    stats.drawcall_count = 0;
    stats.triangle_count = 0;
    //begin clock
    auto start = std::chrono::system_clock::now();

    std::vector<uint32_t> opaque_draws, transparent_draws;
    opaque_draws.reserve(mainDrawContext.OpaqueSurfaces.size());
    transparent_draws.reserve(mainDrawContext.TransparentSurfaces.size());

    for (uint32_t i = 0; i < mainDrawContext.OpaqueSurfaces.size(); i++) {
        // CPU frustum culling.
        if (vkutil::is_visible(mainDrawContext.OpaqueSurfaces[i], sceneData.viewproj)) {
            opaque_draws.push_back(i);
        }
    }

    for (uint32_t i = 0; i < mainDrawContext.TransparentSurfaces.size(); i++) {
        // CPU frustum culling.
        if (vkutil::is_visible(mainDrawContext.TransparentSurfaces[i], sceneData.viewproj)) {
            transparent_draws.push_back(i);
        }
    }

    // sort the opaque surfaces by material and mesh
    std::sort(std::execution::par_unseq, opaque_draws.begin(), opaque_draws.end(), [&](const auto& iA, const auto& iB) {
        const RenderObject& A = mainDrawContext.OpaqueSurfaces[iA];
        const RenderObject& B = mainDrawContext.OpaqueSurfaces[iB];
        if (A.material == B.material) {
            // same material, sort by index buffer.
            return A.indexBuffer < B.indexBuffer;
        } else {
            if (A.material->pipeline == B.material->pipeline) {
                return A.material->materialSet < B.material->materialSet;
            }
            return A.material->pipeline < B.material->pipeline;
        }
    });

    // sort the transparent surfaces by camera distance
    std::sort(std::execution::par_unseq, transparent_draws.begin(), transparent_draws.end(), [&](const auto& iA, const auto& iB) {
        const RenderObject& A = mainDrawContext.TransparentSurfaces[iA];
        const RenderObject& B = mainDrawContext.TransparentSurfaces[iB];

        // transform bbox to world space.
        glm::vec4 A_origin = sceneData.view * A.transform * glm::vec4(A.bounds.origin, 1.0);
        glm::vec4 B_origin = sceneData.view * B.transform * glm::vec4(B.bounds.origin, 1.0);

        // view space comparison. 
        double distanceA = std::abs(A_origin.z);// glm::distance(mainCamera.position, glm::vec3(A_origin.x, A_origin.y, A_origin.z));
        double distanceB = std::abs(B_origin.z);// glm::distance(mainCamera.position, glm::vec3(B_origin.x, B_origin.y, B_origin.z));
        return distanceA > distanceB;
    });

    //allocate a new uniform buffer for the scene data
    //  write on CPU and accessed by GPU: GPU memory accessible by CPU (host visible); write on CPU with fast access on the GPU.
    AllocatedBuffer gpuSceneDataBuffer = vkutil::create_buffer(_allocator, sizeof(GPUSceneData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
    //add it to the deletion queue of this frame so it gets deleted once it has been used.
    get_current_frame()._deletionQueue.push_function([=, this]() {
        // destroy this buffer per frame. no barriers needed.
        vkutil::destroy_buffer(_allocator, gpuSceneDataBuffer);
    });
    //write the buffer. since the buffer is of type CPU write and GPU read, can skip transfer with vulkan commands from CPU to GPU.
    //  also can skip memory barrier as host writes are implicit dependency with vkQueueSubmit.
    GPUSceneData* sceneUniformData = (GPUSceneData*) gpuSceneDataBuffer.allocation->GetMappedData();
    *sceneUniformData = sceneData;
    //create a descriptor set that binds that buffer and update it
    VkDescriptorSet globalDescriptor = get_current_frame()._frameDescriptors.allocate(_device, _gpuSceneDataDescriptorLayout);
    // update descriptor binding for the global descriptor we allocate this frame.
    DescriptorWriter writer;
    writer.write_buffer(0, gpuSceneDataBuffer.buffer, sizeof(GPUSceneData), 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    writer.update_set(_device, globalDescriptor);

    //begin a render pass  connected to our draw image
    VkRenderingAttachmentInfo colorAttachment = vkinit::attachment_info(frame._drawImage.imageView, nullptr, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    VkRenderingAttachmentInfo depthAttachment = vkinit::depth_attachment_info(frame._depthImage.imageView, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);

    VkRenderingInfo renderInfo = vkinit::rendering_info(_drawExtent, &colorAttachment, &depthAttachment);
    vkCmdBeginRendering(cmd, &renderInfo);
    //vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _trianglePipeline);

    float width = static_cast<float>(_drawExtent.width);
    float height = static_cast<float>(_drawExtent.height);
    
    MaterialPipeline* lastPipeline = nullptr;
    MaterialInstance* lastMaterial = nullptr;
    VkBuffer lastIndexBuffer = VK_NULL_HANDLE;
    bool hasBoundGlobalDescriptor = false;

    VkViewport viewport = {};
    viewport.x = 0;
    viewport.y = 0;
    viewport.width = (float)_windowExtent.width;
    viewport.height = (float)_windowExtent.height;
    viewport.minDepth = 0.f;
    viewport.maxDepth = 1.f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor = {};
    scissor.offset.x = 0;
    scissor.offset.y = 0;
    scissor.extent.width = _windowExtent.width;
    scissor.extent.height = _windowExtent.height;
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    auto draw = [&](const RenderObject& draw) {
        // material has ds and pipeline: if material changes
        if (draw.material != lastMaterial) {
            lastMaterial = draw.material;
            //rebind pipeline and descriptors if the material changed: only 2 pipeline. 
            if (draw.material->pipeline != lastPipeline) {
                lastPipeline = draw.material->pipeline;
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, draw.material->pipeline->pipeline);
                if (!hasBoundGlobalDescriptor) {
                    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, draw.material->pipeline->layout, 0, 1, &globalDescriptor, 0, nullptr);
                    hasBoundGlobalDescriptor = true;
                }
            }
            // 
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, draw.material->pipeline->layout, 1, 1, &draw.material->materialSet, 0, nullptr);
        }

        //rebind index buffer if needed
        if (draw.indexBuffer != lastIndexBuffer) {
            lastIndexBuffer = draw.indexBuffer;
            // this is faster to switch.
            vkCmdBindIndexBuffer(cmd, draw.indexBuffer, 0, VK_INDEX_TYPE_UINT32);
        }

        GPUDrawPushConstants pushConstants;
        pushConstants.vertexBuffer = draw.vertexBufferAddress;
        pushConstants.worldMatrix = draw.transform;
        vkCmdPushConstants(cmd, draw.material->pipeline->layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(GPUDrawPushConstants), &pushConstants);

        vkCmdDrawIndexed(cmd, draw.indexCount, 1, draw.firstIndex, 0, 0);
        
        //add counters for triangles and draws
        stats.drawcall_count++;
        stats.triangle_count += draw.indexCount / 3;
    };

    for (auto& r : opaque_draws) {
        draw(mainDrawContext.OpaqueSurfaces[r]);
    }

    // need to properly sort transparent surfaces within the view frustum. 
    for (auto& r : transparent_draws) {
        draw(mainDrawContext.TransparentSurfaces[r]);
    }
    
    vkCmdEndRendering(cmd);

    auto end = std::chrono::system_clock::now();
    //convert to microseconds (integer), and then come back to miliseconds
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    stats.mesh_draw_time = elapsed.count() / 1000.f;
}

void VulkanEngine::draw(float deltaTime)
{
    update_scene(deltaTime);

    // upsampling only from draw image to swapchain. 
    _drawExtent.height = (uint32_t) (std::min(_swapchainExtent.height, get_current_frame()._drawImage.imageExtent.height) * renderScale);
    _drawExtent.width = (uint32_t) (std::min(_swapchainExtent.width, get_current_frame()._drawImage.imageExtent.width) * renderScale);

    // wait until the gpu has finished rendering the last frame. Timeout of 1
    //  meaning current frame GPU resources are reset and ready for reuse.
    VK_CHECK(vkWaitForFences(_device, 1, &get_current_frame()._renderFence, true, MAX_TIMEOUT));
    // fence signaled, need to make it unsignaled again.
    VK_CHECK(vkResetFences(_device, 1, &get_current_frame()._renderFence));
    
    /* dynamic per frame resource handling. */
    // flush deletion queue for this frame that just finish execution on the GPU.
    get_current_frame()._deletionQueue.flush();
    // rellocate descriptor sets every frame. reset all prior descriptors. 
    get_current_frame()._frameDescriptors.clear_pools(_device);

    // request image from the swapchain
    uint32_t swapchainImageIndex;
    // if acquireSemaphore is signaled on GPU, it means prior rendering to this swapchainImageIndex has finished presentation and all the work.
    VkResult e = vkAcquireNextImageKHR(_device, _swapchain, MAX_TIMEOUT, get_current_frame()._acquireSemaphore, nullptr, &swapchainImageIndex);
    if (e == VK_ERROR_OUT_OF_DATE_KHR) {
        resize_requested = true;
        return;
    }

    //naming it cmd for shorter writing
    VkCommandBuffer cmd = get_current_frame()._mainCommandBuffer;

    // now that we are sure that the commands finished executing, we can safely
    //  reset the command buffer to begin recording again
    //  remove all commands free its memory.
    VK_CHECK(vkResetCommandBuffer(cmd, 0));

    //begin the command buffer recording. We will use this command buffer exactly once, so we want to let vulkan know that
    // generally one time submit is good enough; each recording submitted once. 
    // afterwards command buffer needs to be reset and record again.
    VkCommandBufferBeginInfo cmdBeginInfo = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

    // why one image here is sufficient?
    auto& currentFrame = get_current_frame();
    auto& currentImage = currentFrame._drawImage;
    auto& currentDepthImage = currentFrame._depthImage;

    //start the command buffer recording
    VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));

    // pipeline barrier for prior usage of current image. 
    //vkutil::transition_image(cmd, currentImage.image, VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    vkutil::transition_image(cmd, currentImage.image, VK_PIPELINE_STAGE_2_BLIT_BIT, 0, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

    // clear background color.
    draw_background(cmd, currentFrame);
    
    vkutil::transition_image(cmd, currentImage.image, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    // wait until prior usage of the depth image to complete.
    vkutil::transition_image(cmd, currentDepthImage.image, VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);

    draw_geometry(cmd, currentFrame);

    // transfer draw image and swapchin image to transfer layouts.
    vkutil::transition_image(cmd, currentImage.image, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_READ_BIT, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    
    // acquire semaphore waits in the blit stage, so the first synchronization scope should be blit as well from the semaphore wait.
    vkutil::transition_image(cmd, _swapchainImages[swapchainImageIndex], VK_PIPELINE_STAGE_2_BLIT_BIT, 0, VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    // execute a copy from the draw image into the swapchain
    vkutil::copy_image_to_image(cmd, currentImage.image, _swapchainImages[swapchainImageIndex], _drawExtent, _swapchainExtent);

    // 
    vkutil::transition_image(cmd, _swapchainImages[swapchainImageIndex], VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    //draw imgui into the swapchain image
    draw_imgui(cmd, _swapchainImageViews[swapchainImageIndex]);

    // transition swapchain image from blit format to presentable format. 
    //  2nd synchronization scope 
    vkutil::transition_image(cmd, _swapchainImages[swapchainImageIndex], VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

    //finalize the command buffer (we can no longer add commands, but it can now be executed)
    VK_CHECK(vkEndCommandBuffer(cmd));

    // prepare the submission to the queue. 
    //  we want to wait on the _presentSemaphore, as that semaphore is signaled when the swapchain is ready
    //  we will signal the _renderSemaphore, to signal that rendering has finished
    VkCommandBufferSubmitInfo cmdinfo = vkinit::command_buffer_submit_info(cmd);
    // wait before blit for acquire semaphore signal (stage specify second synchronization scope).
    VkSemaphoreSubmitInfo waitInfo = vkinit::semaphore_submit_info(VK_PIPELINE_STAGE_2_BLIT_BIT, get_current_frame()._acquireSemaphore);
    // signal after blit for render semaphore (stage specify first synchronization scope).
    VkSemaphoreSubmitInfo signalInfo = vkinit::semaphore_submit_info(VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, get_current_frame()._renderSemaphore);

    // submit struct prepare.
    VkSubmitInfo2 submit = vkinit::submit_info(&cmdinfo, &signalInfo, &waitInfo);

    //submit command buffer to the queue and execute it.
    // _renderFence will now block until the graphic commands finish execution
    VK_CHECK(vkQueueSubmit2(_graphicsQueue, 1, &submit, get_current_frame()._renderFence));

    //prepare present
    // this will put the image we just rendered to into the visible window.
    // we want to wait on the _renderSemaphore for that, 
    // as its necessary that drawing commands have finished before the image is displayed to the user
    VkPresentInfoKHR presentInfo = {};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.pNext = nullptr;
    presentInfo.pSwapchains = &_swapchain;
    presentInfo.swapchainCount = 1;
    presentInfo.pWaitSemaphores = &get_current_frame()._renderSemaphore;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pImageIndices = &swapchainImageIndex;

    // submit to present queue.
    e = vkQueuePresentKHR(_graphicsQueue, &presentInfo);
    if (e == VK_ERROR_OUT_OF_DATE_KHR) {
        resize_requested = true;
        return;
    }

    //increase the number of frames drawn
    _frameNumber++;
}

// render to targetImageView for the UI.
void VulkanEngine::draw_imgui(VkCommandBuffer cmd, VkImageView targetImageView) {
    // target image view will be in layout_color_attachment_optimal during rendering.
    VkRenderingAttachmentInfo colorAttachment = vkinit::attachment_info(targetImageView, nullptr, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    VkRenderingInfo renderInfo = vkinit::rendering_info(_swapchainExtent, &colorAttachment, nullptr);

    vkCmdBeginRendering(cmd, &renderInfo);

    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);

    vkCmdEndRendering(cmd);
}

void VulkanEngine::run()
{
    SDL_Event e;
    bool bQuit = false;

    std::chrono::steady_clock::time_point lastUpdate = std::chrono::steady_clock::now();
    float deltaTime = 0.0;

    // main loop
    while (!bQuit) {
        //begin clock
        auto start = std::chrono::system_clock::now();

        if (resize_requested) {
            resize_swapchain();
        }

        // Handle events on queue, itearte through each type of event.
        while (SDL_PollEvent(&e) != 0) {
            // close the window when user alt-f4s or clicks the X button
            switch (e.type) {
            case SDL_QUIT: 
                bQuit = true;
                break;
            case SDL_WINDOWEVENT:
                if (e.window.event == SDL_WINDOWEVENT_MINIMIZED) {
                    stop_rendering = true;
                }
                if (e.window.event == SDL_WINDOWEVENT_RESTORED) {
                    stop_rendering = false;
                }
                break;
            default:
                break;
            }
            mainCamera.processSDLEvent(e);
            // send SDL process event.
            ImGui_ImplSDL2_ProcessEvent(&e);
        }

        // do not draw if we are minimized
        if (stop_rendering) {
            // throttle the speed to avoid the endless spinning
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        // imgui new frame
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplSDL2_NewFrame();
        ImGui::NewFrame();

        if (ImGui::Begin("UI")) {
            ComputeEffect& selected = backgroundEffects[currentBackgroundEffect];
            ImGui::Text("Selected effect: ", selected.name);
            ImGui::SliderInt("Effect Index", &currentBackgroundEffect, 0, std::max(0, ((int) backgroundEffects.size()) - 1));

            ImGui::InputFloat4("data1", (float*)&selected.data.data1);
            ImGui::InputFloat4("data2", (float*)&selected.data.data2);

            auto& meshesSelected = _testMeshes[currentMesh];
            ImGui::Text("Selected mesh: ", meshesSelected->name);
            ImGui::SliderInt("Mesh Index", &currentMesh, 0, std::max(0, ((int)_testMeshes.size()) - 1));
            
            auto& imageSelected = _defaultImages[selectedImage];
            ImGui::Text("Selected image: ", imageSelected.name);
            ImGui::SliderInt("Image Index", &selectedImage, 0, std::max(0, (int)(sizeof(_defaultImages) / sizeof(_defaultImages[0]) - 1)));

            float inputFloat[3] = { sceneData.sunlightDirection.x, sceneData.sunlightDirection.y, sceneData.sunlightDirection.z};
            ImGui::InputFloat3("Sun direction", inputFloat);
            sceneData.sunlightDirection = glm::vec4(glm::normalize(glm::vec3(inputFloat[0], inputFloat[1], inputFloat[2])), sceneData.sunlightDirection.w);

            ImGui::SliderFloat("Render Scale", &renderScale, 0.3f, 1.0f);
        }
        ImGui::End();

        ImGui::Begin("Stats");
        ImGui::Text("frametime %f ms", stats.frametime);
        ImGui::Text("draw time %f ms", stats.mesh_draw_time);
        ImGui::Text("update time %f ms", stats.scene_update_time);
        ImGui::Text("triangles %i", stats.triangle_count);
        ImGui::Text("draws %i", stats.drawcall_count);
        ImGui::End();

        //make imgui calculate internal draw structures
        ImGui::Render();

        // update draw with delta time.
        auto now = std::chrono::steady_clock::now();
        auto deltaTimeClock = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastUpdate);
        deltaTime = deltaTimeClock.count() / 1000.0f;   // delta time in seconds
        draw(deltaTime);
        lastUpdate = now;

        //get clock again, compare with start clock
        auto end = std::chrono::system_clock::now();
        //convert to microseconds (integer), and then come back to miliseconds
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        stats.frametime = elapsed.count() / 1000.f;
    }
}

