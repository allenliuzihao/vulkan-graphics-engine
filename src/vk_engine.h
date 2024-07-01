// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <vk_constants.h>
#include <vk_types.h>
#include <vk_descriptors.h>
#include <vk_loader.h>
#include <vk_buffer.h>
#include <vk_material.h>
#include <camera.h>

#include <filesystem>
#include <execution>

struct DeletionQueue
{
	std::deque<std::function<void()>> deletors;

	// growing lambda at the back of the queue.
	void push_function(std::function<void()>&& function) {
		deletors.push_back(function);
	}

	void flush() {
		// reverse iterate the deletion queue to execute all the functions
		//	stack. 
		for (auto it = deletors.rbegin(); it != deletors.rend(); it++) {
			(*it)(); //call functors
		}

		// clear lambdas. 
		deletors.clear();
	}
};

struct FrameData {
	VkSemaphore _acquireSemaphore, _renderSemaphore;
	VkFence _renderFence;

	VkCommandPool _commandPool;
	VkCommandBuffer _mainCommandBuffer;

	AllocatedImage _drawImage, _depthImage;
	VkDescriptorSet _drawImageDescriptors;

	DescriptorAllocatorGrowable _frameDescriptors;
	DeletionQueue _deletionQueue;
};

struct ComputePushConstants {
	glm::vec4 data1;
	glm::vec4 data2;
	//glm::vec4 data3;
	//glm::vec4 data4;
};

struct ComputeEffect {
	const char* name;

	VkPipeline pipeline;
	VkPipelineLayout layout;

	ComputePushConstants data;
};

struct EngineStats {
	float frametime;
	int triangle_count;
	int drawcall_count;
	float scene_update_time;
	float mesh_draw_time;
};

struct LoadedGLTF;
struct MeshAsset;

class VulkanEngine {
public:
	VkInstance _instance;// Vulkan library handle
	VkDebugUtilsMessengerEXT _debug_messenger;// Vulkan debug output handle
	VkPhysicalDevice _chosenGPU;// GPU chosen as the default device
	VkDevice _device; // Vulkan device for commands
	VkSurfaceKHR _surface;// Vulkan window surface

	// pipeline stuff.
	VkPipelineLayout _gradientPipelineLayout;
	// draw triangle.
	VkPipelineLayout _trianglePipelineLayout;
	VkPipeline _trianglePipeline;
	// mesh pipeline.
	VkPipelineLayout _meshPipelineLayout;
	VkPipeline _meshPipeline;
	GPUMeshBuffers _meshData;

	// immediate submit structures
	VkFence _immFence;
	VkCommandBuffer _immCommandBuffer;
	VkCommandPool _immCommandPool;
	void immediate_submit(std::function<void(VkCommandBuffer cmd)>&& function);

	// swapchain stuff.
	VkSwapchainKHR _swapchain;
	VkFormat _swapchainImageFormat;

	// number of outstanding swapchain images.
	std::vector<VkImage> _swapchainImages;
	std::vector<VkImageView> _swapchainImageViews;
	VkExtent2D _swapchainExtent;

	bool _isInitialized{ false };
	uint64_t _frameNumber {0};
	bool stop_rendering{ false };
	VkExtent2D _windowExtent{ 1920 ,1080 };

	struct SDL_Window* _window{ nullptr };

	static VulkanEngine& Get();

	//initializes everything in the engine
	void init();

	//shuts down the engine
	void cleanup();

	//draw loop
	void draw(float deltaTime);

	void draw_imgui(VkCommandBuffer cmd, VkImageView targetImageView);

	void draw_background(VkCommandBuffer cmd, const FrameData& frame);

	void draw_geometry(VkCommandBuffer cmd, const FrameData& frame);

	//run main loop
	void run();

	GPUSceneData sceneData;
	VkDescriptorSetLayout _gpuSceneDataDescriptorLayout;

	FrameData _frames[FRAME_OVERLAP];

	FrameData& get_active_frame(uint32_t acquiredImageIndex) { return _frames[acquiredImageIndex % FRAME_OVERLAP]; };
	FrameData& get_current_frame() { return _frames[_frameNumber % FRAME_OVERLAP]; };
	GPUMeshBuffers uploadMesh(std::span<uint32_t> indices, std::span<Vertex> vertices);

	DeletionQueue _mainDeletionQueue;

	VmaAllocator _allocator;

	// descriptors
	DescriptorAllocatorGrowable globalDescriptorAllocator;
	VkDescriptorSetLayout _drawImageDescriptorLayout;
	VkDescriptorSetLayout _singleImageDescriptorLayout;

	// graphics queue and its family.
	VkQueue _graphicsQueue;
	uint32_t _graphicsQueueFamily;

	// draw resources
	VkExtent2D _drawExtent;
	float renderScale = 1.f;

	// pipeline for drawing.
	std::vector<ComputeEffect> backgroundEffects;
	int currentBackgroundEffect{ 0 };

	std::vector<std::shared_ptr<MeshAsset>> _testMeshes;
	int currentMesh { 0 };
	bool resize_requested = false;

	// white, grey, black, magenta.
	AllocatedImage _defaultImages[4];
	int selectedImage = 3;

	VkSampler _defaultSamplerLinear;
	VkSampler _defaultSamplerNearest;

	MaterialInstance defaultData;
	GLTFMetallic_Roughness metalRoughMaterial;

	DrawContext mainDrawContext;
	std::unordered_map<std::string, std::shared_ptr<Node>> loadedNodes;
	std::unordered_map<std::string, std::shared_ptr<LoadedGLTF>> loadedScenes;
	void update_scene(float deltaTime);

	void record_draw();

	// camera.
	Camera mainCamera;
	EngineStats stats;

	// images. 
	AllocatedImage create_image(VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped = false);
	AllocatedImage create_image(void* data, VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped = false);
	void destroy_image(const AllocatedImage& img);
private:
	void init_default_data();
	void init_imgui();
	void init_vulkan();
	void init_descriptors();
	void init_swapchain();
	void init_commands();
	void init_sync_structures();
	void init_pipelines();
	void init_triangle_pipeline();
	void init_mesh_pipeline();
	void init_background_pipelines();

	void create_swapchain(uint32_t width, uint32_t height);
	void destroy_swapchain();
	void resize_swapchain();

	bool is_visible(const RenderObject& obj, const glm::mat4& viewproj);
};
