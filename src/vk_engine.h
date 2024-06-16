// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <vk_types.h>

class VulkanEngine {
public:
	VkInstance _instance;// Vulkan library handle
	VkDebugUtilsMessengerEXT _debug_messenger;// Vulkan debug output handle
	VkPhysicalDevice _chosenGPU;// GPU chosen as the default device
	VkDevice _device; // Vulkan device for commands
	VkSurfaceKHR _surface;// Vulkan window surface

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
	void draw();

	//run main loop
	void run();
private:
	void init_vulkan();
	void init_swapchain();
	void init_commands();
	void init_sync_structures();

	// TODO: create swapchain.
	void create_swapchain(uint32_t width, uint32_t height);
	void destroy_swapchain();
};
