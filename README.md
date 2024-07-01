This is a lightweight rendering engine written in Vulkan based on the Vulkan Guide post: https://vkguide.dev/, which introduces the modern GPU features for real-time rendering.

Some of the modern engine features include:

1. drawing vertices using GPU memory pointer without using vertex buffer.
2. CPU frustum culling to cull draw calls.
3. Draw call sorting of transparent and opaque objects, respectively.
4. More precise synchronization and pipeline barrier compared to the original project.
