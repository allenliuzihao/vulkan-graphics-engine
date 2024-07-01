#include <vk_utility.h>

bool vkutil::is_visible(const RenderObject& obj, const glm::mat4& viewproj) {
    // box corners.
    std::array<glm::vec3, 8> corners{
        glm::vec3 { 1, 1, 1 },
        glm::vec3 { 1, 1, -1 },
        glm::vec3 { 1, -1, 1 },
        glm::vec3 { 1, -1, -1 },
        glm::vec3 { -1, 1, 1 },
        glm::vec3 { -1, 1, -1 },
        glm::vec3 { -1, -1, 1 },
        glm::vec3 { -1, -1, -1 },
    };

    // project corner from object space to clip space.
    glm::mat4 matrix = viewproj * obj.transform;
    // minimum and maximum of corners.
    glm::vec3 min = { 1.5, 1.5, 1.5 };
    glm::vec3 max = { -1.5, -1.5, -1.5 };

    for (int c = 0; c < 8; c++) {
        // project each corner from object space into clip space
        glm::vec4 v = matrix * glm::vec4(obj.bounds.origin + (corners[c] * obj.bounds.extents), 1.f);

        // perspective correction
        v.x = v.x / v.w;
        v.y = v.y / v.w;
        v.z = v.z / v.w;

        // find clip space min max bounding box.
        min = glm::min(glm::vec3{ v.x, v.y, v.z }, min);
        max = glm::max(glm::vec3{ v.x, v.y, v.z }, max);
    }

    // check the clip space box is within the view
    //  z: [0, 1]
    //  x: [-1, 1]
    //  y: [-1, 1]
    // this checks if the bounding box overlaps with the view frustum. 
    //  if so, it thinks the object is within view (but might not actually be if only bbox intersects viewing volumn).
    if (min.z > 1.f || max.z < 0.f || min.x > 1.f || max.x < -1.f || min.y > 1.f || max.y < -1.f) {
        return false;
    } else {
        return true;
    }
}

void vkutil::ImmediateSubmit::immediate_submit(VkDevice device, VkQueue queue, std::function<void(VkCommandBuffer cmd)>&& function)
{
    // reset fence to unsignaled
    VK_CHECK(vkResetFences(device, 1, &_immFence));
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
    VK_CHECK(vkQueueSubmit2(queue, 1, &submit, _immFence));
    // wait until prior queue submits finish on the CPU.
    VK_CHECK(vkWaitForFences(device, 1, &_immFence, true, MAX_TIMEOUT));
}