#pragma once

#include <vk_constants.h>
#include <vk_types.h>
#include <vk_initializers.h>
#include <vk_constants.h>

namespace vkutil {
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

    bool is_point_within_interval(float point, float MIN, float MAX);
    bool is_point_within_box(glm::vec3 pos, glm::vec3 MIN, glm::vec3 MAX);
    bool is_interval_overlap(float interval1Min, float interval1Max, float interval2Min, float interval2Max);
    bool is_visible(const RenderObject& obj, const glm::mat4& viewproj);

    struct ImmediateSubmit {
        // immediate submit structures
        VkFence _immFence;
        VkCommandBuffer _immCommandBuffer;
        VkCommandPool _immCommandPool;
        void immediate_submit(VkDevice device, VkQueue queue, std::function<void(VkCommandBuffer cmd)>&& function);
    };
};