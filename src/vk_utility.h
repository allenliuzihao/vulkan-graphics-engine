#pragma once

#include <vk_types.h>

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

    bool is_visible(const RenderObject& obj, const glm::mat4& viewproj);
};