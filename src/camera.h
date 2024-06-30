
#include <vk_types.h>
#include <SDL_events.h>
#include <stack>

#include <vk_types.h>

class Camera {
public:
    // velocity in camera space.
    glm::vec3 velocity;
    // position in world space.
    glm::vec3 position;
    // vertical rotation
    float pitch{ 0.f };
    // horizontal rotation
    float yaw{ 0.f };

    glm::mat4 getViewMatrix();
    glm::mat4 getRotationMatrix();

    bool pressDownW = false, pressDownA = false, pressDownS = false, pressDownD = false;

    void processSDLEvent(SDL_Event& e);

    void update(float deltaTime);
};
