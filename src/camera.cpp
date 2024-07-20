#include <camera.h>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/quaternion.hpp>

void Camera::update(float deltaTime)
{
    // camera to world rotation.
    glm::mat4 cameraRotation = getRotationMatrix();
    position += glm::vec3(cameraRotation * glm::vec4(velocity * deltaTime, 0.f));
}

glm::mat4 Camera::getViewMatrix()
{
    // to create a correct model view, we need to move the world in opposite
    // direction to the camera
    //  so we will create the camera model matrix and invert
    glm::mat4 cameraTranslation = glm::translate(glm::mat4(1.f), position);
    glm::mat4 cameraRotation = getRotationMatrix();
    // inverse of camera to world -> world to camera.
    return glm::inverse(cameraTranslation * cameraRotation);
}

glm::mat4 Camera::getRotationMatrix()
{
    // fairly typical FPS style camera. we join the pitch and yaw rotations into
    // the final rotation matrix

    // vertical rotation in world.
    glm::quat pitchRotation = glm::angleAxis(pitch, glm::vec3{ 1.f, 0.f, 0.f });
    // horizontal rotation.
    glm::quat yawRotation = glm::angleAxis(yaw, glm::vec3{ 0.f, -1.f, 0.f });

    // camera to world after rotation. 
    return glm::toMat4(yawRotation) * glm::toMat4(pitchRotation);
}

void Camera::processSDLEvent(SDL_Event& e)
{
    static bool enableMouseMotion = false;

    // improve event handling.
    if (e.type == SDL_KEYDOWN) {
        if (e.key.keysym.sym == SDLK_w) {
            pressDownW = true;
        } else if (e.key.keysym.sym == SDLK_s) {
            pressDownS = true;
        } else if (e.key.keysym.sym == SDLK_a) {
            pressDownA = true;
        } else if (e.key.keysym.sym == SDLK_d) {
            pressDownD = true;
        }
        //fmt::println("pressed {} down", e.key.keysym.sym);
    } else if (e.type == SDL_KEYUP) {
        if (e.key.keysym.sym == SDLK_w) {
            pressDownW = false;
        } else if (e.key.keysym.sym == SDLK_s) {
            pressDownS = false;
        } else if (e.key.keysym.sym == SDLK_a) {
            pressDownA = false;
        } else if (e.key.keysym.sym == SDLK_d) {
            pressDownD = false;
        }
    } else if (e.type == SDL_MOUSEBUTTONDOWN && (SDL_GetMouseState(nullptr, nullptr) & SDL_BUTTON_RMASK)) {
        fmt::println("mouse button down.");
        enableMouseMotion = true;
    } else if (e.type == SDL_MOUSEBUTTONUP) {
        fmt::println("mouse button up.");
        enableMouseMotion = false;
    } else if (e.type == SDL_MOUSEMOTION) {
        fmt::println("mouse motion.");
        if (enableMouseMotion) {
            yaw += (float)e.motion.xrel / 200.f;
            pitch -= (float)e.motion.yrel / 200.f;
        }
    }

    if (!pressDownS && !pressDownW) {
        velocity.z = 0;
    } else if (!pressDownS && pressDownW) {
        velocity.z = -1;
    } else if (pressDownS && !pressDownW) {
        velocity.z = 1;
    } else {
        velocity.z = 0;
    }

    if (!pressDownA && !pressDownD) {
        velocity.x = 0;
    } else if (!pressDownA && pressDownD) {
        velocity.x = 1;
    } else if (pressDownA && !pressDownD) {
        velocity.x = -1;
    } else {
        velocity.x = 0;
    }

    velocity *= 5;
}