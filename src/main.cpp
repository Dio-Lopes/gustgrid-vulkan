#define GLFW_INCLUDE_VULKAN
#include <simulator.h>
#include <vulkan_utils.h>
#include <glfw/include/GLFW/glfw3.h>
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>
#include <OBJ-Loader/Source/OBJ_Loader.h>
#include <freetype/include/ft2build.h>
#include FT_FREETYPE_H
#include <omp.h>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <optional>
#include <map>
#include <set>
#include <fstream>
#include <limits>
#include <algorithm>
#include <array>

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

#define PI 3.14159265358979323846f

#define gridSizeX 64
#define gridSizeY 256
#define gridSizeZ 128
#define worldMinX -2.0f
#define worldMaxX 2.0f
#define worldMinY -4.5f
#define worldMaxY 4.5f
#define worldMinZ -4.0f
#define worldMaxZ 4.0f
#define renderMinX -1.65f
#define renderMaxX 1.7f
#define renderMinY -3.0f
#define renderMaxY 4.22f
#define renderMinZ -3.4f
#define renderMaxZ 3.4f
#define cellSizeX ((worldMaxX - worldMinX) / gridSizeX)
#define cellSizeY ((worldMaxY - worldMinY) / gridSizeY)
#define cellSizeZ ((worldMaxZ - worldMinZ) / gridSizeZ)
#define maxFans 8

const int MAX_FRAMES_IN_FLIGHT = 2;
uint32_t currentFrame = 0;

const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};
std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
#ifdef __APPLE__
    , "VK_KHR_portability_subset"
#endif
};
#ifdef NDEBUG
    const bool enableValidationLayers = false;
#else
    const bool enableValidationLayers = true;
#endif

VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger){
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if(func != nullptr) return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    else return VK_ERROR_EXTENSION_NOT_PRESENT;
}
void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator){
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if(func != nullptr) func(instance, debugMessenger, pAllocator);
}
struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;
    bool isComplete() {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};
struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};
struct Vertex {
    glm::vec3 pos;
    glm::vec2 texCoord;
    glm::vec3 normal;
    static VkVertexInputBindingDescription getBindingDescription(){
        VkVertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Vertex);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        return bindingDescription;
    }
    static std::array<VkVertexInputAttributeDescription, 3> getAttributeDescriptions(){
        std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions{};
        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(Vertex, pos);
        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[1].offset = offsetof(Vertex, normal);
        attributeDescriptions[2].binding = 0;
        attributeDescriptions[2].location = 2;
        attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
        attributeDescriptions[2].offset = offsetof(Vertex, texCoord);
        return attributeDescriptions;
    }
};
struct TextVertex{
    glm::vec2 pos;
    glm::vec2 texCoord;
    static VkVertexInputBindingDescription getBindingDescription(){
        VkVertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(TextVertex);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        return bindingDescription;
    }
    static std::array<VkVertexInputAttributeDescription, 2> getAttributeDescriptions(){
        std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions{};
        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(TextVertex, pos);
        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32G32_SFLOAT;
        attributeDescriptions[1].offset = offsetof(TextVertex, texCoord);
        return attributeDescriptions;
    }
};
struct UIPushConstants {
    glm::vec3 color;
    uint32_t isUI;
};
struct UniformBufferObject {
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 proj;
    glm::vec3 cameraPos;
};
struct ModelData{
    std::string name;
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    VkBuffer vertexBuffer;
    VkSampler textureSampler[4];
    VkDeviceMemory vertexBufferMemory;
    VkBuffer indexBuffer;
    VkDeviceMemory indexBufferMemory;
    uint32_t mipmapLevels[4];
    VkImage textureImage[4];
    VkImageView textureImageView[4];
    VkDeviceMemory textureImageMemory[4];
    std::vector<VkDescriptorSet> descriptorSets;
    std::vector<VkBuffer> uniformBuffers;
    std::vector<VkDeviceMemory> uniformBuffersMemory;
    std::vector<void*> uniformBuffersMapped;
    glm::mat4 modelMatrix = glm::mat4(1.0f);
    bool enabled = true;
    bool loaded = false;
};
struct TextureData {
    VkImage image;
    VkImageView imageView;
    VkDeviceMemory imageMemory;
};
struct VolumeData {
    VkImage volumeImage;
    VkImageView volumeImageView;
    VkDeviceMemory volumeImageMemory;
    VkImage temperatureImage;
    VkImageView temperatureImageView;
    VkDeviceMemory temperatureImageMemory;
    std::vector<VkDescriptorSet> descriptorSets;
    VkBuffer vertexBuffer;
    VkDeviceMemory vertexBufferMemory;
    VkBuffer indexBuffer;
    VkDeviceMemory indexBufferMemory;
    std::vector<VkBuffer> uniformBuffers;
    std::vector<VkDeviceMemory> uniformBuffersMemory;
    std::vector<void*> uniformBuffersMapped;
};
struct VolumeUniformBufferObject {
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 projection;
    alignas(16) glm::vec3 camPos;
    alignas(16) glm::vec3 gridSize;
    alignas(16) glm::vec3 worldMin;
    alignas(16) glm::vec3 worldMax;
    alignas(4) int displayPressure = 0;
    alignas(4) float stepSize;
};
struct ModelOrientation {
    glm::vec3 position = glm::vec3(0.0f);
    glm::vec3 rotation = glm::vec3(0.0f);
    glm::vec3 scale = glm::vec3(1.0f);
};
struct Character{
    VkImage textureImage;
    VkImageView textureImageView;
    VkDeviceMemory textureImageMemory;
    VkDescriptorSet descriptorSet;
    glm::ivec2 size;
    glm::ivec2 bearing;
    unsigned int advance;
};
struct TextData{
    std::string text;
    glm::vec3 color = glm::vec3(1.0f);
    glm::vec2 position = glm::vec2(0.0f);
    float scale = 1.0f;
    bool anchorLeft = true;
    bool anchorTop = true;
    bool enabled = true;
};
struct UIData{
    VkImage textureImage;
    VkImageView textureImageView;
    VkDeviceMemory textureImageMemory;
    VkDescriptorSet descriptorSet;
    VkBuffer vertexBuffer;
    VkDeviceMemory vertexBufferMemory;
    void* vertexBufferMapped;
    glm::vec2 size = glm::vec2(1.0f);
    glm::vec2 position = glm::vec2(0.0f);
    std::string name;
    bool anchorLeft = true;
    bool enabled = true;
};

class GustGrid {
public:
    void run(){
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }
private:
    GLFWwindow* window;
    VkInstance instance;
    VkDebugUtilsMessengerEXT debugMessenger;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device;
    VkQueue graphicsQueue;
    VkQueue presentQueue;
    VkSurfaceKHR surface;
    VkSwapchainKHR swapChain;
    std::vector<VkImage> swapChainImages;
    std::vector<VkImageView> swapChainImageViews;
    std::vector<VkFramebuffer> swapChainFramebuffers;
    VkFormat swapChainImageFormat;
    VkExtent2D swapChainExtent;
    VkRenderPass renderPass;
    VkPipelineLayout pipelineLayout;
    VkPipeline graphicsPipeline;
    VkCommandPool commandPool;
    std::vector<VkCommandBuffer> commandBuffers;
    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence> inFlightFences;
    VkDescriptorSetLayout descriptorSetLayout;
    VkDescriptorPool descriptorPool;
    VkSampler textureSampler;
    VkSampler volumeSampler;
    VkImage depthImage;
    VkDeviceMemory depthImageMemory;
    VkImageView depthImageView;
    VkSampleCountFlagBits msaaSamples = VK_SAMPLE_COUNT_1_BIT;
    VkImage colorImage;
    VkDeviceMemory colorImageMemory;
    VkImageView colorImageView;
    VkBuffer textVertexBuffer;
    VkDeviceMemory textVertexBufferMemory;
    void* textVertexBufferMapped;
    VkBuffer textIndexBuffer;
    VkDeviceMemory textIndexBufferMemory;
    VkBuffer uiIndexBuffer;
    VkDeviceMemory uiIndexBufferMemory;
    void* uiIndexBufferMapped;
    std::vector<TextData> textObjects = {
        {.text = "", .color = glm::vec3(1.0f), .position = {50.0f, 50.0f}, .scale = 0.8f},
        {.text = "100°C", .color = glm::vec3(1.0f), .position = {90.0f, 200.0f}, .scale = 0.6f},
        {.text = "61°C", .color = glm::vec3(1.0f), .position = {90.0f, 430.0f}, .scale = 0.6f},
        {.text = "22°C", .color = glm::vec3(1.0f), .position = {90.0f, 680.0f}, .scale = 0.6f},
        {.text = "Menu", .color = glm::vec3(0.0f), .position = {420.0f, 70.0f}, .scale = 0.9f, .anchorLeft = false},
        {.text = "GPU Enabled", .color = glm::vec3(0.0f), .position = {420.0f, 130.0f}, .scale = 0.6f, .anchorLeft = false},
        {.text = "Top Fan Enabled", .color = glm::vec3(0.0f), .position = {420.0f, 180.0f}, .scale = 0.6f, .anchorLeft = false},
        {.text = "CPU Fan Enabled", .color = glm::vec3(0.0f), .position = {420.0f, 230.0f}, .scale = 0.6f, .anchorLeft = false},
        {.text = "Front Fan Enabled", .color = glm::vec3(0.0f), .position = {420.0f, 280.0f}, .scale = 0.6f, .anchorLeft = false},
        {.text = "Back Fans:", .color = glm::vec3(0.0f), .position = {420.0f, 330.0f}, .scale = 0.6f, .anchorLeft = false},
        {.text = "Fan 1:", .color = glm::vec3(0.0f), .position = {420.0f, 380.0f}, .scale = 0.6f, .anchorLeft = false},
        {.text = "Fan 2:", .color = glm::vec3(0.0f), .position = {420.0f, 430.0f}, .scale = 0.6f, .anchorLeft = false},
        {.text = "Fan 3:", .color = glm::vec3(0.0f), .position = {420.0f, 480.0f}, .scale = 0.6f, .anchorLeft = false},
        {.text = "Show Pressure", .color = glm::vec3(0.0f), .position = {420.0f, 670.0f}, .scale = 0.6f, .anchorLeft = false},
        {.text = "CPU Temperature", .color = glm::vec3(1.0f), .position = {320.0f, 80.0f}, .scale = 0.5f, .anchorLeft = false, .anchorTop = false},
        {.text = "GPU Temperature", .color = glm::vec3(1.0f), .position = {320.0f, 40.0f}, .scale = 0.5f, .anchorLeft = false, .anchorTop = false}
    };
    VkPipeline textPipeline;
    VkPipelineLayout textPipelineLayout;
    VkDescriptorSetLayout textDescriptorSetLayout;
    VkDescriptorPool textDescriptorPool;
    VkDescriptorSetLayout uiDescriptorSetLayout;
    VkDescriptorPool uiDescriptorPool;
    VkPipeline volumePipeline;
    VkPipelineLayout volumePipelineLayout;
    VkDescriptorSetLayout volumeDescriptorSetLayout;
    VkDescriptorPool volumeDescriptorPool;
    VolumeData volumeData;
    VolumeSimulator* volumeSimulator;
    VkDescriptorSetLayout computeDescriptorSetLayout;
    VkDescriptorPool computeDescriptorPool;
    bool gpuEnabled = true;
    bool cpuFanEnabled = true;
    bool frontFanEnabled = true;
    bool topFanEnabled = true;
    bool pressureEnabled = true;
    float backFanLocations[3] = {0.0f, -2.5f, 1.0f};
    bool showUI = true;
    std::string hoverElement = "";
    std::map<std::string, ModelData> models = {
        {"case", {}},
        {"shield", {}},
        {"ram", {}},
        {"cpu", {}},
        {"gpu", {.enabled = gpuEnabled}},
        {"gpufan1", {.enabled = gpuEnabled}},
        {"gpufan2", {.enabled = gpuEnabled}},
        {"ioshield", {}},
        {"motherboard", {}},
        {"psu", {}},
        {"glass", {}},
        {"cpufan", {.enabled = cpuFanEnabled}},
        {"frontfan", {.enabled = frontFanEnabled}},
        {"topfan", {.enabled = topFanEnabled}},
        {"fancpu", {.enabled = cpuFanEnabled}},
        {"fanfront", {.enabled = frontFanEnabled}},
        {"fantop", {.enabled = topFanEnabled}},
        {"backfan1", {}},
        {"backfan2", {}},
        {"backfan3", {}},
        {"fanback1", {}},
        {"fanback2", {}},
        {"fanback3", {}},
    };
    std::map<std::string, TextureData> textureAtlas = {
        {"thermalmap", {}},
        {"checkbox/checked", {}},
        {"checkbox/unchecked", {}},
        {"arrow/up", {}},
        {"arrow/down", {}},
        {"slider/knob", {}},
        {"toggle/open", {}},
        {"toggle/close", {}},
        {"uiWindow", {}}
    };
    std::map<std::string, UIData> uiObjects = {
        {"thermalmap", {.name = "thermalmap", .position = glm::vec2(50.0f, 700.0f), .size = glm::vec2(30.0f, -500.0f)}},
        {"gpuCheckbox", {.name = "checkbox/checked", .position = glm::vec2(80.0f, 125.0f), .size = glm::vec2(30.0f, 30.0f), .anchorLeft = false}},
        {"topFanCheckbox", {.name = "checkbox/checked", .position = glm::vec2(80.0f, 175.0f), .size = glm::vec2(30.0f, 30.0f), .anchorLeft = false}},
        {"cpuFanCheckbox", {.name = "checkbox/checked", .position = glm::vec2(80.0f, 225.0f), .size = glm::vec2(30.0f, 30.0f), .anchorLeft = false}},
        {"frontFanCheckbox", {.name = "checkbox/checked", .position = glm::vec2(80.0f, 275.0f), .size = glm::vec2(30.0f, 30.0f), .anchorLeft = false}},
        {"pressureCheckbox", {.name = "checkbox/checked", .position = glm::vec2(80.0f, 665.0f), .size = glm::vec2(30.0f, 30.0f), .anchorLeft = false}},
        {"fanUp", {.name = "arrow/down", .position = glm::vec2(180.0f, 325.0f), .size = glm::vec2(30.0f, 15.0f), .anchorLeft = false}},
        {"fanDown", {.name = "arrow/up", .position = glm::vec2(180.0f, 340.0f), .size = glm::vec2(30.0f, 15.0f), .anchorLeft = false}},
        {"uiWindow", {.name = "uiWindow", .position = glm::vec2(50.0f, 50.0f), .size = glm::vec2(400.0f, 700.0f), .anchorLeft = false}},
        {"slider1", {.name = "checkbox/unchecked", .position = glm::vec2(80.0f, 375.0f), .size = glm::vec2(230.0f, 30.0f), .anchorLeft = false}},
        {"slider2", {.name = "checkbox/unchecked", .position = glm::vec2(80.0f, 425.0f), .size = glm::vec2(230.0f, 30.0f), .anchorLeft = false}},
        {"slider3", {.name = "checkbox/unchecked", .position = glm::vec2(80.0f, 475.0f), .size = glm::vec2(230.0f, 30.0f), .anchorLeft = false}},
        {"sliderknob1", {.name = "slider/knob", .position = glm::vec2(80.0f, 375.0f), .size = glm::vec2(30.0f, 30.0f), .anchorLeft = false}},
        {"sliderknob2", {.name = "slider/knob", .position = glm::vec2(80.0f, 425.0f), .size = glm::vec2(30.0f, 30.0f), .anchorLeft = false}},
        {"sliderknob3", {.name = "slider/knob", .position = glm::vec2(80.0f, 475.0f), .size = glm::vec2(30.0f, 30.0f), .anchorLeft = false}},
        {"uiToggle", {.name = "toggle/close", .position = glm::vec2(470.0f, 50.0f), .size = glm::vec2(40.0f, 40.0f), .anchorLeft = false}}
    };
    float camRadius = 15.0f;
    float camPitch = PI / 12.0f;
    float camYaw = 4.0f * PI / 8.0f;
    glm::vec3 camPos = glm::vec3(
        sin(camYaw) * cos(camPitch) * camRadius,
        sin(camPitch) * camRadius,
        cos(camYaw) * cos(camPitch) * camRadius
    );
    float camFOV = 45.0f;
    bool firstMouse = true;
    float lastMouseX = WIDTH / 2.0f;
    float lastMouseY = HEIGHT / 2.0f;
    float mouseSensitivity = 0.007f;
    bool framebufferResized = false;
    float currentTime = 0.0f;
    float fanStrength = 10.0f;
    bool shouldUpdateFans = true;
    bool shouldUpdateGrid = true;
    int maxPressureIterations = 8;
    float pressureTolerance = 1e-4f;
    VolumeSimulator::ComputePushConstants currentPushConstants = {};
    float dt = 0.0f;

    std::map<char, Character> Characters;
    void prepareCharacters(){
        FT_Library ft;
        if(FT_Init_FreeType(&ft))
            throw std::runtime_error("Failed to initialize FreeType library");
        FT_Face face;
        if(FT_New_Face(ft, "src/fonts/Lato.ttf", 0, &face))
            throw std::runtime_error("Failed to load font");
        FT_Set_Pixel_Sizes(face, 0, 48);
        FT_ULong codepoints[129];
        for(FT_ULong c = 0; c<128; c++) codepoints[c] = c;
        codepoints[128] = 0x00B0;
        for(FT_ULong c : codepoints){
            if(FT_Load_Char(face, c, FT_LOAD_RENDER)){
                std::cerr<<"Failed to load character: "<<(int) c<<std::endl;
                continue;
            }
            Character character;
            character.size = glm::ivec2(face->glyph->bitmap.width, face->glyph->bitmap.rows);
            character.bearing = glm::ivec2(face->glyph->bitmap_left, face->glyph->bitmap_top);
            character.advance = face->glyph->advance.x;
            if(face->glyph->bitmap.width == 0 || face->glyph->bitmap.rows == 0 || face->glyph->bitmap.buffer == nullptr) {
                unsigned char dummyPixel = 0;
                createTextureImage(1, 1, &dummyPixel, character.textureImage, character.textureImageMemory, VK_FORMAT_R8_UNORM);
            } else createTextureImage(face->glyph->bitmap.width, face->glyph->bitmap.rows, face->glyph->bitmap.buffer, character.textureImage, character.textureImageMemory, VK_FORMAT_R8_UNORM);
            createTextureImageView(VK_FORMAT_R8_UNORM, character.textureImage, character.textureImageView);
            Characters[c] = character;
        }
        FT_Done_Face(face);
        FT_Done_FreeType(ft);
    }

    void initWindow(){
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        window = glfwCreateWindow(WIDTH, HEIGHT, "GustGrid", nullptr, nullptr);
        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
        glfwSetCursorPosCallback(window, mouseCallback);
    }
    void initVulkan(){
        createInstance();
        setupDebugMessenger();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapChain();
        createImageViews();
        createRenderPass();
        createDescriptorSetLayout();
        createGraphicsPipeline();
        createCommandPool();
        createColorResources();
        createDepthResources();
        createTextureImage();
        createTextureImageView();
        createFramebuffers();
        createTextureSampler();
        loadModel();
        createVertexBuffer();
        createIndexBuffer();
        createUniformBuffers();
        createDescriptorPool();
        createDescriptorSets();
        prepareCharacters();
        createTextDescriptorSetLayout();
        createTextDescriptorPool();
        createTextDescriptorSets();
        createTextResources();
        prepareTextureAtlas();
        createUIDescriptorSetLayout();
        createUIDescriptorPool();
        createUIDescriptorSets();
        createUIResources();
        createComputeDescriptorSetLayout();
        createComputeDescriptorPool();
        volumeSimulator = new VolumeSimulator(device, commandPool, graphicsQueue, physicalDevice);
        volumeSimulator->initialize(computeDescriptorSetLayout, computeDescriptorPool);
        int numCells = gridSizeX * gridSizeY * gridSizeZ;
        volumeSimulator->initSimulation(numCells);
        volumeSimulator->addKernel("velocityUpdate", "src/shaders/compiled/velocityupdate.comp.spv", glm::uvec3(8, 8, 8), true);
        volumeSimulator->addKernel("fanUpdate", "src/shaders/compiled/fanaccessupdate.comp.spv", glm::uvec3(8, 8, 8), true);
        volumeSimulator->addKernel("computeDivergence", "src/shaders/compiled/computedivergence.comp.spv", glm::uvec3(8, 8, 8), true);
        volumeSimulator->addKernel("pressureJacobi", "src/shaders/compiled/pressurejacobi.comp.spv", glm::uvec3(8, 8, 8), true);
        volumeSimulator->addKernel("computeResidual", "src/shaders/compiled/computeresidual.comp.spv", glm::uvec3(8, 8, 8), true);
        volumeSimulator->addKernel("subtractPressureGradient", "src/shaders/compiled/subtractpressuregradient.comp.spv", glm::uvec3(8, 8, 8), true);
        volumeSimulator->addKernel("enforceBoundaryConditions", "src/shaders/compiled/enforceboundaryconditions.comp.spv", glm::uvec3(8, 8, 8), true);
        volumeSimulator->addKernel("computeAdvection", "src/shaders/compiled/computeadvection.comp.spv", glm::uvec3(8, 8, 8), true);
        volumeSimulator->addKernel("normalizeForward", "src/shaders/compiled/normalizeforward.comp.spv", glm::uvec3(8, 8, 8), true);
        volumeSimulator->addKernel("convectiveHeat", "src/shaders/compiled/convectiveheat.comp.spv", glm::uvec3(8, 8, 8), true);
        volumeSimulator->addKernel("diffuseKernel", "src/shaders/compiled/diffusekernel.comp.spv", glm::uvec3(8, 8, 8), true);
        volumeSimulator->addKernel("clearBuffers", "src/shaders/compiled/clearbuffers.comp.spv", glm::uvec3(64, 1, 1), true);
        volumeSimulator->addKernel("clearPressure", "src/shaders/compiled/clearpressure.comp.spv", glm::uvec3(64, 1, 1), true);
        volumeSimulator->addKernel("clearPressureOut", "src/shaders/compiled/clearpressureout.comp.spv", glm::uvec3(64, 1, 1), true);
        volumeSimulator->addKernel("copyVelocityTemp", "src/shaders/compiled/copyvelocitytemp.comp.spv", glm::uvec3(64, 1, 1), true);
        volumeSimulator->addKernel("resetFanAccess", "src/shaders/compiled/resetfanaccess.comp.spv", glm::uvec3(64, 1, 1), true);
        volumeSimulator->addKernel("tempReader", "src/shaders/compiled/tempreader.comp.spv", glm::uvec3(1, 1, 1), true);
        currentPushConstants.gridSize = glm::vec4(gridSizeX, gridSizeY, gridSizeZ, 1.0f);
        currentPushConstants.worldMin = glm::vec4(worldMinX, worldMinY, worldMinZ, 1.0f);
        currentPushConstants.worldMax = glm::vec4(worldMaxX, worldMaxY, worldMaxZ, 1.0f);
        currentPushConstants.cellSize = glm::vec4(cellSizeX, cellSizeY, cellSizeZ, 1.0f);
        currentPushConstants.displayPressure = pressureEnabled;
        currentPushConstants.deltaTime = 1.0f / 60.0f;
        createVolumeGeometry();
        createVolumeUniformBuffers();
        createVolumeDescriptorSetLayout();
        createVolumeDescriptorPool();
        createVolumePipeline();
        volumeData.volumeImageView = volumeSimulator->getVolumeImageView();
        volumeData.temperatureImageView = volumeSimulator->getTemperatureImageView();
        createVolumeDescriptorSets();
        createCommandBuffers();
        createSyncObjects();
    }
    void mainLoop(){
        while(!glfwWindowShouldClose(window)){
            glfwPollEvents();
            processInput(window);
            drawFrame();
        }
        vkDeviceWaitIdle(device);
    }
    void cleanup(){
        vkDeviceWaitIdle(device);
        if(volumeSimulator){
            volumeSimulator->cleanup();
            delete volumeSimulator;
            volumeSimulator = nullptr;
        }
        cleanupSwapChain();
        if(textVertexBuffer) vkUnmapMemory(device, textVertexBufferMemory);
        vkDestroyBuffer(device, textVertexBuffer, nullptr);
        vkFreeMemory(device, textVertexBufferMemory, nullptr);
        vkDestroyBuffer(device, textIndexBuffer, nullptr);
        vkFreeMemory(device, textIndexBufferMemory, nullptr);
        vkDestroyPipeline(device, textPipeline, nullptr);
        vkDestroyPipelineLayout(device, textPipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, textDescriptorSetLayout, nullptr);
        vkDestroyDescriptorPool(device, textDescriptorPool, nullptr);
        vkDestroySampler(device, textureSampler, nullptr);
        vkDestroySampler(device, volumeSampler, nullptr);
        vkDestroyDescriptorPool(device, descriptorPool, nullptr);
        for(auto &model : models){
            vkDestroyBuffer(device, model.second.vertexBuffer, nullptr);
            vkFreeMemory(device, model.second.vertexBufferMemory, nullptr);
            vkDestroyBuffer(device, model.second.indexBuffer, nullptr);
            vkFreeMemory(device, model.second.indexBufferMemory, nullptr);
            for(size_t i=0; i<4; i++){
                vkDestroyImageView(device, model.second.textureImageView[i], nullptr);
                vkDestroyImage(device, model.second.textureImage[i], nullptr);
                vkFreeMemory(device, model.second.textureImageMemory[i], nullptr);
                vkDestroySampler(device, model.second.textureSampler[i], nullptr);
            }
            for(size_t i=0; i<MAX_FRAMES_IN_FLIGHT; i++){
                vkDestroyBuffer(device, model.second.uniformBuffers[i], nullptr);
                vkFreeMemory(device, model.second.uniformBuffersMemory[i], nullptr);
            }
        }
        for(size_t i=0; i<MAX_FRAMES_IN_FLIGHT; i++){
            vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
            vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
            vkDestroyFence(device, inFlightFences[i], nullptr);
            vkDestroyBuffer(device, volumeData.uniformBuffers[i], nullptr);
            vkFreeMemory(device, volumeData.uniformBuffersMemory[i], nullptr);
        }
        for(auto &character : Characters){
            vkDestroyImageView(device, character.second.textureImageView, nullptr);
            vkDestroyImage(device, character.second.textureImage, nullptr);
            vkFreeMemory(device, character.second.textureImageMemory, nullptr);
        }
        for(auto &uiObject : uiObjects){
            if(uiObject.second.vertexBufferMapped) vkUnmapMemory(device, uiObject.second.vertexBufferMemory);
            vkDestroyBuffer(device, uiObject.second.vertexBuffer, nullptr);
            vkFreeMemory(device, uiObject.second.vertexBufferMemory, nullptr);
        }
        for(auto &texture : textureAtlas){
            vkDestroyImageView(device, texture.second.imageView, nullptr);
            vkDestroyImage(device, texture.second.image, nullptr);
            vkFreeMemory(device, texture.second.imageMemory, nullptr);
        }
        vkDestroyBuffer(device, volumeData.vertexBuffer, nullptr);
        vkFreeMemory(device, volumeData.vertexBufferMemory, nullptr);
        vkDestroyBuffer(device, volumeData.indexBuffer, nullptr);
        vkFreeMemory(device, volumeData.indexBufferMemory, nullptr);
        vkDestroyPipeline(device, volumePipeline, nullptr);
        vkDestroyPipelineLayout(device, volumePipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, volumeDescriptorSetLayout, nullptr);
        vkDestroyDescriptorPool(device, volumeDescriptorPool, nullptr);
        vkDestroyDescriptorSetLayout(device, uiDescriptorSetLayout, nullptr);
        vkDestroyDescriptorPool(device, uiDescriptorPool, nullptr);
        vkDestroyBuffer(device, uiIndexBuffer, nullptr);
        vkFreeMemory(device, uiIndexBufferMemory, nullptr);
        vkDestroyPipeline(device, graphicsPipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyRenderPass(device, renderPass, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, computeDescriptorSetLayout, nullptr);
        vkDestroyDescriptorPool(device, computeDescriptorPool, nullptr);
        vkDestroyCommandPool(device, commandPool, nullptr);
        vkDestroyDevice(device, nullptr);
        if(enableValidationLayers) DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
        vkDestroySurfaceKHR(instance, surface, nullptr);
        vkDestroyInstance(instance, nullptr);
        glfwDestroyWindow(window);
        glfwTerminate();
    }
    void createInstance(){
        if (enableValidationLayers && !checkValidationLayerSupport())
            throw std::runtime_error("Validation layers requested, but not available!");
        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "GustGrid";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_0;

        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;
        #ifdef __APPLE__
            createInfo.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
        #endif

        std::vector<const char*> extensions = getRequiredExtensions();
        createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        createInfo.ppEnabledExtensionNames = extensions.data();

        VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
        if(enableValidationLayers){
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
            populateDebugMessengerCreateInfo(debugCreateInfo);
            createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*) &debugCreateInfo;
        } else{
            createInfo.enabledLayerCount = 0;
            createInfo.pNext = nullptr;
        }

        if(vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS)
            throw std::runtime_error("Failed to create instance!");
    }
    void createSurface(){
        if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS)
            throw std::runtime_error("Failed to create window surface!");
    }
    void drawFrame(){
        dt = glfwGetTime() - currentTime;
        currentTime = glfwGetTime();
        float fps = 1.0f / dt;
        textObjects[0].text = std::to_string((int) fps) + " FPS";
        int backFansEnabled = 0;
        for(int i = 0; i < 3; i++) if(backFanLocations[i] <= 0.0f) backFansEnabled++;
        textObjects[9].text = "Back Fans: " + std::to_string(backFansEnabled);
        textObjects[10].enabled = backFansEnabled > 0 && showUI;
        textObjects[11].enabled = backFansEnabled > 1 && showUI;
        textObjects[12].enabled = backFansEnabled > 2 && showUI;
        uiObjects["sliderknob1"].position.x = 280.0f - (200.0f * (backFanLocations[0] / -5.0f));
        uiObjects["sliderknob2"].position.x = 280.0f - (200.0f * (backFanLocations[1] / -5.0f));
        uiObjects["sliderknob3"].position.x = 280.0f - (200.0f * (backFanLocations[2] / -5.0f));
        uiObjects["sliderknob1"].enabled = backFansEnabled > 0 && showUI;
        uiObjects["sliderknob2"].enabled = backFansEnabled > 1 && showUI;
        uiObjects["sliderknob3"].enabled = backFansEnabled > 2 && showUI;
        uiObjects["slider1"].enabled = backFansEnabled > 0 && showUI;
        uiObjects["slider2"].enabled = backFansEnabled > 1 && showUI;
        uiObjects["slider3"].enabled = backFansEnabled > 2 && showUI;
        vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);
        uint32_t imageIndex;
        VkResult result = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);
        if(result == VK_ERROR_OUT_OF_DATE_KHR){
            recreateSwapChain();
            return;
        } else if(result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
            throw std::runtime_error("Failed to acquire swap chain image!");
        vkResetFences(device, 1, &inFlightFences[currentFrame]);
        vkResetCommandBuffer(commandBuffers[currentFrame], 0);
        recordCommandBuffer(commandBuffers[currentFrame], imageIndex);
        updateUniformBuffer(currentFrame);
        if(shouldUpdateGrid){
            resetSolidGrid();
            shouldUpdateGrid = false;
        }
        currentPushConstants.displayPressure = pressureEnabled;
        volumeSimulator->setCurrentFrame(currentFrame);
        uint32_t totalCells = gridSizeX * gridSizeY * gridSizeZ;
        uint32_t fanElements = totalCells * maxFans;
        VolumeSimulator::ComputePushConstants dummyConstants = {};
        dummyConstants.gridSize = glm::vec4(gridSizeX, gridSizeY, gridSizeZ, 1.0f);
        dummyConstants.worldMin = glm::vec4(worldMinX, worldMinY, worldMinZ, 1.0f);
        dummyConstants.worldMax = glm::vec4(worldMaxX, worldMaxY, worldMaxZ, 1.0f);
        dummyConstants.cellSize = glm::vec4(cellSizeX, cellSizeY, cellSizeZ, 1.0f);
        dummyConstants.numFans = 0;
        if(shouldUpdateFans){
            updateVolumeTextures();
            VkSemaphore resetFanAccessSemaphore = volumeSimulator->dispatchKernel("resetFanAccess", glm::uvec3(fanElements, 1, 1), dummyConstants);
            VkSemaphore computeSemaphore = volumeSimulator->dispatchKernel("fanUpdate", glm::uvec3(gridSizeX, gridSizeY, gridSizeZ), currentPushConstants);
            shouldUpdateFans = false;
        }
        VkSemaphore clearTempBuffers1 = volumeSimulator->dispatchKernel("clearBuffers", glm::uvec3(totalCells, 1, 1), dummyConstants);
        VkSemaphore velocitySemaphore = volumeSimulator->dispatchKernel("velocityUpdate", glm::uvec3(gridSizeX, gridSizeY, gridSizeZ), currentPushConstants);
        VkSemaphore copyTempSemaphore = volumeSimulator->dispatchKernel("copyVelocityTemp", glm::uvec3(totalCells, 1, 1), dummyConstants);
        VkSemaphore divergenceSemaphore = volumeSimulator->dispatchKernel("computeDivergence", glm::uvec3(gridSizeX, gridSizeY, gridSizeZ), currentPushConstants);
        VkSemaphore clearPressureTemp = volumeSimulator->dispatchKernel("clearPressure", glm::uvec3(totalCells, 1, 1), dummyConstants);
        for(int iter = 0; iter < maxPressureIterations; iter++){
            VkSemaphore jacobiSemaphore = volumeSimulator->dispatchKernel("pressureJacobi", glm::uvec3(gridSizeX, gridSizeY, gridSizeZ), currentPushConstants);
            if(iter % 4 == 3 || iter == maxPressureIterations - 1){
                VkSemaphore residualSemaphore = volumeSimulator->dispatchKernel("computeResidual", glm::uvec3(gridSizeX, gridSizeY, gridSizeZ), currentPushConstants);
                float residualSum = volumeSimulator->computeResidualSum();
                uint32_t totalCells = gridSizeX * gridSizeY * gridSizeZ;
                float avgResidual = residualSum / totalCells;
                if(avgResidual < pressureTolerance){
                    volumeSimulator->swapPressureBuffers();
                    break;
                }
            }
            volumeSimulator->swapPressureBuffers();
        }
        volumeSimulator->copyFinalPressureToMain();
        VkSemaphore gradientSemaphore = volumeSimulator->dispatchKernel("subtractPressureGradient", glm::uvec3(gridSizeX, gridSizeY, gridSizeZ), currentPushConstants);
        VkSemaphore boundarySemaphore = volumeSimulator->dispatchKernel("enforceBoundaryConditions", glm::uvec3(gridSizeX, gridSizeY, gridSizeZ), currentPushConstants);
        VkSemaphore advectSemaphore = volumeSimulator->dispatchKernel("computeAdvection", glm::uvec3(gridSizeX, gridSizeY, gridSizeZ), currentPushConstants);
        VkSemaphore normalizeKernel = volumeSimulator->dispatchKernel("normalizeForward", glm::uvec3(gridSizeX, gridSizeY, gridSizeZ), currentPushConstants);
        VkSemaphore convectiveKernel = volumeSimulator->dispatchKernel("convectiveHeat", glm::uvec3(gridSizeX, gridSizeY, gridSizeZ), currentPushConstants);
        VkSemaphore diffuseKernel = volumeSimulator->dispatchKernel("diffuseKernel", glm::uvec3(gridSizeX, gridSizeY, gridSizeZ), currentPushConstants);
        volumeSimulator->updateVolumeImages(currentPushConstants.displayPressure);
        VolumeSimulator::ComputePushConstants probePC = currentPushConstants;
        probePC.padding = 0;
        volumeSimulator->dispatchKernel("tempReader", glm::uvec3(1, 1, 1), probePC);
        probePC.padding = 1;
        volumeSimulator->dispatchKernel("tempReader", glm::uvec3(1, 1, 1), probePC);
        updateVolumeDescriptorSets();
        vkQueueWaitIdle(graphicsQueue);
        float* probeOut = volumeSimulator->getProbeOut();
        char cpuText[32];
        snprintf(cpuText, sizeof(cpuText), "CPU Temperature: %0.2f°C", probeOut[0]);
        textObjects[14].text = cpuText;
        if(gpuEnabled) {
            char gpuText[32];
            snprintf(gpuText, sizeof(gpuText), "GPU Temperature: %0.2f°C", probeOut[1]);
            textObjects[15].text = gpuText;
        } else textObjects[15].enabled = false;
        std::vector<VkSemaphore> waitSemaphores;
        std::vector<VkPipelineStageFlags> waitStages;
        waitSemaphores.push_back(imageAvailableSemaphores[currentFrame]);
        waitStages.push_back(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.waitSemaphoreCount = static_cast<uint32_t>(waitSemaphores.size());
        submitInfo.pWaitSemaphores = waitSemaphores.data();
        submitInfo.pWaitDstStageMask = waitStages.data();
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffers[currentFrame];
        VkSemaphore signalSemaphores[] = {renderFinishedSemaphores[currentFrame]};
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSemaphores;
        if(vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS)
            throw std::runtime_error("Failed to submit draw command buffer!");
        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = signalSemaphores;
        VkSwapchainKHR swapChains[] = {swapChain};
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = swapChains;
        presentInfo.pImageIndices = &imageIndex;
        presentInfo.pResults = nullptr;
        result = vkQueuePresentKHR(presentQueue, &presentInfo);
        if(result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized){
            framebufferResized = false;
            recreateSwapChain();
        } else if(result != VK_SUCCESS)
            throw std::runtime_error("Failed to present swap chain image!");
        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }
    void updateUniformBuffer(uint32_t currentImage){
        glm::mat4 view = glm::lookAt(camPos, glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        glm::mat4 proj = glm::perspective(glm::radians(camFOV), (float) swapChainExtent.width / (float) swapChainExtent.height, 0.1f, 100.0f);
        proj[1][1] *= -1;
        for(auto &model : models){
            if(!model.second.loaded) continue;
            UniformBufferObject ubo{};
            ubo.model = model.second.modelMatrix;
            ubo.view = view;
            ubo.proj = proj;
            ubo.cameraPos = camPos;
            memcpy(model.second.uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
        }
        VolumeUniformBufferObject vubo{};
        vubo.worldMin = glm::vec3(worldMinX, worldMinY, worldMinZ);
        vubo.worldMax = glm::vec3(worldMaxX, worldMaxY, worldMaxZ);
        vubo.model = glm::mat4(1.0f);
        vubo.view = view;
        vubo.projection = proj;
        vubo.camPos = camPos;
        vubo.gridSize = glm::vec3(float(gridSizeX), float(gridSizeY), float(gridSizeZ));
        vubo.displayPressure = pressureEnabled ? 1 : 0;
        vubo.stepSize = 0.1f;
        memcpy(volumeData.uniformBuffersMapped[currentImage], &vubo, sizeof(vubo));
    }
    void createSyncObjects(){
        imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
            if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS
            ||  vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS
            ||  vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS)
                throw std::runtime_error("Failed to create sync objects!");
    }
    void createLogicalDevice(){
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        std::set<uint32_t> uniqueQueueFamilies = {indices.graphicsFamily.value(), indices.presentFamily.value()};
        float queuePriority = 1.0f;
        for(uint32_t queueFamily : uniqueQueueFamilies){
            VkDeviceQueueCreateInfo queueCreateInfo{};
            queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueCreateInfo.queueFamilyIndex = queueFamily;
            queueCreateInfo.queueCount = 1;
            queueCreateInfo.pQueuePriorities = &queuePriority;
            queueCreateInfos.push_back(queueCreateInfo);
        }
        VkPhysicalDeviceFeatures deviceFeatures{};
        deviceFeatures.samplerAnisotropy = VK_TRUE;
        deviceFeatures.sampleRateShading = VK_TRUE;
        VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        createInfo.pQueueCreateInfos = queueCreateInfos.data();
        createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
        createInfo.pEnabledFeatures = &deviceFeatures;
        createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
        createInfo.ppEnabledExtensionNames = deviceExtensions.data();
        if(enableValidationLayers){
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
        } else createInfo.enabledLayerCount = 0;
        if(vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS)
            throw std::runtime_error("Failed to create logical device!");
        vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
        vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
    }
    void createImageViews(){
        swapChainImageViews.resize(swapChainImages.size());
        for(size_t i=0; i<swapChainImages.size(); i++){
            swapChainImageViews[i] = createImageView(swapChainImages[i], swapChainImageFormat, 1);
        }
    }
    void recreateSwapChain(){
        int width = 0, height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        while(width == 0 || height == 0){
            glfwGetFramebufferSize(window, &width, &height);
            glfwWaitEvents();
        }
        vkDeviceWaitIdle(device);
        cleanupSwapChain();
        createSwapChain();
        createImageViews();
        createColorResources();
        createDepthResources();
        createFramebuffers();
    }
    void cleanupSwapChain(){
        vkDestroyImageView(device, depthImageView, nullptr);
        vkDestroyImage(device, depthImage, nullptr);
        vkFreeMemory(device, depthImageMemory, nullptr);
        vkDestroyImageView(device, colorImageView, nullptr);
        vkDestroyImage(device, colorImage, nullptr);
        vkFreeMemory(device, colorImageMemory, nullptr);
        for(size_t i=0; i<swapChainFramebuffers.size(); i++)
            vkDestroyFramebuffer(device, swapChainFramebuffers[i], nullptr);
        for(size_t i=0; i<swapChainImageViews.size(); i++)
            vkDestroyImageView(device, swapChainImageViews[i], nullptr);
        vkDestroySwapchainKHR(device, swapChain, nullptr);
    }
    void createSwapChain(){
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);
        VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
        VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
        VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);
        uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
        if(swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount)
            imageCount = swapChainSupport.capabilities.maxImageCount;
        VkSwapchainCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface = surface;
        createInfo.minImageCount = imageCount;
        createInfo.imageFormat = surfaceFormat.format;
        createInfo.imageColorSpace = surfaceFormat.colorSpace;
        createInfo.imageExtent = extent;
        createInfo.imageArrayLayers = 1;
        createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
        uint32_t queueFamilyIndices[] = {indices.graphicsFamily.value(), indices.presentFamily.value()};
        if(indices.graphicsFamily != indices.presentFamily){
            createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices = queueFamilyIndices;
        } else createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        createInfo.presentMode = presentMode;
        createInfo.clipped = VK_TRUE;
        createInfo.oldSwapchain = VK_NULL_HANDLE;
        if(vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS)
            throw std::runtime_error("Failed to create swap chain!");
        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
        swapChainImages.resize(imageCount);
        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());
        swapChainImageFormat = surfaceFormat.format;
        swapChainExtent = extent;
    }
    void createVolumeDescriptorPool(){
        VkDescriptorPoolSize uboPoolSize{};
        uboPoolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        uboPoolSize.descriptorCount = MAX_FRAMES_IN_FLIGHT;
        VkDescriptorPoolSize volumePoolSize{};
        volumePoolSize.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        volumePoolSize.descriptorCount = MAX_FRAMES_IN_FLIGHT;
        VkDescriptorPoolSize temperaturePoolSize{};
        temperaturePoolSize.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        temperaturePoolSize.descriptorCount = MAX_FRAMES_IN_FLIGHT;
        std::array<VkDescriptorPoolSize, 3> poolSizes = {uboPoolSize, volumePoolSize, temperaturePoolSize};
        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        poolInfo.maxSets = MAX_FRAMES_IN_FLIGHT;
        if(vkCreateDescriptorPool(device, &poolInfo, nullptr, &volumeDescriptorPool) != VK_SUCCESS)
            throw std::runtime_error("Failed to create descriptor pool for volume data!");
    }
    void createVolumeDescriptorSets(){
        std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, volumeDescriptorSetLayout);
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = volumeDescriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
        allocInfo.pSetLayouts = layouts.data();
        volumeData.descriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
        if(vkAllocateDescriptorSets(device, &allocInfo, volumeData.descriptorSets.data()) != VK_SUCCESS)
            throw std::runtime_error("Failed to allocate descriptor sets for volume data!");
        for(size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++){
            VkDescriptorBufferInfo uboBufferInfo{};
            uboBufferInfo.buffer = volumeData.uniformBuffers[i];
            uboBufferInfo.offset = 0;
            uboBufferInfo.range = sizeof(VolumeUniformBufferObject);
            std::array<VkWriteDescriptorSet, 3> descriptorWrites{};
            descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[0].dstSet = volumeData.descriptorSets[i];
            descriptorWrites[0].dstBinding = 0;
            descriptorWrites[0].dstArrayElement = 0;
            descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites[0].descriptorCount = 1;
            descriptorWrites[0].pBufferInfo = &uboBufferInfo;
            vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
        }
    }
    void createVolumeUniformBuffers(){
        VkDeviceSize bufferSize = sizeof(VolumeUniformBufferObject);
        volumeData.uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        volumeData.uniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
        volumeData.uniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);
        for(size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++){
            createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, volumeData.uniformBuffers[i], volumeData.uniformBuffersMemory[i]);
            vkMapMemory(device, volumeData.uniformBuffersMemory[i], 0, bufferSize, 0, &volumeData.uniformBuffersMapped[i]);
        }
    }
    void createVolumeDescriptorSetLayout(){
        VkDescriptorSetLayoutBinding uboLayoutBinding{};
        uboLayoutBinding.binding = 0;
        uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        uboLayoutBinding.descriptorCount = 1;
        uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        uboLayoutBinding.pImmutableSamplers = nullptr;

        VkDescriptorSetLayoutBinding volumeLayoutBinding{};
        volumeLayoutBinding.binding = 1;
        volumeLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        volumeLayoutBinding.descriptorCount = 1;
        volumeLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        volumeLayoutBinding.pImmutableSamplers = nullptr;

        VkDescriptorSetLayoutBinding temperatureLayoutBinding{};
        temperatureLayoutBinding.binding = 2;
        temperatureLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        temperatureLayoutBinding.descriptorCount = 1;
        temperatureLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        temperatureLayoutBinding.pImmutableSamplers = nullptr;

        std::array<VkDescriptorSetLayoutBinding, 3> bindings = {uboLayoutBinding, volumeLayoutBinding, temperatureLayoutBinding};

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();

        if(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &volumeDescriptorSetLayout) != VK_SUCCESS)
            throw std::runtime_error("Failed to create descriptor set layout!");
    }
    void createDescriptorSetLayout(){
        VkDescriptorSetLayoutBinding uboLayoutBinding{};
        uboLayoutBinding.binding = 0;
        uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        uboLayoutBinding.descriptorCount = 1;
        uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        uboLayoutBinding.pImmutableSamplers = nullptr;

        VkDescriptorSetLayoutBinding baseColorLayoutBinding{};
        baseColorLayoutBinding.binding = 1;
        baseColorLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        baseColorLayoutBinding.descriptorCount = 1;
        baseColorLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        baseColorLayoutBinding.pImmutableSamplers = nullptr;

        VkDescriptorSetLayoutBinding metallicLayoutBinding{};
        metallicLayoutBinding.binding = 2;
        metallicLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        metallicLayoutBinding.descriptorCount = 1;
        metallicLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        metallicLayoutBinding.pImmutableSamplers = nullptr;

        VkDescriptorSetLayoutBinding roughnessLayoutBinding{};
        roughnessLayoutBinding.binding = 3;
        roughnessLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        roughnessLayoutBinding.descriptorCount = 1;
        roughnessLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        roughnessLayoutBinding.pImmutableSamplers = nullptr;

        VkDescriptorSetLayoutBinding normalMapLayoutBinding{};
        normalMapLayoutBinding.binding = 4;
        normalMapLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        normalMapLayoutBinding.descriptorCount = 1;
        normalMapLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        normalMapLayoutBinding.pImmutableSamplers = nullptr;

        std::array<VkDescriptorSetLayoutBinding, 5> bindings = {uboLayoutBinding, baseColorLayoutBinding, metallicLayoutBinding, roughnessLayoutBinding, normalMapLayoutBinding};

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();

        if(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS)
            throw std::runtime_error("Failed to create descriptor set layout!");
    }
    void createGraphicsPipeline(){
        std::vector<char> vertShaderCode = readFile("src/shaders/compiled/main.vert.spv");
        std::vector<char> fragShaderCode = readFile("src/shaders/compiled/main.frag.spv");
        VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
        VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);
        VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
        vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vertShaderStageInfo.module = vertShaderModule;
        vertShaderStageInfo.pName = "main";
        VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
        fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragShaderStageInfo.module = fragShaderModule;
        fragShaderStageInfo.pName = "main";
        VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};
        std::vector<VkDynamicState> dynamicStates = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR
        };
        VkPipelineDynamicStateCreateInfo dynamicState{};
        dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
        dynamicState.pDynamicStates = dynamicStates.data();
        auto bindingDescription = Vertex::getBindingDescription();
        auto attributeDescriptions = Vertex::getAttributeDescriptions();
        VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
        vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();
        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;
        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = static_cast<float>(swapChainExtent.width);
        viewport.height = static_cast<float>(swapChainExtent.height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        VkRect2D scissor{};
        scissor.offset = {0, 0};
        scissor.extent = swapChainExtent;
        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.scissorCount = 1;
        viewportState.pViewports = &viewport;
        viewportState.pScissors = &scissor;
        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.depthClampEnable = VK_FALSE;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = VK_CULL_MODE_NONE;
        rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterizer.depthBiasEnable = VK_FALSE;
        rasterizer.depthBiasConstantFactor = 0.0f;
        rasterizer.depthBiasClamp = 0.0f;
        rasterizer.depthBiasSlopeFactor = 0.0f;
        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_TRUE;
        multisampling.rasterizationSamples = msaaSamples;
        multisampling.minSampleShading = 0.2f;
        multisampling.pSampleMask = nullptr;
        multisampling.alphaToCoverageEnable = VK_FALSE;
        multisampling.alphaToOneEnable = VK_FALSE;
        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.blendEnable = VK_TRUE;
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
        colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;
        VkPipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.logicOp = VK_LOGIC_OP_COPY;
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;
        colorBlending.blendConstants[0] = 0.0f;
        colorBlending.blendConstants[1] = 0.0f;
        colorBlending.blendConstants[2] = 0.0f;
        colorBlending.blendConstants[3] = 0.0f;
        VkPipelineDepthStencilStateCreateInfo depthStencil{};
        depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depthStencil.depthTestEnable = VK_TRUE;
        depthStencil.depthWriteEnable = VK_TRUE;
        depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
        depthStencil.depthBoundsTestEnable = VK_FALSE;
        depthStencil.minDepthBounds = 0.0f;
        depthStencil.maxDepthBounds = 1.0f;
        depthStencil.stencilTestEnable = VK_FALSE;
        depthStencil.front = {};
        depthStencil.back = {};
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
        pipelineLayoutInfo.pushConstantRangeCount = 0;
        pipelineLayoutInfo.pPushConstantRanges = nullptr;
        if(vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS)
            throw std::runtime_error("Failed to create pipeline layout!");
        VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = shaderStages;
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.pDynamicState = &dynamicState;
        pipelineInfo.pDepthStencilState = &depthStencil;
        pipelineInfo.layout = pipelineLayout;
        pipelineInfo.renderPass = renderPass;
        pipelineInfo.subpass = 0;
        pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
        pipelineInfo.basePipelineIndex = -1;
        if(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS)
            throw std::runtime_error("Failed to create graphics pipeline!");
        
        vkDestroyShaderModule(device, fragShaderModule, nullptr);
        vkDestroyShaderModule(device, vertShaderModule, nullptr);
    }
    void createTextPipeline(){
        std::vector<char> vertShaderCode = readFile("src/shaders/compiled/ui.vert.spv");
        std::vector<char> fragShaderCode = readFile("src/shaders/compiled/ui.frag.spv");
        VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
        VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);
        VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
        vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vertShaderStageInfo.module = vertShaderModule;
        vertShaderStageInfo.pName = "main";
        VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
        fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragShaderStageInfo.module = fragShaderModule;
        fragShaderStageInfo.pName = "main";
        VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};
        auto bindingDescription = TextVertex::getBindingDescription();
        auto attributeDescriptions = TextVertex::getAttributeDescriptions();
        VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
        vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();
        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;
        std::vector<VkDynamicState> dynamicStates = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR
        };
        VkPipelineDynamicStateCreateInfo dynamicState{};
        dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
        dynamicState.pDynamicStates = dynamicStates.data();
        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.scissorCount = 1;
        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = VK_CULL_MODE_NONE;
        rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.rasterizationSamples = msaaSamples;
        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable = VK_TRUE;
        colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
        colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;
        VkPipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;
        VkPipelineDepthStencilStateCreateInfo depthStencil{};
        depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        VkPushConstantRange pushConstantRange{};
        pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        pushConstantRange.offset = 0;
        pushConstantRange.size = sizeof(glm::vec3) + sizeof(uint32_t);
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &textDescriptorSetLayout;
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
        if(vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &textPipelineLayout) != VK_SUCCESS)
            throw std::runtime_error("Failed to create text pipeline layout!");
        VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = shaderStages;
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.pDynamicState = &dynamicState;
        pipelineInfo.pDepthStencilState = &depthStencil;
        pipelineInfo.layout = textPipelineLayout;
        pipelineInfo.renderPass = renderPass;
        pipelineInfo.subpass = 0;
        if(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &textPipeline) != VK_SUCCESS)
            throw std::runtime_error("Failed to create text graphics pipeline!");
        vkDestroyShaderModule(device, fragShaderModule, nullptr);
        vkDestroyShaderModule(device, vertShaderModule, nullptr);
    }
    void createVolumePipeline(){
        std::vector<char> vertShaderCode = readFile("src/shaders/compiled/volume.vert.spv");
        std::vector<char> fragShaderCode = readFile("src/shaders/compiled/volume.frag.spv");
        VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
        VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);
        VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
        vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vertShaderStageInfo.module = vertShaderModule;
        vertShaderStageInfo.pName = "main";
        VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
        fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragShaderStageInfo.module = fragShaderModule;
        fragShaderStageInfo.pName = "main";
        VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};
        auto bindingDescription = Vertex::getBindingDescription();
        auto attributeDescriptions = Vertex::getAttributeDescriptions();
        VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
        vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();
        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;
        std::vector<VkDynamicState> dynamicStates = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR
        };
        VkPipelineDynamicStateCreateInfo dynamicState{};
        dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
        dynamicState.pDynamicStates = dynamicStates.data();
        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.scissorCount = 1;
        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = VK_CULL_MODE_NONE;
        rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterizer.depthClampEnable = VK_FALSE;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.rasterizationSamples = msaaSamples;
        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable = VK_TRUE;
        colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
        colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;
        VkPipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;
        VkPipelineDepthStencilStateCreateInfo depthStencil{};
        depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depthStencil.depthTestEnable = VK_TRUE;
        depthStencil.depthWriteEnable = VK_FALSE;
        depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &volumeDescriptorSetLayout;
        if(vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &volumePipelineLayout) != VK_SUCCESS)
            throw std::runtime_error("Failed to create volume pipeline layout!");
        VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = shaderStages;
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.pDynamicState = &dynamicState;
        pipelineInfo.pDepthStencilState = &depthStencil;
        pipelineInfo.layout = volumePipelineLayout;
        pipelineInfo.renderPass = renderPass;
        pipelineInfo.subpass = 0;
        if(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &volumePipeline) != VK_SUCCESS)
            throw std::runtime_error("Failed to create volume graphics pipeline!");
        vkDestroyShaderModule(device, fragShaderModule, nullptr);
        vkDestroyShaderModule(device, vertShaderModule, nullptr);
    }
    void createColorResources(){
        VkFormat colorFormat = swapChainImageFormat;
        createImage(swapChainExtent.width, swapChainExtent.height, 1, msaaSamples, colorFormat, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, colorImage, colorImageMemory);
        colorImageView = createImageView(colorImage, colorFormat, 1, VK_IMAGE_ASPECT_COLOR_BIT);
    }
    void createDepthResources(){
        VkFormat depthFormat = findDepthFormat();
        createImage(swapChainExtent.width, swapChainExtent.height, 1, msaaSamples, depthFormat, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, depthImage, depthImageMemory);
        depthImageView = createImageView(depthImage, depthFormat, 1, VK_IMAGE_ASPECT_DEPTH_BIT);
        transitionImageLayout(depthImage, depthFormat, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    }
    VkFormat findSupportedFormat(const std::vector<VkFormat> &candidates, VkImageTiling tiling, VkFormatFeatureFlags features){
        for(VkFormat format : candidates){
            VkFormatProperties props;
            vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);
            if(tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features)
                return format;
            else if(tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features)
                return format;
        }
        throw std::runtime_error("Failed to find supported format!");
    }
    VkFormat findDepthFormat(){
        return findSupportedFormat({
                VK_FORMAT_D32_SFLOAT,
                VK_FORMAT_D32_SFLOAT_S8_UINT,
                VK_FORMAT_D24_UNORM_S8_UINT
            }, VK_IMAGE_TILING_OPTIMAL, 
            VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
        );
    }
    bool hasStencilComponent(VkFormat format){
        return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
    }
    void createTextureImage(int width, int height, unsigned char* imageBuffer, VkImage &textureImage, VkDeviceMemory &textureImageMemory, VkFormat format){
        if(width == 0 || height == 0) {
            width = std::max(width, 1);
            height = std::max(height, 1);
        }
        VkDeviceSize imageSize = width * height;
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);
        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, imageSize, 0, &data);
        memcpy(data, imageBuffer, (size_t) imageSize);
        vkUnmapMemory(device, stagingBufferMemory);
        createImage(width, height, 1, VK_SAMPLE_COUNT_1_BIT, format, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, textureImage, textureImageMemory);
        transitionImageLayout(textureImage, format, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        copyBufferToImage(stagingBuffer, textureImage, static_cast<uint32_t>(width), static_cast<uint32_t>(height));
        transitionImageLayout(textureImage, format, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }
    void createTextureImage(){
        stbi_set_flip_vertically_on_load(true);
        VkFormat textureFormats[] = {VK_FORMAT_R8G8B8A8_SRGB, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM};
        std::string pathNames[] = {"/basecolor.png", "/metallic.png", "/roughness.png", "/normal.png"};
        for(auto &model : models){
            for(size_t i=0; i<4; i++){
                int texWidth, texHeight, texChannels;
                std::string texturePath;
                if(model.first == "cpufan" || model.first == "frontfan" || model.first == "topfan" || model.first == "fancpu" || model.first == "fanfront" || model.first == "fantop" || model.first == "backfan1" || model.first == "backfan2" || model.first == "backfan3" || model.first == "fanback1" || model.first == "fanback2" || model.first == "fanback3" || model.first == "gpufan1" || model.first == "gpufan2"){
                    texturePath = "src/textures/case" + pathNames[i];
                    if(model.first == "cpufan" || model.first == "frontfan" || model.first == "topfan") model.second.name = model.first;
                    else if(model.first == "backfan1" || model.first == "backfan2" || model.first == "backfan3") model.second.name = "backfan";
                    else if(model.first == "gpufan1" || model.first == "gpufan2") model.second.name = "gpufan";
                    else model.second.name = "fan";
                }
                else{
                    model.second.name = model.first;
                    texturePath = "src/textures/" + model.first + pathNames[i];
                }
                stbi_uc* pixels = stbi_load(texturePath.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
                VkDeviceSize imageSize = texWidth * texHeight * 4;
                if(!pixels) throw std::runtime_error("Failed to load texture image!");
                model.second.mipmapLevels[i] = static_cast<uint32_t>(std::floor(std::log2(std::max(texWidth, texHeight)))) + 1;
                VkBuffer stagingBuffer;
                VkDeviceMemory stagingBufferMemory;
                createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);
                void* data;
                VkImage textureImage;
                VkDeviceMemory textureImageMemory;
                vkMapMemory(device, stagingBufferMemory, 0, imageSize, 0, &data);
                memcpy(data, pixels, (size_t) imageSize);
                vkUnmapMemory(device, stagingBufferMemory);
                stbi_image_free(pixels);
                createImage(texWidth, texHeight, model.second.mipmapLevels[i], VK_SAMPLE_COUNT_1_BIT, textureFormats[i], VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, textureImage, textureImageMemory);
                transitionImageLayout(textureImage, textureFormats[i], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, model.second.mipmapLevels[i]);
                copyBufferToImage(stagingBuffer, textureImage, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));
                transitionImageLayout(textureImage, textureFormats[i], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, model.second.mipmapLevels[i]);
                generateMipmaps(textureImage, texWidth, texHeight, model.second.mipmapLevels[i]);
                vkDestroyBuffer(device, stagingBuffer, nullptr);
                vkFreeMemory(device, stagingBufferMemory, nullptr);
                createTextureSampler(model.second.textureSampler[i], model.second.mipmapLevels[i]);
                model.second.textureImage[i] = textureImage;
                model.second.textureImageMemory[i] = textureImageMemory;
            }
            model.second.loaded = true;
        }
    }
    void createTextureImageView(VkFormat textureFormat, VkImage textureImage, VkImageView &textureImageView){
        textureImageView = createImageView(textureImage, textureFormat, 1);
    }
    void createTextureImageView(){
        VkFormat textureFormats[] = {VK_FORMAT_R8G8B8A8_SRGB, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM};
        for(auto &model : models) for(size_t i=0; i<4; i++) model.second.textureImageView[i] = createImageView(model.second.textureImage[i], textureFormats[i], model.second.mipmapLevels[i]);
    }
    VkImageView createImageView(VkImage image, VkFormat format, uint32_t mipLevels, VkImageAspectFlags aspectFlags = VK_IMAGE_ASPECT_COLOR_BIT){
        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = format;
        viewInfo.subresourceRange.aspectMask = aspectFlags;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = mipLevels;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;
        VkImageView imageView;
        if(vkCreateImageView(device, &viewInfo, nullptr, &imageView) != VK_SUCCESS)
            throw std::runtime_error("Failed to create image view!");
        return imageView;
    }
    void generateMipmaps(VkImage image, int32_t texWidth, uint32_t texHeight, uint32_t mipLevels){
        VkFormatProperties formatProperties;
        vkGetPhysicalDeviceFormatProperties(physicalDevice, VK_FORMAT_R8G8B8A8_SRGB, &formatProperties);
        if(!(formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT))
            throw std::runtime_error("Texture image format does not support linear blitting!");
        if(mipLevels < 2) return;
        VkCommandBuffer commandBuffer = beginSingleTimeCommands();
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.image = image;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;
        barrier.subresourceRange.levelCount = 1;
        int32_t mipWidth = texWidth;
        int32_t mipHeight = texHeight;
        for(uint32_t i=1; i<mipLevels; i++){
            barrier.subresourceRange.baseMipLevel = i - 1;
            barrier.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
            VkImageBlit blit{};
            blit.srcOffsets[0] = {0, 0, 0};
            blit.srcOffsets[1] = {mipWidth, mipHeight, 1};
            blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            blit.srcSubresource.mipLevel = i - 1;
            blit.srcSubresource.baseArrayLayer = 0;
            blit.srcSubresource.layerCount = 1;
            int32_t nextMipWidth = mipWidth > 1 ? mipWidth / 2 : 1;
            int32_t nextMipHeight = mipHeight > 1 ? mipHeight / 2 : 1;
            blit.dstOffsets[0] = {0, 0, 0};
            blit.dstOffsets[1] = {nextMipWidth, nextMipHeight, 1};
            blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            blit.dstSubresource.mipLevel = i;
            blit.dstSubresource.baseArrayLayer = 0;
            blit.dstSubresource.layerCount = 1;
            vkCmdBlitImage(commandBuffer, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);
            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
            mipWidth = nextMipWidth;
            mipHeight = nextMipHeight;
        }
        barrier.subresourceRange.baseMipLevel = mipLevels - 1;
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
        endSingleTimeCommands(commandBuffer);
    }
    void createTextureSampler(){
        VkSamplerCreateInfo samplerInfo{};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.anisotropyEnable = VK_TRUE;
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(physicalDevice, &properties);
        samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy;
        samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        samplerInfo.unnormalizedCoordinates = VK_FALSE;
        samplerInfo.compareEnable = VK_FALSE;
        samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        samplerInfo.mipLodBias = 0.0f;
        samplerInfo.minLod = 0.0f;
        samplerInfo.maxLod = VK_LOD_CLAMP_NONE;
        if(vkCreateSampler(device, &samplerInfo, nullptr, &textureSampler) != VK_SUCCESS)
            throw std::runtime_error("Failed to create texture sampler!");
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        samplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
        samplerInfo.maxLod = 1.0f;
        if(vkCreateSampler(device, &samplerInfo, nullptr, &volumeSampler) != VK_SUCCESS)
            throw std::runtime_error("Failed to create volume texture sampler!");
    }
    void createTextureSampler(VkSampler &sampler, uint32_t mipLevels = 1){
        VkSamplerCreateInfo samplerInfo{};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.anisotropyEnable = VK_TRUE;
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(physicalDevice, &properties);
        samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy;
        samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        samplerInfo.unnormalizedCoordinates = VK_FALSE;
        samplerInfo.compareEnable = VK_FALSE;
        samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        samplerInfo.mipLodBias = 0.0f;
        samplerInfo.minLod = 0.0f;
        samplerInfo.maxLod = static_cast<float>(mipLevels);
        if(vkCreateSampler(device, &samplerInfo, nullptr, &sampler) != VK_SUCCESS)
            throw std::runtime_error("Failed to create texture sampler!");
    }
    void createImage(uint32_t width, uint32_t height, uint32_t mipLevels, VkSampleCountFlagBits numSamples, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage &image, VkDeviceMemory &imageMemory){
        VkImageCreateInfo imageInfo{};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.extent.width = width;
        imageInfo.extent.height = height;
        imageInfo.extent.depth = 1;
        imageInfo.mipLevels = mipLevels;
        imageInfo.arrayLayers = 1;
        imageInfo.format = format;
        imageInfo.tiling = tiling;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imageInfo.usage = usage;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageInfo.samples = numSamples;
        imageInfo.flags = 0;
        if(vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS)
            throw std::runtime_error("Failed to create image!");
        VkMemoryRequirements memRequirements;
        vkGetImageMemoryRequirements(device, image, &memRequirements);
        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);
        if(vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS)
            throw std::runtime_error("Failed to allocate image memory!");
        vkBindImageMemory(device, image, imageMemory, 0);
    }
    void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout, uint32_t mipLevels = 1){
        VkCommandBuffer commandBuffer = beginSingleTimeCommands();
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = oldLayout;
        barrier.newLayout = newLayout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = mipLevels;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;
        VkPipelineStageFlags sourceStage;
        VkPipelineStageFlags destinationStage;
        if(oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL){
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        } else if(oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL){
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        } else if(oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL){
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            destinationStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        } else throw std::runtime_error("Unsupported layout transition!");
        if(newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL){
            barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
            if(hasStencilComponent(format))
                barrier.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
        } else barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vkCmdPipelineBarrier(commandBuffer, sourceStage, destinationStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
        endSingleTimeCommands(commandBuffer);
    }
    void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height){
        VkCommandBuffer commandBuffer = beginSingleTimeCommands();
        VkBufferImageCopy region{};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;
        region.imageOffset = {0, 0, 0};
        region.imageExtent = {width, height, 1};
        vkCmdCopyBufferToImage(commandBuffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        endSingleTimeCommands(commandBuffer);
    }
    void createCommandPool(){
        QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);
        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();
        if(vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS)
            throw std::runtime_error("Failed to create command pool!");
    }
    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer &buffer, VkDeviceMemory &bufferMemory){
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if(vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS)
            throw std::runtime_error("Failed to create buffer!");
        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device, buffer, &memRequirements);
        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);
        if(vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS)
            throw std::runtime_error("Failed to allocate buffer memory!");
        vkBindBufferMemory(device, buffer, bufferMemory, 0);
    }
    void createIndexBuffer(){
        std::vector<std::string> modelNames;
        for(const auto &model : models) if(model.second.loaded) modelNames.push_back(model.first);
        #pragma omp parallel for
        for(int i = 0; i < static_cast<int>(modelNames.size()); i++){
            const std::string& modelName = modelNames[i];
            auto& model = models[modelName];
            if(!model.loaded) continue;
            VkDeviceSize bufferSize = sizeof(model.indices[0]) * model.indices.size();
            VkBuffer stagingBuffer;
            VkDeviceMemory stagingBufferMemory;
            #pragma omp critical
            {
                createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);
            }
            VkBuffer indexBuffer;
            VkDeviceMemory indexBufferMemory;
            void* data;
            vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
            if(model.indices.size() > 10000) {
                #pragma omp parallel for
                for(int v = 0; v < static_cast<int>(model.indices.size()); v++)
                    memcpy(static_cast<char*>(data) + v * sizeof(uint32_t), 
                        &model.indices[v], sizeof(uint32_t));
            } else memcpy(data, model.indices.data(), (size_t) bufferSize);
            vkUnmapMemory(device, stagingBufferMemory);
            #pragma omp critical
            {
                createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBuffer, indexBufferMemory);
                copyBuffer(stagingBuffer, indexBuffer, bufferSize);
                vkDestroyBuffer(device, stagingBuffer, nullptr);
                vkFreeMemory(device, stagingBufferMemory, nullptr);
            }
            model.indexBuffer = indexBuffer;
            model.indexBufferMemory = indexBufferMemory;
        }
    }
    void loadModel(){
        for(auto &model : models){
            if(!model.second.loaded) continue;
            std::string file = "src/models/" + model.second.name + ".obj";
            ModelData data;
            objl::Loader loader;
            if(!loader.LoadFile(file.c_str())){
                std::cerr<<"Failed to load OBJ file"<<std::endl;
                model.second.loaded = false;
                continue;
            }
            objl::Mesh mesh = loader.LoadedMeshes[0];
            int vertexCount = static_cast<int>(mesh.Vertices.size());
            data.vertices.resize(vertexCount);
            #pragma omp parallel for
            for(int i=0; i<vertexCount; i++){
                size_t v = static_cast<size_t>(i);
                const auto &vertex = mesh.Vertices[v];
                data.vertices[v].pos = glm::vec3(vertex.Position.X, vertex.Position.Y, vertex.Position.Z);
                data.vertices[v].texCoord = glm::vec2(vertex.TextureCoordinate.X, vertex.TextureCoordinate.Y);
                data.vertices[v].normal = glm::vec3(vertex.Normal.X, vertex.Normal.Y, vertex.Normal.Z);
            }
            data.indices = mesh.Indices;
            model.second.indices = data.indices;
            model.second.vertices = data.vertices;
            model.second.loaded = true;
        }
    }
    void createVolumeGeometry(){
        std::vector<Vertex> cubeVertices = {
            {{renderMinX, renderMinY,  renderMaxZ}, {0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}},
            {{renderMaxX, renderMinY,  renderMaxZ}, {1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}},
            {{renderMaxX, renderMaxY, renderMaxZ}, {1.0f, 1.0f}, {0.0f, 0.0f, 1.0f}},
            {{renderMinX,  renderMaxY, renderMaxZ}, {0.0f, 1.0f}, {0.0f, 0.0f, 1.0f}},
            {{renderMinX, renderMinY, renderMinZ}, {1.0f, 0.0f}, {0.0f, 0.0f, -1.0f}},
            {{renderMaxX, renderMinY, renderMinZ}, {0.0f, 0.0f}, {0.0f, 0.0f, -1.0f}},
            {{renderMaxX, renderMaxY, renderMinZ}, {0.0f, 1.0f}, {0.0f, 0.0f, -1.0f}},
            {{renderMinX, renderMaxY, renderMinZ}, {1.0f, 1.0f}, {0.0f, 0.0f, -1.0f}}
        };
        std::vector<uint32_t> cubeIndices = {
            0, 1, 2, 2, 3, 0,
            4, 5, 6, 6, 7, 4,
            7, 3, 0, 0, 4, 7,
            1, 5, 6, 6, 2, 1,
            3, 2, 6, 6, 7, 3,
            0, 1, 5, 5, 4, 0
        };
        VkDeviceSize bufferSize = sizeof(cubeVertices[0]) * cubeVertices.size();
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);
        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
        memcpy(data, cubeVertices.data(), (size_t) bufferSize);
        vkUnmapMemory(device, stagingBufferMemory);
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, volumeData.vertexBuffer, volumeData.vertexBufferMemory);
        copyBuffer(stagingBuffer, volumeData.vertexBuffer, bufferSize);
        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
        VkDeviceSize indexBufferSize = sizeof(cubeIndices[0]) * cubeIndices.size();
        createBuffer(indexBufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);
        vkMapMemory(device, stagingBufferMemory, 0, indexBufferSize, 0, &data);
        memcpy(data, cubeIndices.data(), (size_t) indexBufferSize);
        vkUnmapMemory(device, stagingBufferMemory);
        createBuffer(indexBufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, volumeData.indexBuffer, volumeData.indexBufferMemory);
        copyBuffer(stagingBuffer, volumeData.indexBuffer, indexBufferSize);
        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }
    void updateVolumeTextures(){
        std::vector<glm::vec4> fanLocations(maxFans);
        std::vector<glm::vec4> fanDirections(maxFans);
        int index = 0;
        if(topFanEnabled){
            fanLocations[index] = glm::vec4(-0.22f, 4.2f, 1.6f, 1.0f);
            fanDirections[index] = glm::vec4(0.0f, 1.0f, 0.0f, 0.0f);
            index++;
        }
        if(frontFanEnabled){
            fanLocations[index] = glm::vec4(0.48f, 2.6f, 3.5f, 1.0f);
            fanDirections[index] = glm::vec4(0.0f, 0.0f, 1.0f, 0.0f);
            index++;
        }
        if(cpuFanEnabled){
            fanLocations[index] = glm::vec4(0.1f, 2.4f, 0.85f, 1.0f);
            fanDirections[index] = glm::vec4(0.0f, 0.0f, 1.0f, 0.0f);
            index++;
        }
        if(gpuEnabled){
            fanLocations[index] = glm::vec4(-0.36f, 0.75f, 0.34f, 1.0f);
            fanDirections[index] = glm::vec4(0.0f, 1.0f, 0.0f, 0.0f);
            index++;
            fanLocations[index] = glm::vec4(-0.36f, 0.75f, 2.71f, 1.0f);
            fanDirections[index] = glm::vec4(0.0f, 1.0f, 0.0f, 0.0f);
            index++;
        }
        for(int i=0; i<3; i++){
            if(backFanLocations[i] > 0.0f) continue;
            fanLocations[index] = glm::vec4(0.0f, 2.36343f + backFanLocations[i], -3.36426f, 1.0f);
            fanDirections[index] = glm::vec4(0.0f, 0.0f, 1.0f, 0.0f);
            index++;
        }
        currentPushConstants.numFans = index;
        for(int i=0; i<currentPushConstants.numFans && i<maxFans; i++){
            currentPushConstants.fanPositions[i] = fanLocations[i];
            currentPushConstants.fanDirections[i] = fanDirections[i];
        }
    }
    void resetSolidGrid(){
        if(!volumeSimulator){
            std::cerr << "ERROR: VolumeSimulator not initialized when trying to reset solid grid!" << std::endl;
            return;
        }
        try {
            std::vector<uint32_t> h_solidGrid(gridSizeX * gridSizeY * gridSizeZ, 0);
            std::vector<float> h_heatSources(gridSizeX * gridSizeY * gridSizeZ, 0.0f);

            #pragma omp parallel for collapse(3)
            for(int z = 0; z < gridSizeZ; z++){
                for(int y = 0; y < gridSizeY; y++){
                    for(int x = 0; x < gridSizeX; x++){
                        float worldX = worldMinX + (x + 0.5f) * cellSizeX;
                        float worldY = worldMinY + (y + 0.5f) * cellSizeY;
                        float worldZ = worldMinZ + (z + 0.5f) * cellSizeZ;
                        int index = x + y * gridSizeX + z * gridSizeX * gridSizeY;
                        bool insideSolid = false;
                        float heatSource = 0.0f;

                        // case
                        if((worldY < -4.22f && worldZ > 0.5f) || (worldZ > -1.8f && worldY < -4.26f && worldY > -4.22f) || (worldY > 4.4f && worldZ > -3.65 && worldZ < -0.65)) insideSolid = true;
                        if(worldX < -1.8f || worldX > 1.8f) insideSolid = true;
                        if(worldZ > 3.8f && (worldY < 1.5f || worldX < -0.6f || worldY > 3.8f)) insideSolid = true;

                        // gpu
                        if(gpuEnabled && worldZ > -0.53f && worldZ < 3.7 && worldY > 0.95f && worldY < 1.09f && worldX < 0.5f) insideSolid = true;

                        h_solidGrid[index] = insideSolid ? 1 : 0;

                         // ram
                        if(worldX < -0.95f && worldX > -1.57f && worldY < 3.6f && worldY > 1.2f && worldZ < 0.6f && worldZ > 0.18f) heatSource = 0.04f / 0.62;

                        // gpu
                        if(gpuEnabled && worldZ > -0.53f && worldZ < 3.7 && worldY > 0.8f && worldY < 1.09f && worldX < 0.5f) heatSource = 2.0f / 0.8f;

                        // cpu
                        if(worldZ > 1.2f && worldZ < 1.9f && worldY < 2.7f && worldY > 2.0f && worldX < -1.2f && worldX > -1.7f) heatSource = 0.8f / 0.25f;

                        // psu
                        if(worldY < -2.2f && worldY > -2.6f && worldZ > 0.4f && worldZ < 3.6f && worldX < 1.4f && worldX > -1.4f) heatSource = 0.28f / 3.6f;

                        h_heatSources[index] = heatSource;
                    }
                }
            }
            volumeSimulator->setSolidGrid(h_solidGrid.data());
            volumeSimulator->setHeatSources(h_heatSources.data());
        } catch(const std::exception& e) {
            std::cerr << "ERROR in resetSolidGrid: " << e.what() << std::endl;
        }
    }
    void updateVolumeDescriptorSets(){
        if(volumeSimulator->getTemperatureImageView() == VK_NULL_HANDLE
        || volumeSimulator->getVolumeImageView() == VK_NULL_HANDLE){
            std::cerr<<"Volume textures not ready!"<<std::endl;
            return;
        }
        for(size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++){
            VkDescriptorImageInfo volumeImageInfo{};
            volumeImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            volumeImageInfo.imageView = volumeSimulator->getVolumeImageView();
            volumeImageInfo.sampler = volumeSampler;
            VkDescriptorImageInfo temperatureImageInfo{};
            temperatureImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            temperatureImageInfo.imageView = volumeSimulator->getTemperatureImageView();
            temperatureImageInfo.sampler = volumeSampler;
            std::array<VkWriteDescriptorSet, 2> descriptorWrites{};
            descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[0].dstSet = volumeData.descriptorSets[i];
            descriptorWrites[0].dstBinding = 1;
            descriptorWrites[0].dstArrayElement = 0;
            descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites[0].descriptorCount = 1;
            descriptorWrites[0].pImageInfo = &volumeImageInfo;
            descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[1].dstSet = volumeData.descriptorSets[i];
            descriptorWrites[1].dstBinding = 2;
            descriptorWrites[1].dstArrayElement = 0;
            descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites[1].descriptorCount = 1;
            descriptorWrites[1].pImageInfo = &temperatureImageInfo;
            vkUpdateDescriptorSets(device, 2, descriptorWrites.data(), 0, nullptr);
        }
    }
    void createVertexBuffer(){
        std::vector<std::string> modelNames;
        for(const auto &model : models) if(model.second.loaded) modelNames.push_back(model.first);
        #pragma omp parallel for
        for(int i = 0; i < static_cast<int>(modelNames.size()); i++){
            const std::string& modelName = modelNames[i];
            auto& model = models[modelName];
            if(!model.loaded) continue;
            VkDeviceSize bufferSize = sizeof(model.vertices[0]) * model.vertices.size();
            VkBuffer stagingBuffer;
            VkDeviceMemory stagingBufferMemory;
            #pragma omp critical
            {
                createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);
            }
            void* data;
            vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
            if(model.vertices.size() > 10000) {
                #pragma omp parallel for
                for(int v = 0; v < static_cast<int>(model.vertices.size()); v++)
                    memcpy(static_cast<char*>(data) + v * sizeof(Vertex), 
                        &model.vertices[v], sizeof(Vertex));
            } else memcpy(data, model.vertices.data(), (size_t) bufferSize);
            vkUnmapMemory(device, stagingBufferMemory);
            VkBuffer vertexBuffer;
            VkDeviceMemory vertexBufferMemory;
            #pragma omp critical
            {
                createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer, vertexBufferMemory);
                copyBuffer(stagingBuffer, vertexBuffer, bufferSize);
                vkDestroyBuffer(device, stagingBuffer, nullptr);
                vkFreeMemory(device, stagingBufferMemory, nullptr);
            }
            model.vertexBuffer = vertexBuffer;
            model.vertexBufferMemory = vertexBufferMemory;
        }
    }
    void createTextResources(){
        const size_t maxChars = 1000;
        const size_t maxVertices = maxChars * 4;
        const size_t maxIndices = maxChars * 6;
        VkDeviceSize vertexBufferSize = sizeof(TextVertex) * maxVertices;
        createBuffer(vertexBufferSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, textVertexBuffer, textVertexBufferMemory);
        vkMapMemory(device, textVertexBufferMemory, 0, vertexBufferSize, 0, &textVertexBufferMapped);
        std::vector<uint32_t> indices;
        for(uint32_t i=0; i<maxChars; i++){
            uint32_t baseVertex = i * 4;
            indices.insert(indices.end(), {
                baseVertex + 0, baseVertex + 1, baseVertex + 2,
                baseVertex + 2, baseVertex + 3, baseVertex + 0
            });
        }
        VkDeviceSize indexBufferSize = sizeof(uint32_t) * indices.size();
        createBuffer(indexBufferSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, textIndexBuffer, textIndexBufferMemory);
        VkBuffer stagingBuffer; 
        VkDeviceMemory stagingBufferMemory;
        createBuffer(indexBufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);
        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, indexBufferSize, 0, &data);
        memcpy(data, indices.data(), indexBufferSize);
        vkUnmapMemory(device, stagingBufferMemory);
        copyBuffer(stagingBuffer, textIndexBuffer, indexBufferSize);
        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
        createTextPipeline();
    }
    void createUIResources(){
        std::vector<uint32_t> uiIndices = {0, 1, 2, 2, 3, 0};
        VkDeviceSize indexBufferSize = sizeof(uint32_t) * 6;
        createBuffer(indexBufferSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, uiIndexBuffer, uiIndexBufferMemory);
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(indexBufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);
        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, indexBufferSize, 0, &data);
        memcpy(data, uiIndices.data(), indexBufferSize);
        vkUnmapMemory(device, stagingBufferMemory);
        copyBuffer(stagingBuffer, uiIndexBuffer, indexBufferSize);
        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
        for(auto &uiObject : uiObjects){
            float xpos = uiObject.second.position.x;
            float ypos = uiObject.second.position.y;
            float w = uiObject.second.size.x;
            float h = uiObject.second.size.y;
            float ndcX1 = (2.0f * xpos / swapChainExtent.width) - 1.0f;
            float ndcY1 = (2.0f * ypos / swapChainExtent.height) - 1.0f;
            float ndcX2 = (2.0f * (xpos + w) / swapChainExtent.width) - 1.0f;
            float ndcY2 = (2.0f * (ypos + h) / swapChainExtent.height) - 1.0f;
            std::vector<TextVertex> uiVertices = {
                {{ndcX1, ndcY1}, {0.0f, 0.0f}},
                {{ndcX1, ndcY2}, {0.0f, 1.0f}},
                {{ndcX2, ndcY2}, {1.0f, 1.0f}},
                {{ndcX2, ndcY1}, {1.0f, 0.0f}}
            };
            VkDeviceSize vertexBufferSize = sizeof(TextVertex) * 4;
            createBuffer(vertexBufferSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uiObject.second.vertexBuffer, uiObject.second.vertexBufferMemory);
            vkMapMemory(device, uiObject.second.vertexBufferMemory, 0, vertexBufferSize, 0, &uiObject.second.vertexBufferMapped);
            memcpy(uiObject.second.vertexBufferMapped, uiVertices.data(), vertexBufferSize);
        }
    }
    void updateUIVertexBuffer(){
        for(auto &uiObject : uiObjects){
            float xpos = uiObject.second.position.x;
            float ypos = uiObject.second.position.y;
            float w = uiObject.second.size.x;
            float h = uiObject.second.size.y;
            float finalX = uiObject.second.anchorLeft ? xpos : (swapChainExtent.width - xpos - w);
            float ndcX1 = (2.0f * finalX / swapChainExtent.width) - 1.0f;
            float ndcY1 = (2.0f * ypos / swapChainExtent.height) - 1.0f;
            float ndcX2 = (2.0f * (finalX + w) / swapChainExtent.width) - 1.0f;
            float ndcY2 = (2.0f * (ypos + h) / swapChainExtent.height) - 1.0f;
            std::vector<TextVertex> uiVertices = {
                {{ndcX1, ndcY1}, {0.0f, 0.0f}},
                {{ndcX1, ndcY2}, {0.0f, 1.0f}},
                {{ndcX2, ndcY2}, {1.0f, 1.0f}},
                {{ndcX2, ndcY1}, {1.0f, 0.0f}}
            };
            memcpy(uiObject.second.vertexBufferMapped, uiVertices.data(), sizeof(TextVertex) * 4);
        }
    }
    void createTextDescriptorSetLayout(){
        VkDescriptorSetLayoutBinding samplerLayoutBinding{};
        samplerLayoutBinding.binding = 0;
        samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        samplerLayoutBinding.descriptorCount = 1;
        samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        samplerLayoutBinding.pImmutableSamplers = nullptr;
        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = 1;
        layoutInfo.pBindings = &samplerLayoutBinding;
        if(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &textDescriptorSetLayout) != VK_SUCCESS)
            throw std::runtime_error("Failed to create text descriptor set layout!");
    }
    void createUIDescriptorSetLayout(){
        VkDescriptorSetLayoutBinding samplerLayoutBinding{};
        samplerLayoutBinding.binding = 0;
        samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        samplerLayoutBinding.descriptorCount = 1;
        samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        samplerLayoutBinding.pImmutableSamplers = nullptr;
        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = 1;
        layoutInfo.pBindings = &samplerLayoutBinding;
        if(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &uiDescriptorSetLayout) != VK_SUCCESS)
            throw std::runtime_error("Failed to create text descriptor set layout!");
    }
    void updateTextVertexBuffer(){
        std::vector<TextVertex> vertices;
        vertices.reserve(1000 * 4);
        for(const auto &textObject : textObjects){
            if(!textObject.enabled) continue;
            if(!textObject.text.empty()){
                float x = textObject.position.x;
                float y = textObject.position.y;
                float scale = textObject.scale;
                float startX = textObject.anchorLeft ? x : (swapChainExtent.width - x);
                float startY = textObject.anchorTop ? y : (swapChainExtent.height - y);
                float baselineBearing = 0.0f;
                if(Characters.find('A') != Characters.end()) baselineBearing = Characters['A'].bearing.y * scale;
                for(char c : textObject.text){
                    if(Characters.find(c) == Characters.end()) continue;
                    const Character &ch = Characters[c];
                    float xpos = startX + ch.bearing.x * scale;
                    float ypos = startY + baselineBearing - ch.bearing.y * scale;
                    float w = ch.size.x * scale;
                    float h = ch.size.y * scale;
                    float ndcX1 = (2.0f * xpos / swapChainExtent.width) - 1.0f;
                    float ndcY1 = (2.0f * ypos / swapChainExtent.height) - 1.0f;
                    float ndcX2 = (2.0f * (xpos + w) / swapChainExtent.width) - 1.0f;
                    float ndcY2 = (2.0f * (ypos + h) / swapChainExtent.height) - 1.0f;
                    vertices.push_back({{ndcX1, ndcY1}, {0.0f, 0.0f}});
                    vertices.push_back({{ndcX1, ndcY2}, {0.0f, 1.0f}});
                    vertices.push_back({{ndcX2, ndcY2}, {1.0f, 1.0f}});
                    vertices.push_back({{ndcX2, ndcY1}, {1.0f, 0.0f}});
                    startX += (ch.advance >> 6) * scale;
                }
            }
        }
        if(!vertices.empty() && vertices.size() * sizeof(TextVertex) <= 1000 * 4 * sizeof(TextVertex))
            memcpy(textVertexBufferMapped, vertices.data(), vertices.size() * sizeof(TextVertex));
    }
    void createTextDescriptorPool(){
        VkDescriptorPoolSize poolSize{};
        poolSize.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSize.descriptorCount = static_cast<uint32_t>(Characters.size());
        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = 1;
        poolInfo.pPoolSizes = &poolSize;
        poolInfo.maxSets = static_cast<uint32_t>(Characters.size());
        if(vkCreateDescriptorPool(device, &poolInfo, nullptr, &textDescriptorPool) != VK_SUCCESS)
            throw std::runtime_error("Failed to create text descriptor pool!");
    }
    void createUIDescriptorPool(){
        VkDescriptorPoolSize poolSize{};
        poolSize.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSize.descriptorCount = static_cast<uint32_t>(uiObjects.size());
        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        poolInfo.poolSizeCount = 1;
        poolInfo.pPoolSizes = &poolSize;
        poolInfo.maxSets = static_cast<uint32_t>(uiObjects.size());
        if(vkCreateDescriptorPool(device, &poolInfo, nullptr, &uiDescriptorPool) != VK_SUCCESS)
            throw std::runtime_error("Failed to create text descriptor pool!");
    }
    void createComputeDescriptorSetLayout(){
        std::vector<VkDescriptorSetLayoutBinding> bindings(17);
        for(int i = 0; i < 17; i++){
            bindings[i].binding = i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            bindings[i].pImmutableSamplers = nullptr;
        }
        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();
        if(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &computeDescriptorSetLayout) != VK_SUCCESS)
            throw std::runtime_error("Failed to create compute descriptor set layout!");
    }
    void createComputeDescriptorPool(){
        VkDescriptorPoolSize poolSize{};
        poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSize.descriptorCount = 17;
        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = 1;
        poolInfo.pPoolSizes = &poolSize;
        poolInfo.maxSets = 1;
        if(vkCreateDescriptorPool(device, &poolInfo, nullptr, &computeDescriptorPool) != VK_SUCCESS)
            throw std::runtime_error("Failed to create compute descriptor pool!");
    }
    void createTextDescriptorSets(){
        for(auto &[c, character] : Characters){
            VkDescriptorSetAllocateInfo allocInfo{};
            allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            allocInfo.descriptorPool = textDescriptorPool;
            allocInfo.descriptorSetCount = 1;
            allocInfo.pSetLayouts = &textDescriptorSetLayout;
            VkDescriptorSet descriptorSet;
            if(vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet) != VK_SUCCESS)
                throw std::runtime_error("Failed to allocate text descriptor sets!");
            VkDescriptorImageInfo imageInfo{};
            imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imageInfo.imageView = character.textureImageView;
            imageInfo.sampler = textureSampler;
            VkWriteDescriptorSet descriptorWrite{};
            descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrite.dstSet = descriptorSet;
            descriptorWrite.dstBinding = 0;
            descriptorWrite.dstArrayElement = 0;
            descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrite.descriptorCount = 1;
            descriptorWrite.pImageInfo = &imageInfo;
            vkUpdateDescriptorSets(device, 1, &descriptorWrite, 0, nullptr);
            character.descriptorSet = descriptorSet;
        }
    }
    void createUIDescriptorSets(){
        for(auto &ui : uiObjects){
            TextureData &textureData = textureAtlas[ui.second.name];
            VkDescriptorSetAllocateInfo allocInfo{};
            allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            allocInfo.descriptorPool = uiDescriptorPool;
            allocInfo.descriptorSetCount = 1;
            allocInfo.pSetLayouts = &uiDescriptorSetLayout;
            VkDescriptorSet descriptorSet;
            if(vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet) != VK_SUCCESS)
                throw std::runtime_error("Failed to allocate UI descriptor sets!");
            VkDescriptorImageInfo imageInfo{};
            imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imageInfo.imageView = textureData.imageView;
            imageInfo.sampler = textureSampler;
            VkWriteDescriptorSet descriptorWrite{};
            descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrite.dstSet = descriptorSet;
            descriptorWrite.dstBinding = 0;
            descriptorWrite.dstArrayElement = 0;
            descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrite.descriptorCount = 1;
            descriptorWrite.pImageInfo = &imageInfo;
            vkUpdateDescriptorSets(device, 1, &descriptorWrite, 0, nullptr);
            ui.second.descriptorSet = descriptorSet;
        }
    }
    void updateUIDescriptorSet(std::string name){
        try {
            if(uiObjects.find(name) == uiObjects.end()) {
                std::cerr << "ERROR: UI object '" << name << "' not found!" << std::endl;
                return;
            }
            
            auto &ui = uiObjects[name];
            if(textureAtlas.find(ui.name) == textureAtlas.end()) {
                std::cerr << "ERROR: Texture '" << ui.name << "' not found in atlas!" << std::endl;
                return;
            }
            
            TextureData &textureData = textureAtlas[ui.name];
            if(textureData.imageView == VK_NULL_HANDLE) {
                std::cerr << "ERROR: Invalid image view for texture '" << ui.name << "'!" << std::endl;
                return;
            }
            
            VkDescriptorImageInfo imageInfo{};
            imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imageInfo.imageView = textureData.imageView;
            imageInfo.sampler = textureSampler;
            VkWriteDescriptorSet descriptorWrite{};
            descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrite.dstSet = ui.descriptorSet;
            descriptorWrite.dstBinding = 0;
            descriptorWrite.dstArrayElement = 0;
            descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrite.descriptorCount = 1;
            descriptorWrite.pImageInfo = &imageInfo;
            vkUpdateDescriptorSets(device, 1, &descriptorWrite, 0, nullptr);
        } catch(const std::exception& e) {
            std::cerr << "ERROR in updateUIDescriptorSet for '" << name << "': " << e.what() << std::endl;
        }
    }
    void prepareTextureAtlas(){
        for(auto &texture : textureAtlas){
            stbi_set_flip_vertically_on_load(true);
            int texWidth, texHeight, texChannels;
            std::string texturePath = "src/textures/ui/" + texture.first + ".png";
            stbi_uc* pixels = stbi_load(texturePath.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
            VkDeviceSize imageSize = texWidth * texHeight * 4;
            if(!pixels) throw std::runtime_error("Failed to load texture image!");
            VkBuffer stagingBuffer;
            VkDeviceMemory stagingBufferMemory;
            createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);
            void* data;
            VkImage textureImage;
            VkDeviceMemory textureImageMemory;
            vkMapMemory(device, stagingBufferMemory, 0, imageSize, 0, &data);
            memcpy(data, pixels, (size_t) imageSize);
            vkUnmapMemory(device, stagingBufferMemory);
            stbi_image_free(pixels);
            createImage(texWidth, texHeight, 1, VK_SAMPLE_COUNT_1_BIT, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, textureImage, textureImageMemory);
            transitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            copyBufferToImage(stagingBuffer, textureImage, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));
            transitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            vkDestroyBuffer(device, stagingBuffer, nullptr);
            vkFreeMemory(device, stagingBufferMemory, nullptr);
            texture.second.image = textureImage;
            texture.second.imageMemory = textureImageMemory;
            createTextureImageView(VK_FORMAT_R8G8B8A8_SRGB, textureImage, texture.second.imageView);
        }
    }
    void createUniformBuffers(){
        VkDeviceSize bufferSize = sizeof(UniformBufferObject);
        for(auto &model : models){
            if(!model.second.loaded) continue;
            model.second.uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
            model.second.uniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
            model.second.uniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);
            for(size_t i=0; i<MAX_FRAMES_IN_FLIGHT; i++){
                createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, model.second.uniformBuffers[i], model.second.uniformBuffersMemory[i]);
                vkMapMemory(device, model.second.uniformBuffersMemory[i], 0, bufferSize, 0, &model.second.uniformBuffersMapped[i]);
            }
        }
    }
    void createDescriptorPool(){
        size_t modelCount = 0;
        for(const auto &model : models) if(model.second.loaded) modelCount++;
        std::array<VkDescriptorPoolSize, 2> poolSizes{};
        poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes[0].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT * modelCount * 2);
        poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSizes[1].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT * modelCount * 4);
        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        poolInfo.maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT * modelCount);
        if(vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS)
            throw std::runtime_error("Failed to create descriptor pool!");
    }
    void createDescriptorSets(){
        for(auto &model : models){
            if(!model.second.loaded) continue;
            std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, descriptorSetLayout);
            VkDescriptorSetAllocateInfo allocInfo{};
            allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            allocInfo.descriptorPool = descriptorPool;
            allocInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
            allocInfo.pSetLayouts = layouts.data();
            model.second.descriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
            if(vkAllocateDescriptorSets(device, &allocInfo, model.second.descriptorSets.data()) != VK_SUCCESS)
                throw std::runtime_error("Failed to allocate descriptor sets!");
            for(size_t i=0; i<MAX_FRAMES_IN_FLIGHT; i++){
                VkDescriptorBufferInfo bufferInfo{};
                bufferInfo.buffer = model.second.uniformBuffers[i];
                bufferInfo.offset = 0;
                bufferInfo.range = sizeof(UniformBufferObject);
                std::array<VkDescriptorImageInfo, 4> imageInfos{};
                for(size_t j = 0; j < 4; j++){
                    imageInfos[j].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                    imageInfos[j].imageView = model.second.textureImageView[j];
                    imageInfos[j].sampler = model.second.textureSampler[j];
                }

                std::array<VkWriteDescriptorSet, 5> descriptorWrites{};
                descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                descriptorWrites[0].dstSet = model.second.descriptorSets[i];
                descriptorWrites[0].dstBinding = 0;
                descriptorWrites[0].dstArrayElement = 0;
                descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                descriptorWrites[0].descriptorCount = 1;
                descriptorWrites[0].pBufferInfo = &bufferInfo;
                for(size_t j = 0; j < 4; j++){
                    descriptorWrites[j + 1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    descriptorWrites[j + 1].dstSet = model.second.descriptorSets[i];
                    descriptorWrites[j + 1].dstBinding = static_cast<uint32_t>(j + 1);
                    descriptorWrites[j + 1].dstArrayElement = 0;
                    descriptorWrites[j + 1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                    descriptorWrites[j + 1].descriptorCount = 1;
                    descriptorWrites[j + 1].pImageInfo = &imageInfos[j];
                }
                vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
            }
        }
    }
    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size){
        VulkanCommandUtils::copyBuffer(device, commandPool, graphicsQueue, srcBuffer, dstBuffer, size);
    }
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties){
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
        for(uint32_t i=0; i<memProperties.memoryTypeCount; i++)
            if(typeFilter & (1 << i) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
                return i;
        throw std::runtime_error("Failed to find suitable memory type!");
    }
    VkCommandBuffer beginSingleTimeCommands(){
        return VulkanCommandUtils::beginSingleTimeCommands(device, commandPool);
    }
    void endSingleTimeCommands(VkCommandBuffer commandBuffer){
        VulkanCommandUtils::endSingleTimeCommands(device, commandPool, graphicsQueue, commandBuffer);
    }
    void createCommandBuffers(){
        commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = (uint32_t) commandBuffers.size();
        if(vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS)
            throw std::runtime_error("Failed to allocate command buffers!");
    }
    void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex){
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = 0;
        beginInfo.pInheritanceInfo = nullptr;
        if(vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS)
            throw std::runtime_error("Failed to begin recording command buffer!");
        VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = swapChainFramebuffers[imageIndex];
        renderPassInfo.renderArea.offset = {0, 0};
        renderPassInfo.renderArea.extent = swapChainExtent;
        std::array<VkClearValue, 2> clearValues{};
        clearValues[0].color = {0.012f, 0.012f, 0.05f, 1.0f};
        clearValues[1].depthStencil = {1.0f, 0};
        renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        renderPassInfo.pClearValues = clearValues.data();
        vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);
        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = static_cast<float>(swapChainExtent.width);
        viewport.height = static_cast<float>(swapChainExtent.height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
        VkRect2D scissor{};
        scissor.offset = {0, 0};
        scissor.extent = swapChainExtent;
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);
        VkDeviceSize offsets[] = {0};
        std::map<std::string, ModelOrientation> orientations = {
            {"fancpu", {
                .position = glm::vec3(0.112314f, 2.302f, 0.894292f),
                .rotation = glm::vec3(0.0f, 0.0f, currentTime * fanStrength),
                .scale = glm::vec3(0.63f, 0.63f, 1.0f)
            }},
            {"fantop", {
                .position = glm::vec3(-0.18513f, 4.0492f, 1.5371f),
                .rotation = glm::vec3(-PI / 2.0f, 0.0f, currentTime * fanStrength)
            }},
            {"fanfront", {
                .position = glm::vec3(0.48427, 2.60047, 3.42548),
                .rotation = glm::vec3(0.0f, 0.0f, currentTime * fanStrength)
            }},
            {"gpufan1", {
                .position = glm::vec3(-0.364, 0.3565, 0.4438),
                .rotation = glm::vec3(-PI / 2.0f, 0.0f, currentTime * fanStrength),
            }},
            {"gpufan2", {
                .position = glm::vec3(-0.364, 0.3565, 2.5671),
                .rotation = glm::vec3(-PI / 2.0f, 0.0f, currentTime * fanStrength),
            }}
        };
        for(auto &model : models){
            if(!model.second.loaded || !model.second.enabled) continue;
            if(model.first == "glass" || model.first == "shield") continue;
            if(model.first.substr(0, 7) == "fanback"){
                int i = std::stoi(model.first.substr(7, 1)) - 1;
                if(backFanLocations[i] > 0.0f) continue;
                model.second.modelMatrix = glm::mat4(1.0f);
                model.second.modelMatrix = glm::translate(model.second.modelMatrix, glm::vec3(0.0f, 2.36343f + backFanLocations[i], -3.36426));
                model.second.modelMatrix = glm::rotate(model.second.modelMatrix, currentTime * fanStrength, glm::vec3(0.0f, 0.0f, 1.0f));
            } else if(model.first.substr(0, 7) == "backfan"){
                int i = std::stoi(model.first.substr(7, 1)) - 1;
                if(backFanLocations[i] > 0.0f) continue;
                model.second.modelMatrix = glm::mat4(1.0f);
                model.second.modelMatrix = glm::translate(model.second.modelMatrix, glm::vec3(0.0f, backFanLocations[i], 0.0f));
            } else if(model.second.name == "fan" || model.second.name == "gpufan"){
                ModelOrientation &orientation = orientations[model.first];
                model.second.modelMatrix = glm::mat4(1.0f);
                model.second.modelMatrix = glm::translate(model.second.modelMatrix, orientation.position);
                model.second.modelMatrix = glm::scale(model.second.modelMatrix, orientation.scale);
                model.second.modelMatrix = glm::rotate(model.second.modelMatrix, orientation.rotation.x, glm::vec3(1.0f, 0.0f, 0.0f));
                model.second.modelMatrix = glm::rotate(model.second.modelMatrix, orientation.rotation.y, glm::vec3(0.0f, 1.0f, 0.0f));
                model.second.modelMatrix = glm::rotate(model.second.modelMatrix, orientation.rotation.z, glm::vec3(0.0f, 0.0f, 1.0f));
            }
            VkBuffer vertexBuffer = model.second.vertexBuffer;
            VkBuffer indexBuffer = model.second.indexBuffer;
            vkCmdBindVertexBuffers(commandBuffer, 0, 1, &vertexBuffer, offsets);
            vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT32);
            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &model.second.descriptorSets[currentFrame], 0, nullptr);
            vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(model.second.indices.size()), 1, 0, 0, 0);
        }
        VkBuffer vertexBuffer = models["shield"].vertexBuffer;
        VkBuffer indexBuffer = models["shield"].indexBuffer;
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, &vertexBuffer, offsets);
        vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &models["shield"].descriptorSets[currentFrame], 0, nullptr);
        vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(models["shield"].indices.size()), 1, 0, 0, 0);
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, volumePipeline);
        VkBuffer volumeVertexBuffer = volumeData.vertexBuffer;
        VkBuffer volumeIndexBuffer = volumeData.indexBuffer;
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, &volumeVertexBuffer, offsets);
        vkCmdBindIndexBuffer(commandBuffer, volumeIndexBuffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, volumePipelineLayout, 0, 1, &volumeData.descriptorSets[currentFrame], 0, nullptr);
        vkCmdDrawIndexed(commandBuffer, 36, 1, 0, 0, 0);
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);
        vertexBuffer = models["glass"].vertexBuffer;
        indexBuffer = models["glass"].indexBuffer;
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, &vertexBuffer, offsets);
        vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &models["glass"].descriptorSets[currentFrame], 0, nullptr);
        vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(models["glass"].indices.size()), 1, 0, 0, 0);
        renderUI(commandBuffer);
        renderText(commandBuffer);
        vkCmdEndRenderPass(commandBuffer);
        if(vkEndCommandBuffer(commandBuffer) != VK_SUCCESS)
            throw std::runtime_error("Failed to record command buffer!");
    }
    void renderText(VkCommandBuffer commandBuffer){
        if(textObjects.empty()) return;
        updateTextVertexBuffer();
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, textPipeline);
        VkDeviceSize offsets[] = {0};
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, &textVertexBuffer, offsets);
        vkCmdBindIndexBuffer(commandBuffer, textIndexBuffer, 0, VK_INDEX_TYPE_UINT32);
        uint32_t indexOffset = 0;
        for(const auto &textObject : textObjects){
            if(!textObject.enabled) continue;
            UIPushConstants pushData = {textObject.color, 0};
            vkCmdPushConstants(commandBuffer, textPipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(UIPushConstants), &pushData);
            for(char c : textObject.text){
                if(Characters.find(c) == Characters.end()) continue;
                Character &ch = Characters[c];
                vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, textPipelineLayout, 0, 1, &ch.descriptorSet, 0, nullptr);
                vkCmdDrawIndexed(commandBuffer, 6, 1, indexOffset, 0, 0);
                indexOffset += 6;
            }
        }
    }
    void renderUI(VkCommandBuffer commandBuffer){
        UIPushConstants pushData = {{1.0f, 1.0f, 1.0f}, 1};
        updateUIVertexBuffer();
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, textPipeline);
        vkCmdPushConstants(commandBuffer, textPipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(UIPushConstants), &pushData);
        VkDeviceSize offsets[] = {0};
        const auto &uiObjectWindow = uiObjects.find("uiWindow");
        if(uiObjectWindow != uiObjects.end() && uiObjectWindow->second.enabled){
            vkCmdBindVertexBuffers(commandBuffer, 0, 1, &uiObjectWindow->second.vertexBuffer, offsets);
            vkCmdBindIndexBuffer(commandBuffer, uiIndexBuffer, 0, VK_INDEX_TYPE_UINT32);
            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, textPipelineLayout, 0, 1, &uiObjectWindow->second.descriptorSet, 0, nullptr);
            vkCmdDrawIndexed(commandBuffer, 6, 1, 0, 0, 0);
        }
        for(const auto &uiObject : uiObjects){
            if(!uiObject.second.enabled) continue;
            if(uiObject.first == "uiWindow") continue;
            vkCmdBindVertexBuffers(commandBuffer, 0, 1, &uiObject.second.vertexBuffer, offsets);
            vkCmdBindIndexBuffer(commandBuffer, uiIndexBuffer, 0, VK_INDEX_TYPE_UINT32);
            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, textPipelineLayout, 0, 1, &uiObject.second.descriptorSet, 0, nullptr);
            vkCmdDrawIndexed(commandBuffer, 6, 1, 0, 0, 0);
        }
    }
    void createRenderPass(){
        VkAttachmentDescription colorAttachment{};
        colorAttachment.format = swapChainImageFormat;
        colorAttachment.samples = msaaSamples;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        VkAttachmentReference colorAttachmentRef{};
        colorAttachmentRef.attachment = 0;
        colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        VkAttachmentDescription colorAttachmentResolve{};
        colorAttachmentResolve.format = swapChainImageFormat;
        colorAttachmentResolve.samples = VK_SAMPLE_COUNT_1_BIT;
        colorAttachmentResolve.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachmentResolve.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachmentResolve.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachmentResolve.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachmentResolve.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachmentResolve.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        VkAttachmentReference colorAttachmentResolveRef{};
        colorAttachmentResolveRef.attachment = 2;
        colorAttachmentResolveRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        VkAttachmentDescription depthAttachment{};
        depthAttachment.format = findDepthFormat();
        depthAttachment.samples = msaaSamples;
        depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        VkAttachmentReference depthAttachmentRef{};
        depthAttachmentRef.attachment = 1;
        depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorAttachmentRef;
        subpass.pDepthStencilAttachment = &depthAttachmentRef;
        subpass.pResolveAttachments = &colorAttachmentResolveRef;
        VkSubpassDependency dependency{};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependency.srcAccessMask = 0;
        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        std::array<VkAttachmentDescription, 3> attachments = {colorAttachment, depthAttachment, colorAttachmentResolve};
        VkRenderPassCreateInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
        renderPassInfo.pAttachments = attachments.data();
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;
        renderPassInfo.dependencyCount = 1;
        renderPassInfo.pDependencies = &dependency;
        if(vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS)
            throw std::runtime_error("Failed to create render pass!");
    }
    VkShaderModule createShaderModule(const std::vector<char> &code){
        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = code.size();
        createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());
        VkShaderModule shaderModule;
        if(vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS)
            throw std::runtime_error("Failed to create shader module!");
        return shaderModule;
    }
    void createFramebuffers(){
        swapChainFramebuffers.resize(swapChainImageViews.size());
        for(size_t i=0; i<swapChainImageViews.size(); i++){
            std::array<VkImageView, 3> attachments = {colorImageView, depthImageView, swapChainImageViews[i]};
            VkFramebufferCreateInfo framebufferInfo{};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass;
            framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
            framebufferInfo.pAttachments = attachments.data();
            framebufferInfo.width = swapChainExtent.width;
            framebufferInfo.height = swapChainExtent.height;
            framebufferInfo.layers = 1;
            if(vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS)
                throw std::runtime_error("Failed to create framebuffer!");
        }
    }
    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device){
        SwapChainSupportDetails details;
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);
        uint32_t formatCount;
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);
        if(formatCount != 0){
            details.formats.resize(formatCount);
            vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
        }
        uint32_t presentModeCount;
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);
        if(presentModeCount != 0){
            details.presentModes.resize(presentModeCount);
            vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
        }
        return details;
    }
    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR> &availableFormats){
        for(const auto &availableFormat : availableFormats){
            if(availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
                return availableFormat;
        }
        return availableFormats[0];
    }
    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR> &availablePresentModes){
        for(const auto &availablePresentMode : availablePresentModes){
            if(availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR)
                return availablePresentMode;
        }
        return VK_PRESENT_MODE_FIFO_KHR;
    }
    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR &capabilities){
        if(capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
            return capabilities.currentExtent;
        else{
            int width, height;
            glfwGetFramebufferSize(window, &width, &height);
            VkExtent2D actualExtent = {
                static_cast<uint32_t>(width),
                static_cast<uint32_t>(height)
            };
            actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
            actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
            return actualExtent;
        }
    }
    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device){
        QueueFamilyIndices indices;
        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);
        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());
        for(uint32_t i=0; i<queueFamilyCount; i++){
            if(queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) indices.graphicsFamily = i;
            VkBool32 presentSupport = false;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);
            if(presentSupport) indices.presentFamily = i;
            if(indices.isComplete()) break;
        }
        return indices;
    }
    bool isDeviceSuitable(VkPhysicalDevice device){
        QueueFamilyIndices indices = findQueueFamilies(device);
        bool extensionsSupported = checkDeviceExtensionSupport(device);
        bool swapChainAdequate = false;
        if(extensionsSupported){
            SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
            swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
        }
        VkPhysicalDeviceFeatures supportedFeatures;
        vkGetPhysicalDeviceFeatures(device, &supportedFeatures);
        return indices.isComplete() && extensionsSupported && swapChainAdequate && supportedFeatures.samplerAnisotropy;
    }
    bool checkDeviceExtensionSupport(VkPhysicalDevice device){
        uint32_t extensionCount;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);
        std::vector<VkExtensionProperties> availableExtensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());
        std::set<std::string> requiredExtensions = {deviceExtensions.begin(), deviceExtensions.end()};
        for(const auto& extension : availableExtensions){
            requiredExtensions.erase(extension.extensionName);
        }
        return requiredExtensions.empty();
    }
    VkSampleCountFlagBits getMaxUsableSampleCount(){
        VkPhysicalDeviceProperties physicalDeviceProperties;
        vkGetPhysicalDeviceProperties(physicalDevice, &physicalDeviceProperties);
        VkSampleCountFlags counts = physicalDeviceProperties.limits.framebufferColorSampleCounts & physicalDeviceProperties.limits.framebufferDepthSampleCounts;
        if(counts & VK_SAMPLE_COUNT_64_BIT) return VK_SAMPLE_COUNT_64_BIT;
        if(counts & VK_SAMPLE_COUNT_32_BIT) return VK_SAMPLE_COUNT_32_BIT;
        if(counts & VK_SAMPLE_COUNT_16_BIT) return VK_SAMPLE_COUNT_16_BIT;
        if(counts & VK_SAMPLE_COUNT_8_BIT) return VK_SAMPLE_COUNT_8_BIT;
        if(counts & VK_SAMPLE_COUNT_4_BIT) return VK_SAMPLE_COUNT_4_BIT;
        if(counts & VK_SAMPLE_COUNT_2_BIT) return VK_SAMPLE_COUNT_2_BIT;
        return VK_SAMPLE_COUNT_1_BIT;
    }
    void pickPhysicalDevice(){
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
        if(deviceCount == 0) throw std::runtime_error("Failed to find GPUs with Vulkan support!");
        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());
        std::multimap<int, VkPhysicalDevice> candidates;
        for(const auto &device : devices){
            int score = rateDeviceSuitability(device);
            candidates.insert(std::make_pair(score, device));
        }
        if(candidates.rbegin()->first > 0 && candidates.rbegin()->first > 0)
            physicalDevice = candidates.rbegin()->second;
        else throw std::runtime_error("Failed to find a suitable GPU!");
        msaaSamples = getMaxUsableSampleCount();
    }
    int rateDeviceSuitability(VkPhysicalDevice device){
        VkPhysicalDeviceProperties deviceProperties;
        VkPhysicalDeviceFeatures deviceFeatures;
        vkGetPhysicalDeviceProperties(device, &deviceProperties);
        vkGetPhysicalDeviceFeatures(device, &deviceFeatures);
        int score = 0;
        if(deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) score += 1000;
        #ifdef __APPLE__
            score += 500;
        #endif
        score += deviceProperties.limits.maxImageDimension2D;
        return score;
    }
    void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT &createInfo){
        createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        createInfo.messageSeverity = 
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT
        |   VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
        |   VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        createInfo.messageType = 
            VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT
        |   VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
        |   VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        createInfo.pfnUserCallback = debugCallback;
    }
    void setupDebugMessenger(){
        if(!enableValidationLayers) return;
        VkDebugUtilsMessengerCreateInfoEXT createInfo;
        populateDebugMessengerCreateInfo(createInfo);
        if(CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS)
            throw std::runtime_error("Failed to set up debug messenger!");
    }
    std::vector<const char*> getRequiredExtensions(){
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions;
        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
        std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
        if(enableValidationLayers) extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        #ifdef __APPLE__
            extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
            extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
        #endif
        return extensions;
    }
    bool checkValidationLayerSupport(){
        uint32_t layerCount;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
        std::vector<VkLayerProperties> availableLayers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());
        for(const char* layerName : validationLayers){
            bool layerFound = false;
            for(const auto &layerProperties : availableLayers){
                if(strcmp(layerName, layerProperties.layerName) == 0){
                    layerFound = true;
                    break;
                }
            }
            if(!layerFound) return false;
        }
        return true;
    }
    void setCurrentHoverElement(float scaledMouseX, float scaledMouseY){
        std::vector<std::string> elements = {
            "gpuCheckbox", "cpuFanCheckbox", "topFanCheckbox", "frontFanCheckbox", "pressureCheckbox", "fanUp", "fanDown", "slider1", "slider2", "slider3", "uiToggle"
        };
        for(const auto &element : elements){
            if(uiObjects.find(element) == uiObjects.end()) continue;
            const auto &uiObj = uiObjects[element];
            float actualX = uiObj.anchorLeft ? uiObj.position.x : (swapChainExtent.width - uiObj.position.x - uiObj.size.x);
            float actualY = uiObj.position.y;
            if(scaledMouseX >= actualX && scaledMouseX <= actualX + uiObj.size.x
            && scaledMouseY >= actualY && scaledMouseY <= actualY + uiObj.size.y){
                hoverElement = element;
                return;
            }
        }
        const auto &uiWindow = uiObjects["uiWindow"];
        float actualX = uiWindow.anchorLeft ? uiWindow.position.x : (swapChainExtent.width - uiWindow.position.x - uiWindow.size.x);
        float actualY = uiWindow.position.y;
        if(scaledMouseX >= actualX && scaledMouseX <= actualX + uiWindow.size.x
        && scaledMouseY >= actualY && scaledMouseY <= actualY + uiWindow.size.y){
            hoverElement = "uiWindow";
            return;
        }
        hoverElement = "";
    }
    std::map<std::string, bool*> getHoverBoolean = {
        {"gpuCheckbox", &gpuEnabled},
        {"cpuFanCheckbox", &cpuFanEnabled},
        {"topFanCheckbox", &topFanEnabled},
        {"frontFanCheckbox", &frontFanEnabled},
        {"pressureCheckbox", &pressureEnabled}
    };
    std::map<std::string, std::vector<std::string>> uiToModel = {
        {"gpuCheckbox", {"gpu", "gpufan1", "gpufan2"}},
        {"cpuFanCheckbox", {"fancpu", "cpufan"}},
        {"topFanCheckbox", {"fantop", "topfan"}},
        {"frontFanCheckbox", {"fanfront", "frontfan"}}
    };
    static void processInput(GLFWwindow* window){
        auto app = reinterpret_cast<GustGrid*>(glfwGetWindowUserPointer(window));
        static bool wasPressed = false;
        static double lastClickTime = 0.0;
        bool isPressed = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
        double currentTime = glfwGetTime();
        if(isPressed && !wasPressed){
            if(currentTime - lastClickTime < 0.2){
                wasPressed = isPressed;
                return;
            }
            lastClickTime = currentTime;
            auto hoverElement = app->hoverElement;
            if(hoverElement.empty()) return;
            if(app->getHoverBoolean.find(hoverElement) != app->getHoverBoolean.end()){
                *app->getHoverBoolean[hoverElement] = !(*app->getHoverBoolean[hoverElement]);
                app->uiObjects[hoverElement].name = *app->getHoverBoolean[hoverElement] ? "checkbox/checked" : "checkbox/unchecked";
                if(app->uiToModel.find(hoverElement) != app->uiToModel.end()){
                    for(const auto &modelName : app->uiToModel[hoverElement])
                        app->models[modelName].enabled = *app->getHoverBoolean[hoverElement];
                    app->shouldUpdateGrid = true;
                }
                app->updateUIDescriptorSet(hoverElement);
                app->shouldUpdateFans = true;
            } else if(hoverElement == "fanUp"){
                int currentFanCount = 0;
                for(int i=0; i<3; i++) if(app->backFanLocations[i]<=0.0f) currentFanCount++;
                if(currentFanCount==3) return;
                else if(currentFanCount==0) app->backFanLocations[0] = 0.0f;
                else if(currentFanCount==1){
                    app->backFanLocations[0] = std::max(app->backFanLocations[0], -2.5f);
                    app->backFanLocations[1] = app->backFanLocations[0] - 2.5f;
                }
                else if(currentFanCount==2) for(int i=0; i<3; i++) app->backFanLocations[i] = i * -2.5f;
                app->shouldUpdateFans = true;
            } else if(hoverElement == "fanDown"){
                int currentFanCount = 0;
                for(int i=0; i<3; i++) if(app->backFanLocations[i]<=0.0f) currentFanCount++;
                if(currentFanCount==0) return;
                app->backFanLocations[currentFanCount-1] = 1.0f;
                app->shouldUpdateFans = true;
            } else if(hoverElement == "uiToggle"){
                app->showUI = !app->showUI;
                app->uiObjects[hoverElement].name = app->showUI ? "toggle/close" : "toggle/open";
                app->updateUIDescriptorSet(hoverElement);
                for(int i=4; i<14; i++) app->textObjects[i].enabled = app->showUI;
                app->uiObjects["uiToggle"].position.x = app->showUI ? 470.0f : 25.0f;
                app->uiObjects["gpuCheckbox"].enabled = app->showUI;
                app->uiObjects["cpuFanCheckbox"].enabled = app->showUI;
                app->uiObjects["topFanCheckbox"].enabled = app->showUI;
                app->uiObjects["frontFanCheckbox"].enabled = app->showUI;
                app->uiObjects["pressureCheckbox"].enabled = app->showUI;
                app->uiObjects["fanUp"].enabled = app->showUI;
                app->uiObjects["fanDown"].enabled = app->showUI;
                app->uiObjects["uiWindow"].enabled = app->showUI;
                app->hoverElement = "";
            } else if(hoverElement == "slider1" || hoverElement == "slider2" || hoverElement == "slider3") return;
        }
        wasPressed = isPressed;
        if(!isPressed) app->firstMouse = true;
    }
    static void mouseCallback(GLFWwindow* window, double xpos, double ypos){
        auto app = reinterpret_cast<GustGrid*>(glfwGetWindowUserPointer(window));
        float xposFloat = static_cast<float>(xpos);
        float yposFloat = static_cast<float>(ypos);
        int windowWidth, windowHeight;
        int framebufferWidth, framebufferHeight;
        glfwGetWindowSize(window, &windowWidth, &windowHeight);
        glfwGetFramebufferSize(window, &framebufferWidth, &framebufferHeight);
        float scaleX = static_cast<float>(framebufferWidth) / static_cast<float>(windowWidth);
        float scaleY = static_cast<float>(framebufferHeight) / static_cast<float>(windowHeight);
        float scaledMouseX = xposFloat * scaleX;
        float scaledMouseY = yposFloat * scaleY;
        if(glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_RELEASE){
            app->lastMouseX = xposFloat;
            app->lastMouseY = yposFloat;
            app->firstMouse = true;
            app->setCurrentHoverElement(scaledMouseX, scaledMouseY);
            return;
        }
        if(app->firstMouse){
            app->lastMouseX = xposFloat;
            app->lastMouseY = yposFloat;
            app->firstMouse = false;
        } else if(app->showUI) {
            app->setCurrentHoverElement(scaledMouseX, scaledMouseY);
            if(app->hoverElement == "slider1" || app->hoverElement == "slider2" || app->hoverElement == "slider3"){
                int i = app->hoverElement.back() - '1';
                if(app->backFanLocations[i] > 0.0f) return;
                app->backFanLocations[i] = -((scaledMouseX - (app->swapChainExtent.width - 310.0f)) / 230.0f) * 5.0f;
                if(i<2 && app->backFanLocations[i+1] <= 0.0f && app->backFanLocations[i+1] - app->backFanLocations[i] >= -2.5f) app->backFanLocations[i] = app->backFanLocations[i+1] + 2.5f;
                else if(i>0 && app->backFanLocations[i] - app->backFanLocations[i-1] >= -2.5f) app->backFanLocations[i] = app->backFanLocations[i-1] - 2.5f;
                app->backFanLocations[i] = std::clamp(app->backFanLocations[i], -5.0f, 0.0f);
                app->shouldUpdateFans = true;
                return;
            } else if(app->hoverElement != "") return;
        } else{
            app->setCurrentHoverElement(scaledMouseX, scaledMouseY);
            if(app->hoverElement == "uiToggle") return;
        }
        float xOffset = (xposFloat - app->lastMouseX) * app->mouseSensitivity;
        float yOffset = (app->lastMouseY - yposFloat) * app->mouseSensitivity;
        app->lastMouseX = xposFloat;
        app->lastMouseY = yposFloat;
        app->camYaw += xOffset;
        app->camPitch += yOffset;
        if(app->camPitch > 1.6f) app->camPitch = 1.6f;
        else if(app->camPitch < -1.6f) app->camPitch = -1.6f;
        app->camPos = glm::vec3(
            sin(app->camYaw) * cos(app->camPitch) * app->camRadius,
            sin(app->camPitch) * app->camRadius,
            cos(app->camYaw) * cos(app->camPitch) * app->camRadius
        );
    }
    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
        VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
        VkDebugUtilsMessageTypeFlagsEXT messageType,
        const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
        void* pUserData
    ){
        std::cerr<<"Validation layer: "<<pCallbackData->pMessage<<std::endl;
        return VK_FALSE;
    }
    static std::vector<char> readFile(const std::string &filename){
        std::ifstream file(filename, std::ios::ate | std::ios::binary);
        if(!file.is_open()) throw std::runtime_error("Failed to open file!");
        size_t fileSize = (size_t) file.tellg();
        std::vector<char> buffer(fileSize);
        file.seekg(0);
        file.read(buffer.data(), fileSize);
        file.close();
        return buffer;
    }
    static void framebufferResizeCallback(GLFWwindow* window, int width, int height){
        auto app = reinterpret_cast<GustGrid*>(glfwGetWindowUserPointer(window));
        app->framebufferResized = true;
    }
};
int main(){
    GustGrid app;
    try{
        app.run();
    } catch(const std::exception& e){
        std::cerr<<e.what()<<std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}