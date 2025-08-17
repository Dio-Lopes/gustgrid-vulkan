#define GLFW_INCLUDE_VULKAN
#include <glfw/include/GLFW/glfw3.h>
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <vector>
#include <cstring>
#include <map>
#include <array>
#define maxFans 8
#define gridSizeX 64
#define gridSizeY 256
#define gridSizeZ 128
class VulkanMemoryPool;
class SimulationMemory;
class VolumeSimulator {
private:
    VkDevice device;
    VkCommandPool commandPool;
    VkQueue computeQueue;
    VkPhysicalDevice physicalDevice;
    VkDescriptorSetLayout sharedDescriptorSetLayout;
    VkDescriptorPool sharedDescriptorPool;
    std::vector<VkDescriptorSet> descriptorSets;
    SimulationMemory* simulationMemory;
    std::vector<VkSemaphore> computeFinishedSemaphores;
    std::vector<VkCommandBuffer> computeCommandBuffers;
    uint32_t currentFrame = 0;
    struct ComputeKernel {
        VkPipeline pipeline;
        VkPipelineLayout pipelineLayout;
        VkShaderModule shaderModule;
        std::string name;
        glm::uvec3 workgroupSize;
        bool needsBarrier;
    };
    std::map<std::string, ComputeKernel> kernels;
public:
    struct ComputePushConstants {
        alignas(16) glm::vec4 gridSize;
        alignas(16) glm::vec4 worldMin;
        alignas(16) glm::vec4 worldMax;
        alignas(16) glm::vec4 cellSize;
        alignas(4) float deltaTime;
        alignas(4) uint32_t numFans;
        alignas(4) int displayPressure;
        alignas(4) uint32_t padding;
    };
    VolumeSimulator(VkDevice device, VkCommandPool commandPool, VkQueue computeQueue, VkPhysicalDevice physicalDevice);
    ~VolumeSimulator();
    void initialize(VkDescriptorSetLayout descriptorSetLayout, VkDescriptorPool descriptorPool);
    void cleanup();
    void addKernel(const std::string &name, const std::string &shaderPath, glm::uvec3 workgroupSize = glm::uvec3(8, 8, 8), bool needsBarrier = true);
    VkImageView getVolumeImageView();
    VkImageView getTemperatureImageView();
    void setSolidGrid(const uint32_t* solidGrid, size_t numCells = gridSizeX * gridSizeY * gridSizeZ);
    void setHeatSources(const float* heatSources, size_t numCells = gridSizeX * gridSizeY * gridSizeZ);
    VkSemaphore dispatchKernel(const std::string &kernelName, glm::uvec3 gridSize, const ComputePushConstants &pushConstants = {});
    std::array<float, 2> getProbeOut() const;
    void updateVolumeImages(bool displayPressure);
    void initSimulation(int numCells);
    void updateDescriptorSetsWithBuffers();
    void setFanParams(const glm::vec4* positions, const glm::vec4* directions, uint32_t count);
    void swapPressureBuffers();
    void copyFinalPressureToMain();
    float computeResidualSum();
    void setCurrentFrame(uint32_t frame) { currentFrame = frame; }
    void cleanupSimulation();
    static std::vector<char> readFile(const std::string &filename);
private:
    void createKernelPipelineLayout(ComputeKernel &kernel);
    void createKernelPipeline(ComputeKernel &kernel);
    void addMemoryBarrier(VkCommandBuffer commandBuffer);
    void createSharedDescriptorSets();
    VkShaderModule createShaderModule(const std::vector<char>& code);
    VkCommandBuffer beginSingleTimeCommands();
    void endSingleTimeCommands(VkCommandBuffer commandBuffer);
    void copyBufferToImage(VkCommandBuffer commandBuffer, VkBuffer srcBuffer, VkImage dstImage, uint32_t width, uint32_t height, uint32_t depth);
public:
    void clearTempSumsCPU(int numCells);
    void clearPressureCPU(int numCells);
    void copyVelocityTempCPU(int numCells);
    void resetFanAccessCPU(int numCells);
};