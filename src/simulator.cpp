#define GLFW_INCLUDE_VULKAN
#include <glfw/include/GLFW/glfw3.h>
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <map>
#include <unordered_set>
#include <limits>
#include <fstream>
#include <algorithm>
#include <array>
#include <simulator.h>
#include <vulkan_utils.h>

#define gridSizeX 64
#define gridSizeY 256
#define gridSizeZ 128
#define worldMinX -2.0f
#define worldMaxX 2.0f
#define worldMinY -4.5f
#define worldMaxY 4.5f
#define worldMinZ -4.0f
#define worldMaxZ 4.0f
#define maxFans 8

static bool g_vulkanDeviceValid = true;

class VulkanMemoryPool {
private:
    struct Block {
        VkBuffer buffer;
        VkDeviceMemory memory;
        size_t size;
        bool inUse;
        void* mappedData;
        Block(VkBuffer buf, VkDeviceMemory mem, size_t sz)
            : buffer(buf), memory(mem), size(sz), inUse(false), mappedData(nullptr) {}
    };
    std::vector<Block> blocks;
    std::unordered_set<VkBuffer> allocatedBuffers;
    VkDevice device;
    VkPhysicalDevice physicalDevice;
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties){
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
        for(uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
            if((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
                return i;
        throw std::runtime_error("Failed to find suitable memory type");
    }
public:
    static void markDeviceDestroyed(){
        g_vulkanDeviceValid = false;
    }
    VulkanMemoryPool(VkDevice device, VkPhysicalDevice physicalDevice)
        : device(device), physicalDevice(physicalDevice) {}
    ~VulkanMemoryPool(){
        if(g_vulkanDeviceValid)
            for(auto& block : blocks){
                if(block.mappedData) vkUnmapMemory(device, block.memory);
                vkDestroyBuffer(device, block.buffer, nullptr);
                vkFreeMemory(device, block.memory, nullptr);
            }
    }
    VkBuffer allocate(size_t size, VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VkMemoryPropertyFlags properties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT){
        size = ((size + 255) / 256) * 256;
        for(auto& block : blocks){
            if(!block.inUse && block.size >= size){
                block.inUse = true;
                allocatedBuffers.insert(block.buffer);
                return block.buffer;
            }
        }
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VkBuffer newBuffer;
        if(vkCreateBuffer(device, &bufferInfo, nullptr, &newBuffer) != VK_SUCCESS)
            throw std::runtime_error("Failed to create buffer");
        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device, newBuffer, &memRequirements);
        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);
        VkDeviceMemory newMemory;
        if(vkAllocateMemory(device, &allocInfo, nullptr, &newMemory) != VK_SUCCESS)
            throw std::runtime_error("Failed to allocate buffer memory");
        if(vkBindBufferMemory(device, newBuffer, newMemory, 0) != VK_SUCCESS)
            throw std::runtime_error("Failed to bind buffer memory");
        blocks.emplace_back(newBuffer, newMemory, size);
        blocks.back().inUse = true;
        allocatedBuffers.insert(newBuffer);
        return newBuffer;
    }
    void deallocate(VkBuffer buffer){
        if(buffer == VK_NULL_HANDLE) return;
        if(allocatedBuffers.find(buffer) == allocatedBuffers.end())
            throw std::runtime_error("Buffer not found in memory pool");
        for(auto& block : blocks){
            if(block.buffer == buffer){
                block.inUse = false;
                allocatedBuffers.erase(buffer);
                if(block.mappedData){
                    vkUnmapMemory(device, block.memory);
                    block.mappedData = nullptr;
                }
                return;
            }
        }
    }
    static VulkanMemoryPool &getInstance(VkDevice device = VK_NULL_HANDLE, VkPhysicalDevice physicalDevice = VK_NULL_HANDLE){
        static std::unique_ptr<VulkanMemoryPool> instance;
        if(!instance && device != VK_NULL_HANDLE)
            instance = std::make_unique<VulkanMemoryPool>(device, physicalDevice);
        return *instance;
    }
    static std::unique_ptr<VulkanMemoryPool>& getInstancePtr(){
        static std::unique_ptr<VulkanMemoryPool> instance;
        return instance;
    }
    static void destroyInstance(){
        auto& instance = getInstancePtr();
        if(instance) instance.reset();
    }
};
class SimulationMemory {
private:
    VkBuffer d_divergence = VK_NULL_HANDLE;
    VkBuffer d_pressure = VK_NULL_HANDLE;
    VkBuffer d_pressureOut = VK_NULL_HANDLE;
    VkBuffer d_residual = VK_NULL_HANDLE;
    VkBuffer d_tempVelocity = VK_NULL_HANDLE;
    VkBuffer d_velocity = VK_NULL_HANDLE;
    VkBuffer d_speed = VK_NULL_HANDLE;
    VkBuffer d_heatSources = VK_NULL_HANDLE;
    VkBuffer d_temperature = VK_NULL_HANDLE;
    VkBuffer d_pressureTemp = VK_NULL_HANDLE;
    VkBuffer d_tempTemperature = VK_NULL_HANDLE;
    VkBuffer d_tempSum = VK_NULL_HANDLE;
    VkBuffer d_weightSum = VK_NULL_HANDLE;
    VkBuffer d_tempSumDiss = VK_NULL_HANDLE;
    VkBuffer d_fanAccess = VK_NULL_HANDLE;
    VkBuffer d_solidGrid = VK_NULL_HANDLE;
    VkImage d_temperatureImage = VK_NULL_HANDLE;
    VkImage d_volumeImage = VK_NULL_HANDLE;
    VkImageView d_temperatureImageView = VK_NULL_HANDLE;
    VkImageView d_volumeImageView = VK_NULL_HANDLE;
    VkDeviceMemory d_temperatureImageMemory = VK_NULL_HANDLE;
    VkDeviceMemory d_volumeImageMemory = VK_NULL_HANDLE;
    int allocatedGridSize = 0;
    VkDevice device;
    VkPhysicalDevice physicalDevice;
    VkCommandPool commandPool;
    VkQueue computeQueue;
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties){
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
        for(uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
            if((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
                return i;
        throw std::runtime_error("Failed to find suitable memory type");
    }
    void create3DStorageImage(uint32_t width, uint32_t height, uint32_t depth, VkFormat format, VkImage &image, VkDeviceMemory &memory, VkImageView &imageView){
        VkImageCreateInfo imageInfo{};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType = VK_IMAGE_TYPE_3D;
        imageInfo.format = format;
        imageInfo.extent.width = width;
        imageInfo.extent.height = height;
        imageInfo.extent.depth = depth;
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        if(vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS)
            throw std::runtime_error("Failed to create 3D storage image");
        VkMemoryRequirements memRequirements;
        vkGetImageMemoryRequirements(device, image, &memRequirements);
        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if(vkAllocateMemory(device, &allocInfo, nullptr, &memory) != VK_SUCCESS)
            throw std::runtime_error("Failed to allocate memory for 3D storage image");
        if(vkBindImageMemory(device, image, memory, 0) != VK_SUCCESS)
            throw std::runtime_error("Failed to bind memory to 3D storage image");
        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_3D;
        viewInfo.format = format;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;
        if(vkCreateImageView(device, &viewInfo, nullptr, &imageView) != VK_SUCCESS)
            throw std::runtime_error("Failed to create image view for 3D storage image");
    }
public:
    static void markDeviceDestroyed(){
        g_vulkanDeviceValid = false;
        auto &pool = VulkanMemoryPool::getInstance();
        pool.markDeviceDestroyed();
    }
    SimulationMemory(VkDevice device, VkPhysicalDevice physicalDevice, VkCommandPool commandPool, VkQueue computeQueue)
        : device(device), physicalDevice(physicalDevice), commandPool(commandPool), computeQueue(computeQueue) {}
    ~SimulationMemory(){
        cleanup();
    }
    void cleanup(){
        if(allocatedGridSize == 0) return;
        if(g_vulkanDeviceValid && computeQueue != VK_NULL_HANDLE) vkQueueWaitIdle(computeQueue);
        if(g_vulkanDeviceValid && d_temperatureImageView != VK_NULL_HANDLE){
            vkDestroyImageView(device, d_temperatureImageView, nullptr);
            d_temperatureImageView = VK_NULL_HANDLE;
        }
        if(g_vulkanDeviceValid && d_temperatureImage != VK_NULL_HANDLE){
            vkDestroyImage(device, d_temperatureImage, nullptr);
            vkFreeMemory(device, d_temperatureImageMemory, nullptr);
            d_temperatureImage = VK_NULL_HANDLE;
            d_temperatureImageMemory = VK_NULL_HANDLE;
        }
        if(g_vulkanDeviceValid && d_volumeImageView != VK_NULL_HANDLE){
            vkDestroyImageView(device, d_volumeImageView, nullptr);
            d_volumeImageView = VK_NULL_HANDLE;
        }
        if(g_vulkanDeviceValid && d_volumeImage != VK_NULL_HANDLE){
            vkDestroyImage(device, d_volumeImage, nullptr);
            vkFreeMemory(device, d_volumeImageMemory, nullptr);
            d_volumeImage = VK_NULL_HANDLE;
            d_volumeImageMemory = VK_NULL_HANDLE;
        }
        try{
            auto &pool = VulkanMemoryPool::getInstance();
            std::vector<VkBuffer> buffers = {
                d_divergence, d_pressure, d_pressureOut, d_residual,
                d_tempVelocity, d_velocity, d_speed, d_temperature, d_heatSources,
                d_pressureTemp, d_tempTemperature, d_tempSum,
                d_weightSum, d_tempSumDiss, d_fanAccess, d_solidGrid
            };
            for(auto buffer : buffers){
                if(g_vulkanDeviceValid && buffer != VK_NULL_HANDLE) pool.deallocate(buffer);
            }
        } catch(const std::exception &e){
            std::cerr << "Error during cleanup: " << e.what() << std::endl;
        }
        d_divergence = d_pressure = d_pressureOut = d_residual =
        d_tempVelocity = d_velocity = d_speed = d_temperature =
        d_pressureTemp = d_tempTemperature = d_tempSum = d_heatSources =
        d_weightSum = d_tempSumDiss = d_fanAccess = d_solidGrid = VK_NULL_HANDLE;
        allocatedGridSize = 0;
    }
    void ensureAllocated(int numCells){
        if(allocatedGridSize >= numCells) return;
        cleanup();
        auto &pool = VulkanMemoryPool::getInstance();
        VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        VkMemoryPropertyFlags properties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        d_divergence = pool.allocate(numCells * sizeof(float), usage, properties);
        d_pressure = pool.allocate(numCells * sizeof(float), usage, properties);
        d_pressureOut = pool.allocate(numCells * sizeof(float), usage, properties);
        d_residual = pool.allocate(numCells * sizeof(float), usage, properties);
        d_tempVelocity = pool.allocate(numCells * 4 * sizeof(float), usage, properties);
        d_heatSources = pool.allocate(numCells * sizeof(float), usage, properties);
        d_velocity = pool.allocate(numCells * 4 * sizeof(float), usage, properties);
        d_speed = pool.allocate(numCells * sizeof(float), usage, properties);
        d_temperature = pool.allocate(numCells * sizeof(float), usage, properties);
        d_pressureTemp = pool.allocate(numCells * sizeof(float), usage, properties);
        d_tempTemperature = pool.allocate(numCells * sizeof(float), usage, properties);
        d_tempSum = pool.allocate(numCells * sizeof(float), usage, properties);
        d_weightSum = pool.allocate(numCells * sizeof(float), usage, properties);
        d_tempSumDiss = pool.allocate(numCells * sizeof(float), usage, properties);
        d_fanAccess = pool.allocate(numCells * maxFans * sizeof(uint32_t), usage, properties);
        d_solidGrid = pool.allocate(numCells * sizeof(unsigned char), usage, properties);
        initializeBuffers(numCells);
        uint32_t gridX = gridSizeX;
        uint32_t gridY = gridSizeY;
        uint32_t gridZ = gridSizeZ;
        create3DStorageImage(gridX, gridY, gridZ, VK_FORMAT_R32_SFLOAT, d_temperatureImage, d_temperatureImageMemory, d_temperatureImageView);
        create3DStorageImage(gridX, gridY, gridZ, VK_FORMAT_R32_SFLOAT, d_volumeImage, d_volumeImageMemory, d_volumeImageView);
        allocatedGridSize = numCells;
    }
    VkBuffer getDivergence() { return d_divergence; }
    VkBuffer getPressure() { return d_pressure; }
    VkBuffer getPressureOut() { return d_pressureOut; }
    VkBuffer getPressureTemp() { return d_pressureTemp; }
    VkBuffer getResidual() { return d_residual; }
    VkBuffer getTempVelocity() { return d_tempVelocity; }
    VkBuffer getHeatSources() { return d_heatSources; }
    VkBuffer getVelocity() { return d_velocity; }
    VkBuffer getSpeed() { return d_speed; }
    VkBuffer getTemperature() { return d_temperature; }
    VkBuffer getTempTemperature() { return d_tempTemperature; }
    VkBuffer getTempSum() { return d_tempSum; }
    VkBuffer getWeightSum() { return d_weightSum; }
    VkBuffer getTempSumDiss() { return d_tempSumDiss; }
    VkBuffer getFanAccess() { return d_fanAccess; }
    VkBuffer getSolidGrid() { return d_solidGrid; }
    VkImage getTemperatureImage() { return d_temperatureImage; }
    VkImage getVolumeImage() { return d_volumeImage; }
    VkImageView getTemperatureImageView() { return d_temperatureImageView; }
    VkImageView getVolumeImageView() { return d_volumeImageView; }
    void swapPressureBuffers() {
        std::swap(d_pressure, d_pressureOut);
    }
    void swapIterationBuffers(){
        std::swap(d_pressureTemp, d_pressureOut);
    }
    static SimulationMemory &getInstance(VkDevice device = VK_NULL_HANDLE, VkPhysicalDevice physicalDevice = VK_NULL_HANDLE, VkCommandPool commandPool = VK_NULL_HANDLE, VkQueue computeQueue = VK_NULL_HANDLE){
        static std::unique_ptr<SimulationMemory> instance;
        if(!instance && device != VK_NULL_HANDLE)
            instance = std::make_unique<SimulationMemory>(device, physicalDevice, commandPool, computeQueue);
        return *instance;
    }
    static std::unique_ptr<SimulationMemory>& getInstancePtr(){
        static std::unique_ptr<SimulationMemory> instance;
        return instance;
    }
    static void destroyInstance(){
        auto& instance = getInstancePtr();
        if(instance){
            instance->cleanup();
            instance.reset();
        }
        auto &pool = VulkanMemoryPool::getInstance();
        pool.destroyInstance();
    }
    void createStagingBuffer(VkDeviceSize size, VkBuffer &stagingBuffer, VkDeviceMemory &stagingMemory){
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if(vkCreateBuffer(device, &bufferInfo, nullptr, &stagingBuffer) != VK_SUCCESS)
            throw std::runtime_error("Failed to create staging buffer");
        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device, stagingBuffer, &memRequirements);
        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if(vkAllocateMemory(device, &allocInfo, nullptr, &stagingMemory) != VK_SUCCESS)
            throw std::runtime_error("Failed to allocate memory for staging buffer");
        if(vkBindBufferMemory(device, stagingBuffer, stagingMemory, 0) != VK_SUCCESS)
            throw std::runtime_error("Failed to bind memory to staging buffer");
    }
    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size){
        VulkanCommandUtils::copyBuffer(device, getCommandPool(), getComputeQueue(), srcBuffer, dstBuffer, size);
    }
private:
    void initializeBuffers(int numCells){
        size_t bufferSize = numCells * sizeof(float);
        std::vector<float> initialData(numCells, 0.0f);
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingMemory;
        createStagingBuffer(bufferSize, stagingBuffer, stagingMemory);
        void* data;
        vkMapMemory(device, stagingMemory, 0, bufferSize, 0, &data);
        std::memcpy(data, initialData.data(), bufferSize);
        vkUnmapMemory(device, stagingMemory);
        std::vector<VkBuffer> floatBuffers = {
            d_divergence, d_pressure, d_pressureOut, d_residual, d_heatSources,
            d_speed, d_pressureTemp, d_tempSum, d_weightSum, d_tempSumDiss
        };
        for(auto &buffer : floatBuffers) copyBuffer(stagingBuffer, buffer, bufferSize);
        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingMemory, nullptr);
        std::vector<float> initialTemperatureData(numCells, 22.0f);
        createStagingBuffer(bufferSize, stagingBuffer, stagingMemory);
        vkMapMemory(device, stagingMemory, 0, bufferSize, 0, &data);
        std::memcpy(data, initialTemperatureData.data(), bufferSize);
        vkUnmapMemory(device, stagingMemory);
        copyBuffer(stagingBuffer, d_temperature, bufferSize);
        copyBuffer(stagingBuffer, d_tempTemperature, bufferSize);
        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingMemory, nullptr);
        size_t velocityBufferSize = numCells * 4 * sizeof(float);
        std::vector<float> initialVelocityData(numCells * 4, 0.0f);
        createStagingBuffer(velocityBufferSize, stagingBuffer, stagingMemory);
        vkMapMemory(device, stagingMemory, 0, velocityBufferSize, 0, &data);
        std::memcpy(data, initialVelocityData.data(), velocityBufferSize);
        vkUnmapMemory(device, stagingMemory);
        copyBuffer(stagingBuffer, d_tempVelocity, velocityBufferSize);
        copyBuffer(stagingBuffer, d_velocity, velocityBufferSize);
        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingMemory, nullptr);
        size_t fanAccessBufferSize = numCells * maxFans * sizeof(uint32_t);
        size_t solidGridBufferSize = numCells * sizeof(unsigned char);
        std::vector<uint32_t> initialFanAccessData(numCells * maxFans, 0);
        std::vector<unsigned char> initialSolidGridData(numCells, 0);
        createStagingBuffer(fanAccessBufferSize, stagingBuffer, stagingMemory);
        vkMapMemory(device, stagingMemory, 0, fanAccessBufferSize, 0 , &data);
        std::memcpy(data, initialFanAccessData.data(), fanAccessBufferSize);
        vkUnmapMemory(device, stagingMemory);
        copyBuffer(stagingBuffer, d_fanAccess, fanAccessBufferSize);
        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingMemory, nullptr);
        createStagingBuffer(solidGridBufferSize, stagingBuffer, stagingMemory);
        vkMapMemory(device, stagingMemory, 0, solidGridBufferSize, 0, &data);
        std::memcpy(data, initialSolidGridData.data(), solidGridBufferSize);
        vkUnmapMemory(device, stagingMemory);
        copyBuffer(stagingBuffer, d_solidGrid, solidGridBufferSize);
        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingMemory, nullptr);
    }
    VkCommandPool getCommandPool() const { return commandPool; }
    VkQueue getComputeQueue() const { return computeQueue; }
};
VolumeSimulator::VolumeSimulator(VkDevice device, VkCommandPool commandPool, VkQueue computeQueue, VkPhysicalDevice physicalDevice)
: device(device), commandPool(commandPool), computeQueue(computeQueue), physicalDevice(physicalDevice) {
    VulkanMemoryPool::getInstance(device, physicalDevice);
    simulationMemory = &SimulationMemory::getInstance(device, physicalDevice, commandPool, computeQueue);
}
VolumeSimulator::~VolumeSimulator() {
    cleanup();
}
void VolumeSimulator::initialize(VkDescriptorSetLayout descriptorSetLayout, VkDescriptorPool descriptorPool){
    sharedDescriptorSetLayout = descriptorSetLayout;
    sharedDescriptorPool = descriptorPool;
    createSharedDescriptorSets();
    computeFinishedSemaphores.resize(2);
    computeCommandBuffers.resize(2);
    for(size_t i = 0; i < computeFinishedSemaphores.size(); i++){
        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        if(vkCreateSemaphore(device, &semaphoreInfo, nullptr, &computeFinishedSemaphores[i]) != VK_SUCCESS)
            throw std::runtime_error("Failed to create compute finished semaphore");
    }
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = static_cast<uint32_t>(computeCommandBuffers.size());
    if(vkAllocateCommandBuffers(device, &allocInfo, computeCommandBuffers.data()) != VK_SUCCESS)
        throw std::runtime_error("Failed to allocate compute command buffers");
}
void VolumeSimulator::initSimulation(int numCells){
    simulationMemory->ensureAllocated(numCells);
    updateDescriptorSetsWithBuffers();
}
void VolumeSimulator::setSolidGrid(const uint32_t* solidGrid, size_t numCells){
    if(numCells != static_cast<size_t>(gridSizeX * gridSizeY * gridSizeZ))
        throw std::runtime_error("Solid grid size mismatch");
    std::vector<unsigned char> charData(numCells);
    for(size_t i = 0; i < numCells; i++)
        charData[i] = static_cast<unsigned char>(solidGrid[i]);
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingMemory;
    simulationMemory->createStagingBuffer(numCells * sizeof(unsigned char), stagingBuffer, stagingMemory);
    void* data;
    vkMapMemory(device, stagingMemory, 0, numCells * sizeof(unsigned char), 0, &data);
    std::memcpy(data, charData.data(), numCells * sizeof(unsigned char));
    vkUnmapMemory(device, stagingMemory);
    simulationMemory->copyBuffer(stagingBuffer, simulationMemory->getSolidGrid(), numCells * sizeof(unsigned char));
    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingMemory, nullptr);
}
void VolumeSimulator::setHeatSources(const float* heatSources, size_t numCells){
    if(numCells != static_cast<size_t>(gridSizeX * gridSizeY * gridSizeZ))
        throw std::runtime_error("Heat sources size mismatch");
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingMemory;
    simulationMemory->createStagingBuffer(numCells * sizeof(float), stagingBuffer, stagingMemory);
    void* data;
    vkMapMemory(device, stagingMemory, 0, numCells * sizeof(float), 0, &data);
    std::memcpy(data, heatSources, numCells * sizeof(float));
    vkUnmapMemory(device, stagingMemory);
    simulationMemory->copyBuffer(stagingBuffer, simulationMemory->getHeatSources(), numCells * sizeof(float));
    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingMemory, nullptr);
}
void VolumeSimulator::updateDescriptorSetsWithBuffers(){
    std::vector<VkBuffer> buffers = {
        simulationMemory->getDivergence(),
        simulationMemory->getPressure(),
        simulationMemory->getPressureOut(),
        simulationMemory->getResidual(),
        simulationMemory->getTempVelocity(),
        simulationMemory->getVelocity(),
        simulationMemory->getSpeed(),
        simulationMemory->getTemperature(),
        simulationMemory->getPressureTemp(),
        simulationMemory->getTempTemperature(),
        simulationMemory->getTempSum(),
        simulationMemory->getWeightSum(),
        simulationMemory->getTempSumDiss(),
        simulationMemory->getFanAccess(),
        simulationMemory->getSolidGrid(),
        simulationMemory->getHeatSources()
    };
    for(size_t i = 0; i < buffers.size(); i++)
        if(buffers[i] == VK_NULL_HANDLE)
            throw std::runtime_error("Buffer " + std::to_string(i) + " is not allocated");
    if(descriptorSets.empty()) throw std::runtime_error("Descriptor sets not allocated");
    std::vector<VkDescriptorBufferInfo> bufferInfos(buffers.size());
    std::vector<VkWriteDescriptorSet> descriptorWrites(buffers.size());
    for(size_t i = 0; i < buffers.size(); i++){
        bufferInfos[i].buffer = buffers[i];
        bufferInfos[i].offset = 0;
        bufferInfos[i].range = VK_WHOLE_SIZE;
        descriptorWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[i].pNext = nullptr;
        descriptorWrites[i].dstSet = descriptorSets[0];
        descriptorWrites[i].dstBinding = static_cast<uint32_t>(i);
        descriptorWrites[i].dstArrayElement = 0;
        descriptorWrites[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorWrites[i].descriptorCount = 1;
        descriptorWrites[i].pBufferInfo = &bufferInfos[i];
        descriptorWrites[i].pImageInfo = nullptr;
        descriptorWrites[i].pTexelBufferView = nullptr;
    }
    vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
}
VkImageView VolumeSimulator::getVolumeImageView() {
    return simulationMemory->getVolumeImageView();
}
VkImageView VolumeSimulator::getTemperatureImageView() {
    return simulationMemory->getTemperatureImageView();
}
void VolumeSimulator::copyBufferToImage(VkCommandBuffer commandBuffer, VkBuffer srcBuffer, VkImage dstImage, uint32_t width, uint32_t height, uint32_t depth){
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = dstImage;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {width, height, depth};
    vkCmdCopyBufferToImage(commandBuffer, srcBuffer, dstImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}
void VolumeSimulator::addKernel(const std::string &name, const std::string &shaderPath, glm::uvec3 workgroupSize, bool needsBarrier){
    ComputeKernel kernel;
    kernel.name = name;
    kernel.workgroupSize = workgroupSize;
    kernel.needsBarrier = needsBarrier;
    std::vector<char> shaderCode = readFile(shaderPath);
    kernel.shaderModule = createShaderModule(shaderCode);
    createKernelPipelineLayout(kernel);
    createKernelPipeline(kernel);
    kernels[name] = kernel;
}
void VolumeSimulator::updateVolumeImages(bool displayPressure){
    VkCommandBuffer commandBuffer = computeCommandBuffers[currentFrame];
    vkResetCommandBuffer(commandBuffer, 0);
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if(vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS)
        throw std::runtime_error("Failed to begin command buffer");
    copyBufferToImage(commandBuffer, simulationMemory->getTemperature(), simulationMemory->getTemperatureImage(), gridSizeX, gridSizeY, gridSizeZ);
    if(displayPressure)
        copyBufferToImage(commandBuffer, simulationMemory->getPressure(), simulationMemory->getVolumeImage(), gridSizeX, gridSizeY, gridSizeZ);
    else copyBufferToImage(commandBuffer, simulationMemory->getSpeed(), simulationMemory->getVolumeImage(), gridSizeX, gridSizeY, gridSizeZ);
    vkEndCommandBuffer(commandBuffer);
    VkFence timeoutFence;
    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if(vkCreateFence(device, &fenceInfo, nullptr, &timeoutFence) != VK_SUCCESS)
        throw std::runtime_error("Failed to create timeout fence");
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &computeFinishedSemaphores[currentFrame];
    if(vkQueueSubmit(computeQueue, 1, &submitInfo, timeoutFence) != VK_SUCCESS){
        vkDestroyFence(device, timeoutFence, nullptr);
        throw std::runtime_error("Failed to submit compute command buffer");
    }
    const uint64_t TIMEOUT_NANOSECONDS = 5000000000ULL;
    VkResult result = vkWaitForFences(device, 1, &timeoutFence, VK_TRUE, TIMEOUT_NANOSECONDS);
    if(result == VK_TIMEOUT){
        std::cerr << "WARNING: Updating volume images timed out after 5 seconds!" << std::endl;
        vkDestroyFence(device, timeoutFence, nullptr);
        vkResetCommandPool(device, commandPool, 0);
        throw std::runtime_error("Updating volume images timed out");
    }
    else if(result != VK_SUCCESS){
        vkDestroyFence(device, timeoutFence, nullptr);
        throw std::runtime_error("Failed to wait for compute completion: " + std::to_string(result));
    }
    vkDestroyFence(device, timeoutFence, nullptr);
}
VkSemaphore VolumeSimulator::dispatchKernel(const std::string &kernelName, glm::uvec3 gridSize, const ComputePushConstants &pushConstants){
    int numCells = gridSize.x * gridSize.y * gridSize.z;
    VkCommandBuffer commandBuffer = computeCommandBuffers[currentFrame];
    vkResetCommandBuffer(commandBuffer, 0);
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if(vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS)
        throw std::runtime_error("Failed to begin command buffer");
    if(kernels.find(kernelName) == kernels.end())
        throw std::runtime_error("Kernel not found: " + kernelName);
    const ComputeKernel &kernel = kernels[kernelName];
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, kernel.pipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, kernel.pipelineLayout, 0, 1, descriptorSets.data(), 0, nullptr);
    if(sizeof(pushConstants) > 0)
        vkCmdPushConstants(commandBuffer, kernel.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ComputePushConstants), &pushConstants);
    glm::uvec3 numGroups = (gridSize + kernel.workgroupSize - 1u) / kernel.workgroupSize;
    vkCmdDispatch(commandBuffer, numGroups.x, numGroups.y, numGroups.z);
    if(kernel.needsBarrier) addMemoryBarrier(commandBuffer);
    vkEndCommandBuffer(commandBuffer);
    VkFence timeoutFence;
    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if(vkCreateFence(device, &fenceInfo, nullptr, &timeoutFence) != VK_SUCCESS)
        throw std::runtime_error("Failed to create timeout fence");
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &computeFinishedSemaphores[currentFrame];
    if(vkQueueSubmit(computeQueue, 1, &submitInfo, timeoutFence) != VK_SUCCESS){
        vkDestroyFence(device, timeoutFence, nullptr);
        throw std::runtime_error("Failed to submit compute command buffer");
    }
    const uint64_t TIMEOUT_NANOSECONDS = 5000000000ULL;
    VkResult result = vkWaitForFences(device, 1, &timeoutFence, VK_TRUE, TIMEOUT_NANOSECONDS);
    if(result == VK_TIMEOUT){
        std::cerr << "WARNING: Compute kernel '" << kernelName << "' timed out after 5 seconds!" << std::endl;
        std::cerr << "Dispatch size: " << numGroups.x << "x" << numGroups.y << "x" << numGroups.z << " work groups" << std::endl;
        vkDestroyFence(device, timeoutFence, nullptr);
        vkResetCommandPool(device, commandPool, 0);
        throw std::runtime_error("Compute kernel '" + kernelName + "' timed out - possible infinite loop in shader");
    }
    else if(result != VK_SUCCESS){
        vkDestroyFence(device, timeoutFence, nullptr);
        throw std::runtime_error("Failed to wait for compute completion: " + std::to_string(result));
    }
    vkDestroyFence(device, timeoutFence, nullptr);
    return computeFinishedSemaphores[currentFrame];
}
void VolumeSimulator::swapPressureBuffers(){
    simulationMemory->swapIterationBuffers();
    updateDescriptorSetsWithBuffers();
}
void VolumeSimulator::copyFinalPressureToMain(){
    uint32_t totalCells = gridSizeX * gridSizeY * gridSizeZ;
    VkDeviceSize size = totalCells * sizeof(float);
    VkBuffer srcBuffer = simulationMemory->getPressureTemp(); 
    VkBuffer dstBuffer = simulationMemory->getPressure();
    VulkanCommandUtils::copyBuffer(device, commandPool, computeQueue, srcBuffer, dstBuffer, size);
}
float VolumeSimulator::computeResidualSum(){
    int numCells = gridSizeX * gridSizeY * gridSizeZ;
    VkDeviceSize bufferSize = numCells * sizeof(float);
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingMemory;
    simulationMemory->createStagingBuffer(bufferSize, stagingBuffer, stagingMemory);
    simulationMemory->copyBuffer(simulationMemory->getResidual(), stagingBuffer, bufferSize);
    void* data;
    vkMapMemory(device, stagingMemory, 0, bufferSize, 0, &data);   
    float* residuals = static_cast<float*>(data);
    float sum = 0.0f;
    for(int i = 0; i < numCells; i++) {
        sum += residuals[i];
    }
    vkUnmapMemory(device, stagingMemory);
    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingMemory, nullptr);
    return sum / numCells;
}
void VolumeSimulator::cleanup(){
    for(auto &[name, kernel] : kernels){
        if(kernel.shaderModule != VK_NULL_HANDLE)
            vkDestroyShaderModule(device, kernel.shaderModule, nullptr);
        if(kernel.pipeline != VK_NULL_HANDLE)
            vkDestroyPipeline(device, kernel.pipeline, nullptr);
        if(kernel.pipelineLayout != VK_NULL_HANDLE)
            vkDestroyPipelineLayout(device, kernel.pipelineLayout, nullptr);
    }
    kernels.clear();
    for(auto &semaphore : computeFinishedSemaphores){
        if(semaphore != VK_NULL_HANDLE)
            vkDestroySemaphore(device, semaphore, nullptr);
    }
    computeFinishedSemaphores.clear();
    computeCommandBuffers.clear();
    descriptorSets.clear();
    sharedDescriptorSetLayout = VK_NULL_HANDLE;
    sharedDescriptorPool = VK_NULL_HANDLE;
    simulationMemory->markDeviceDestroyed();
}
void VolumeSimulator::cleanupSimulation(){
    simulationMemory->cleanup();
}
void VolumeSimulator::createKernelPipelineLayout(ComputeKernel &kernel){
    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(ComputePushConstants);
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &sharedDescriptorSetLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
    if(vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &kernel.pipelineLayout) != VK_SUCCESS)
        throw std::runtime_error("Failed to create compute pipeline layout");
}
void VolumeSimulator::createKernelPipeline(ComputeKernel &kernel){
    VkPipelineShaderStageCreateInfo shaderStageInfo{};
    shaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    shaderStageInfo.module = kernel.shaderModule;
    shaderStageInfo.pName = "main";
    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = shaderStageInfo;
    pipelineInfo.layout = kernel.pipelineLayout;
    if(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &kernel.pipeline) != VK_SUCCESS)
        throw std::runtime_error("Failed to create compute pipeline");
}
void VolumeSimulator::addMemoryBarrier(VkCommandBuffer commandBuffer){
    VkMemoryBarrier memoryBarrier{};
    memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memoryBarrier, 0, nullptr, 0, nullptr);
}
void VolumeSimulator::createSharedDescriptorSets(){
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = sharedDescriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &sharedDescriptorSetLayout;
    descriptorSets.resize(1);
    if(vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.data()) != VK_SUCCESS)
        throw std::runtime_error("Failed to allocate descriptor sets");
}
VkShaderModule VolumeSimulator::createShaderModule(const std::vector<char>& code){
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());
    VkShaderModule shaderModule;
    if(vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS)
        throw std::runtime_error("Failed to create shader module");
    return shaderModule;
}
VkCommandBuffer VolumeSimulator::beginSingleTimeCommands(){
    return VulkanCommandUtils::beginSingleTimeCommands(device, commandPool);
}
void VolumeSimulator::endSingleTimeCommands(VkCommandBuffer commandBuffer){
    VulkanCommandUtils::endSingleTimeCommands(device, commandPool, computeQueue, commandBuffer);
}
std::vector<char> VolumeSimulator::readFile(const std::string& filename){
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if(!file.is_open())
        throw std::runtime_error("Failed to open file");
    size_t fileSize = (size_t) file.tellg();
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();
    return buffer;
}