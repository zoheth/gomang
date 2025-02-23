#pragma once

namespace gomang
{
enum class MemoryLayout
{
	kNHWC,
	kNCHW
};

enum class MemoryType
{
	kCPU,
	kGPU,
	kCPU_PINNED
};

enum class DataType
{
	kFLOAT32,
	kFLOAT16,
	kINT8,
	kINT32
};

inline size_t getDataTypeSize(DataType type) {
	switch (type) {
		case DataType::kFLOAT32: return 4;
		case DataType::kFLOAT16: return 2;
		case DataType::kINT8:    return 1;
		case DataType::kINT32:   return 4;
		default: return 0;
	}
}

class IMemoryAllocator
{
  public:
	virtual void *allocate(size_t size, MemoryType type) = 0;
	virtual void  deallocate(void *ptr, MemoryType type) = 0;
	virtual ~IMemoryAllocator()                          = default;
};
}        // namespace gomang