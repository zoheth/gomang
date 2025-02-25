#pragma once

namespace gomang
{
enum class MemoryLayout
{
	kNHWC,
	kNCHW,
	kNC4HW4
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

inline size_t getDataTypeSize(DataType type)
{
	switch (type)
	{
		case DataType::kFLOAT32:
			return 4;
		case DataType::kFLOAT16:
			return 2;
		case DataType::kINT8:
			return 1;
		case DataType::kINT32:
			return 4;
		default:
			return 0;
	}
}

inline const char *getDataTypeName(const DataType type)
{
	switch (type)
	{
		case DataType::kFLOAT32:
			return "FLOAT32";
		case DataType::kFLOAT16:
			return "FLOAT16";
		case DataType::kINT8:
			return "INT8";
		case DataType::kINT32:
			return "INT32";
		default:
			return "UNKNOWN";
	}
}

inline const char *getMemoryLayoutName(const MemoryLayout layout)
{
	switch (layout)
	{
		case MemoryLayout::kNHWC:
			return "NHWC";
		case MemoryLayout::kNCHW:
			return "NCHW";
		default:
			return "UNKNOWN";
	}
}

inline const char *getMemoryTypeName(const MemoryType type)
{
	switch (type)
	{
		case MemoryType::kCPU:
			return "CPU";
		case MemoryType::kGPU:
			return "GPU";
		case MemoryType::kCPU_PINNED:
			return "CPU_PINNED";
		default:
			return "UNKNOWN";
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