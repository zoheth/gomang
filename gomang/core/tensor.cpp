#include "tensor.h"

#include <iostream>
#include <numeric>
#include <utility>

namespace gomang
{

size_t TensorDesc::getElementsCount() const
{
	return std::reduce(shape.begin(), shape.end(),
	                   1ULL, std::multiplies<>());
}
size_t TensorDesc::calculateSize() const
{
	size_t element_size = getDataTypeSize(data_type);
	size_t num_elements = 0;

	switch (layout)
	{
		case MemoryLayout::kNCHW:
		case MemoryLayout::kNHWC:
			num_elements = std::reduce(shape.begin(), shape.end(),
			                           1ULL, std::multiplies<>());
			break;

			// case MemoryLayout::kNC4HW4: {
			// 	// 假设shape顺序为NCHW
			// 	if (shape.size() < 4) return 0;
			// 	size_t N = shape[0];
			// 	size_t C = ((shape[1] + 3) / 4) * 4;
			// 	size_t H = shape[2];
			// 	size_t W = shape[3];
			// 	num_elements = N * C * H * W;
			// 	break;
			// }
	}

	size_t total_size = num_elements * element_size;
	return (total_size + alignment - 1) & ~(alignment - 1);
}
void TensorDesc::print() const
{
	std::cout << "Tensor: " << name
	          << " | Shape: [";
	for (size_t i = 0; i < shape.size(); ++i)
	{
		std::cout << shape[i];
		if (i < shape.size() - 1)
			std::cout << ",";
	}
	std::cout << "] | " << getDataTypeName(data_type)
	          // << " | Layout: " << getMemoryLayoutName(layout)
	          << " | " << getMemoryTypeName(mem_type)
	          << std::endl;
}

Tensor::Tensor(TensorDesc desc, IMemoryAllocator *allocator) :
    desc_(std::move(desc)), allocator_(allocator)
{
	allocaMemory();
}
Tensor::~Tensor()
{
	if (allocator_ && data_)
	{
		allocator_->deallocate(data_, desc_.mem_type);
	}
}
const void *Tensor::data() const
{
	return data_;
}
void *Tensor::data()
{
	return data_;
}
const TensorDesc &Tensor::desc() const
{
	return desc_;
}

size_t Tensor::size() const
{
	return size_;
}
}        // namespace gomang